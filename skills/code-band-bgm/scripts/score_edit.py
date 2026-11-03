"""定向改谱与保留证明：缩短、循环、加声部，以及两份乐谱（或冻结事件）的逐声部差异。

每个改谱命令都只写新文件（已存在则拒绝），并在输出里附带 diff：哪些声部逐音一致、哪些变了、变了什么。
差异按谱面字段比对（起点 tick、时值、音高、力度、奏法、演奏控制）；给了事件文件时再按渲染帧比对。
改谱不改 score_id：采样轮换与人性化的随机流以 score_id 为命名空间，改了 id 会让未改动的声部换录音。
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_tools import tick_frame  # noqa: E402

SCHEMA_VERSION = "score_edit_v1"
NOTE_FIELDS = ("onset_tick", "duration_ticks", "pitch", "velocity", "articulation", "performance")
# program/bank/采样 ID 也要比：采样轮换换了录音时，起点与音高都不变，但声音变了
EVENT_FIELDS = ("onset_frame", "gate_frames", "midi", "velocity", "articulation", "bank", "program", "soundfont_sample_ids")


def digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def note_key(note):
    return {field: note.get(field) for field in NOTE_FIELDS}


def diff_scores(before, after):
    """逐声部比对。返回 {part_id: {status, before_notes, after_notes, changed_fields...}} 与全局字段变化。"""
    old = {p["part_id"]: p for p in before["parts"]}
    new = {p["part_id"]: p for p in after["parts"]}
    parts = {}
    for part_id in sorted(set(old) | set(new)):
        if part_id not in new:
            parts[part_id] = {"status": "removed", "before_notes": len(old[part_id]["notes"])}
            continue
        if part_id not in old:
            parts[part_id] = {"status": "added", "after_notes": len(new[part_id]["notes"])}
            continue
        a = {n["note_id"]: note_key(n) for n in old[part_id]["notes"]}
        b = {n["note_id"]: note_key(n) for n in new[part_id]["notes"]}
        same = [k for k in a if k in b and a[k] == b[k]]
        changed = {}
        for k in a:
            if k in b and a[k] != b[k]:
                for field in NOTE_FIELDS:
                    if a[k][field] != b[k][field]:
                        changed[field] = changed.get(field, 0) + 1
        mix = {f: (old[part_id].get(f), new[part_id].get(f)) for f in ("gain_db", "pan", "resource_binding", "articulation_map")
               if old[part_id].get(f) != new[part_id].get(f)}
        removed = sorted(set(a) - set(b))
        added = sorted(set(b) - set(a))
        identical = len(same) == len(a) == len(b) and not mix
        parts[part_id] = {"status": "identical" if identical else "changed", "before_notes": len(a), "after_notes": len(b),
                          "identical_notes": len(same), "removed_notes": len(removed), "added_notes": len(added),
                          "changed_note_fields": changed, "mix_changes": {k: {"before": v[0], "after": v[1]} for k, v in mix.items()},
                          "notes_sha256": {"before": digest([note_key(n) for n in old[part_id]["notes"]]),
                                           "after": digest([note_key(n) for n in new[part_id]["notes"]])}}
    globals_changed = {key: {"before": before.get(key), "after": after.get(key)}
                       for key in ("timeline", "render", "randomness", "mix") if before.get(key) != after.get(key)}
    return {"parts": parts, "global_changes": globals_changed}


def diff_events(before, after):
    """冻结事件逐声部比对（渲染帧级）。"""
    def table(doc):
        native = doc.get("native_events", doc)
        rows = {}
        for event in native["events"]:
            rows.setdefault(event["part_id"], []).append(json.dumps([event.get(f) for f in EVENT_FIELDS]))
        return {k: sorted(v) for k, v in rows.items()}
    a, b = table(before), table(after)
    return {part: {"identical": a.get(part) == b.get(part), "before_events": len(a.get(part, [])), "after_events": len(b.get(part, []))}
            for part in sorted(set(a) | set(b))}


def check_preserved(report, preserve):
    failures = [p for p in preserve if report["parts"].get(p, {}).get("status") != "identical"]
    return {"requested": list(preserve), "all_identical": not failures, "not_identical": failures}


def _cut(score, end_tick, guard=30, shortest=60):
    """截断点前留 guard tick：引擎把门控对齐到 64 帧网格，贴着结尾的音取整后会越界。截完不足 shortest 的音丢弃。"""
    result = copy.deepcopy(score)
    for part in result["parts"]:
        kept = []
        for note in part["notes"]:
            if note["onset_tick"] >= end_tick - shortest - guard:
                continue
            if note["onset_tick"] + note["duration_ticks"] > end_tick - guard:
                note["duration_ticks"] = end_tick - guard - note["onset_tick"]
                performance = note.get("performance") or {}
                for name in ("pitch_curve", "expression_curve"):
                    if name in performance:
                        performance[name] = [p for p in performance[name] if p["at_tick"] <= note["duration_ticks"]] or performance[name][:1]
                vibrato = performance.get("vibrato")
                if vibrato and vibrato.get("start_tick", 0) >= note["duration_ticks"]:
                    del performance["vibrato"]
            kept.append(note)
        part["notes"] = kept
    result["parts"] = [p for p in result["parts"] if p["notes"]]
    sections = []
    for section in result["sections"]:
        if section["start_tick"] < end_tick:
            section["end_tick"] = min(section["end_tick"], end_tick)
            sections.append(section)
    result["sections"] = sections
    result["timeline"]["music_end_tick"] = end_tick
    return result


def bar_tick(score, bars):
    meter = score["timeline"]["meter_map"]
    if len(meter) != 1:
        raise SystemExit("只支持单一拍号的乐谱")
    m = meter[0]
    return bars * m["numerator"] * score["timeline"]["ppq"] * 4 // m["denominator"]


def shorten(score, bars, tail_seconds=2.0):
    end = bar_tick(score, bars)
    if end >= score["timeline"]["music_end_tick"]:
        raise SystemExit("缩短后的长度不短于原曲")
    result = _cut(score, end)
    rate = result["render"]["sample_rate_hz"]
    render = result["render"]
    render.pop("target_file_frames", None)
    render.update(duration_mode="musical_length", tail_policy="append", tail_frames=round(tail_seconds * rate),
                  end_fade_frames=min(render["end_fade_frames"], round(tail_seconds * rate)), loop={"enabled": False})
    return result


def make_loop(score, bars, bpm=None, tail_seconds=2.0):
    end = bar_tick(score, bars)
    result = _cut(score, end) if end < score["timeline"]["music_end_tick"] else copy.deepcopy(score)
    if bpm is not None:
        result["timeline"]["tempo_map"] = [{"start_tick": 0, "bpm": {"numerator": bpm, "denominator": 1}}]
    rate = result["render"]["sample_rate_hz"]
    frames = tick_frame(end, result["timeline"], rate)
    if frames % 64:
        raise SystemExit(f"循环周期 {frames} 帧不是 64 的整数倍；换小节数或用 --bpm（例如 48 kHz 下 90、96、120 BPM 的整小节都满足）")
    render = result["render"]
    render.pop("target_file_frames", None)
    render.update(duration_mode="musical_length", tail_policy="wrap_into_loop", tail_frames=round(tail_seconds * rate),
                  end_fade_frames=0,
                  loop={"enabled": True, "start_tick": 0, "end_tick": end, "boundary_method": "tail_wrap_crossfade",
                        "crossfade_frames": 0, "notes": "尾音折回开头；首尾衔接由引擎三周期预热处理。"})
    return result


def add_part(score, donor, part_id):
    parts = {p["part_id"]: p for p in donor["parts"]}
    if part_id not in parts:
        raise SystemExit(f"来源乐谱没有声部 {part_id}")
    if any(p["part_id"] == part_id for p in score["parts"]):
        raise SystemExit(f"目标乐谱已有声部 {part_id}")
    if donor["timeline"] != score["timeline"]:
        raise SystemExit("来源与目标的时间轴不同，不能直接并入")
    result = copy.deepcopy(score)
    result["parts"].append(copy.deepcopy(parts[part_id]))
    return result


def write_new(path, data):
    path = Path(path)
    if path.exists():
        raise SystemExit(f"拒绝覆盖已存在的文件：{path}")
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.rename(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    d = sub.add_parser("diff", help="两份乐谱逐声部差异；可附事件文件做帧级比对")
    d.add_argument("--before", required=True)
    d.add_argument("--after", required=True)
    d.add_argument("--before-events")
    d.add_argument("--after-events")
    d.add_argument("--preserve", nargs="*", default=[])
    for name, text in (("shorten", "保留前 N 小节，截断跨界音并加尾音"), ("loop", "取前 N 小节做无缝循环")):
        p = sub.add_parser(name, help=text)
        p.add_argument("--score", required=True)
        p.add_argument("--bars", type=int, required=True)
        p.add_argument("--out", required=True)
        p.add_argument("--tail-seconds", type=float, default=2.0)
        if name == "loop":
            p.add_argument("--bpm", type=int)
    a = sub.add_parser("add-part", help="从同一时间轴的另一份乐谱并入一个声部")
    a.add_argument("--score", required=True)
    a.add_argument("--donor", required=True)
    a.add_argument("--part", required=True)
    a.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    load = lambda p: json.loads(Path(p).read_text(encoding="utf-8"))  # noqa: E731
    if args.command == "diff":
        report = diff_scores(load(args.before), load(args.after))
        if args.before_events and args.after_events:
            report["events"] = diff_events(load(args.before_events), load(args.after_events))
        report["preserve_check"] = check_preserved(report, args.preserve)
        report["schema_version"] = SCHEMA_VERSION
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["preserve_check"]["all_identical"] else 3
    score = load(args.score)
    if args.command == "shorten":
        result = shorten(score, args.bars, args.tail_seconds)
    elif args.command == "loop":
        result = make_loop(score, args.bars, args.bpm, args.tail_seconds)
    else:
        result = add_part(score, load(args.donor), args.part)
    write_new(args.out, result)
    report = diff_scores(score, result)
    print(json.dumps({"schema_version": SCHEMA_VERSION, "status": "通过", "out": args.out,
                      "parts": {k: {f: v[f] for f in ("status", "before_notes", "after_notes", "identical_notes", "removed_notes",
                                                          "added_notes", "changed_note_fields") if f in v}
                                for k, v in report["parts"].items()},
                      "global_changes": sorted(report["global_changes"])}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
