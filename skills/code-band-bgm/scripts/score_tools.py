"""乐谱契约、精确时间、分流随机与可追踪修改。"""
from __future__ import annotations

import copy
import hashlib
import json
from fractions import Fraction
from pathlib import Path

from jsonschema import Draft202012Validator

VERSION = "0.1.0"
ROOT = Path(__file__).resolve().parents[1]


class ScoreError(ValueError):
    def __init__(self, code, message, details=None):
        super().__init__(message)
        self.code, self.message, self.details = code, message, details or {}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(1048576), b""):
            h.update(data)
    return h.hexdigest()


def load(path):
    def reject(value):
        raise ScoreError("nonfinite_json", "JSON 不允许 NaN 或无穷值")
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject)


def schema_check(doc, kind):
    schema = load(ROOT / "assets" / "schemas" / (kind + ".schema.json"))
    errors = sorted(Draft202012Validator(schema).iter_errors(doc), key=lambda e: str(e.path))
    if errors:
        raise ScoreError("schema_invalid", "结构校验失败", {"errors": [
            {"field": "/" + "/".join(map(str, e.path)), "reason": "字段类型、范围或必填约束不符合契约", "rule": e.validator}
            for e in errors[:30]]})


def require(condition, field, reason):
    if not condition:
        raise ScoreError("semantic_invalid", reason, {"field": field})


def unique(values, field):
    require(len(values) == len(set(values)), field, "标识必须唯一")


def round_fraction(value):
    sign = -1 if value < 0 else 1
    value = abs(value)
    return sign * ((2 * value.numerator + value.denominator) // (2 * value.denominator))


def tick_seconds(tick, timeline):
    total = Fraction(0)
    tempo = timeline["tempo_map"]
    for i, item in enumerate(tempo):
        start = item["start_tick"]
        end = min(tick, tempo[i + 1]["start_tick"] if i + 1 < len(tempo) else tick)
        if end > start:
            bpm = item["bpm"]
            total += Fraction((end - start) * 60 * bpm["denominator"], timeline["ppq"] * bpm["numerator"])
        if end >= tick:
            break
    return total


def tick_frame(tick, timeline, rate):
    return round_fraction(tick_seconds(tick, timeline) * rate)


def random_int(seed, namespace, event_id, size, counter=0):
    require(size > 0, "randomness", "抽样范围必须大于零")
    limit = (2**64 // size) * size
    while True:
        fields = ["code-band-bgm/v1", str(seed), namespace, event_id, str(counter)]
        require(all("\0" not in f for f in fields), "randomness", "随机标识不能包含零字节")
        n = int.from_bytes(hashlib.sha256("\0".join(fields).encode()).digest()[:8], "big")
        if n < limit:
            return n % size
        counter += 1


def triangular(seed, namespace, event_id, maximum):
    # 独立计数器用于两个均匀抽样，避免不同用途共享状态。
    return random_int(seed, namespace, event_id, maximum + 1, 0) - random_int(seed, namespace, event_id, maximum + 1, 1000000)


def validate_capabilities(cap):
    schema_check(cap, "引擎能力")
    evid = [e["evidence_id"] for e in cap["evidence"]]
    unique(evid, "evidence")
    for group in ("execution", "timing", "performance", "mixing", "export", "reproducibility"):
        for key, feature in cap[group].items():
            require(set(feature["evidence_ids"]) <= set(evid), group + "/" + key, "能力引用的证据不存在")
            if cap["probe_state"] == "unprobed":
                require(feature["state"] == "unknown", group + "/" + key, "未探测能力必须保持未知")
    resources = cap["resources"]
    unique([r["instrument_id"] for r in resources], "resources")
    if resources:
        require(cap["resource_manifest_sha256"] == digest(resources), "resource_manifest_sha256", "资源清单摘要不匹配")
    for r in resources:
        require(r["pitch_range"]["min_midi"] <= r["pitch_range"]["max_midi"], "resources/pitch_range", "音域上下界倒置")
        unique([s["sample_id"] for s in r["sample_assets"]], "resources/sample_assets")
        for s in r["sample_assets"]:
            require(s["velocity_min"] <= s["velocity_max"], "sample_assets", "力度层上下界倒置")
    return cap


def validate_score(score):
    schema_check(score, "乐谱")
    t, r = score["timeline"], score["render"]
    end = t["music_end_tick"]
    require(int(score["randomness"]["master_seed"]) < 2**64, "randomness/master_seed", "种子超出无符号64位范围")
    for field in ("tempo_map", "meter_map"):
        ticks = [x["start_tick"] for x in t[field]]
        require(ticks[0] == 0 and all(a < b for a, b in zip(ticks, ticks[1:])) and ticks[-1] < end, "timeline/" + field, "时间表必须从零开始，严格递增且不超主体")
    for previous, current in zip(t["meter_map"], t["meter_map"][1:]):
        bar = Fraction(t["ppq"] * 4 * previous["numerator"], previous["denominator"])
        require(Fraction(current["start_tick"] - previous["start_tick"], 1) % bar == 0, "timeline/meter_map", "拍号变化必须位于旧拍号小节边界")
    unique([s["section_id"] for s in score["sections"]], "sections")
    cursor = 0
    for s in score["sections"]:
        require(s["start_tick"] == cursor and s["end_tick"] > cursor, "sections", "段落必须顺序连续覆盖主体，不允许重叠或空隙")
        cursor = s["end_tick"]
    require(cursor == end, "sections", "段落末端必须等于主体结束位置")
    part_ids = [p["part_id"] for p in score["parts"]]
    unique(part_ids, "parts")
    for p in score["parts"]:
        field = "parts/" + p["part_id"]
        unique([n["note_id"] for n in p["notes"]], field + "/notes")
        groups = {}
        fixed_notes = {}
        for n in p["notes"]:
            require(n["onset_tick"] + n["duration_ticks"] <= end, field + "/" + n["note_id"], "音符逻辑结束超出主体")
            require(n["articulation"] in p["articulation_map"], field, "音符引用的演奏法不存在")
            if "performance" in n:
                from note_expression import validate_performance
                try:
                    validate_performance(n["performance"], n["duration_ticks"])
                except ValueError as exc:
                    require(False, field + "/" + n["note_id"] + "/performance", str(exc))
            if "humanize_group_id" in n:
                group = n["humanize_group_id"]
                groups.setdefault(group, n["onset_tick"])
                require(groups[group] == n["onset_tick"], field, "共同微偏移组必须在同一tick起音")
            if "performance" not in n and p["articulation_map"][n["articulation"]].get("engine_articulation_id") in (None, "normal") and (not p["humanize"]["enabled"] or n["timing_locked"]):
                pitch = n["pitch"]
                key = (pitch["kind"], pitch.get("midi") if pitch["kind"] == "pitched" else pitch.get("hit_key"))
                fixed_notes.setdefault(key, []).append(n)
        for notes in fixed_notes.values():
            previous_end = -1
            for n in sorted(notes, key=lambda item: (item["onset_tick"], item["note_id"])):
                require(n["onset_tick"] >= previous_end, field + "/notes", "同声部同音高门控重叠，当前引擎无法独立停止")
                previous_end = n["onset_tick"] + n["duration_ticks"]
    unique([e["effect_id"] for e in score["mix"]["effects"]], "mix/effects")
    for e in score["mix"]["effects"]:
        require(set(e["target_part_ids"]) <= set(part_ids), "mix/effects", "效果引用了不存在的声部")
        if e['kind'] == 'chorus':
            from chorus_effect import validate_chorus_effects
            try:
                validate_chorus_effects(score['mix']['effects'], set(part_ids), r['loop']['enabled'])
            except ValueError as exc:
                require(False, 'mix/effects', str(exc))
        if e['kind'] == 'duck':
            from ducking_effect import duck_parameters
            try:
                duck_parameters(e, set(part_ids))
            except ValueError as exc:
                require(False, 'mix/effects', str(exc))
    if any(e['kind'] == 'eq' for e in score['mix']['effects']):
        from delay_effect import validate_effects
        try:
            validate_effects(score['mix']['effects'], set(part_ids), r['loop']['enabled'],
                             sample_rate=r['sample_rate_hz'])
        except ValueError as exc:
            require(False, 'mix/effects', str(exc))
    frames = tick_frame(end, t, r["sample_rate_hz"])
    total = frames + (r["tail_frames"] if r["tail_policy"] == "append" else 0)
    if r["duration_mode"] == "exact_file":
        require(total == r["target_file_frames"], "render/target_file_frames", "主体与尾音之和不等于目标总帧数")
    require(r["end_fade_frames"] <= total, "render/end_fade_frames", "淡出超过文件总长")
    loop = r["loop"]
    if loop["enabled"]:
        require(loop["end_tick"] == end and loop["crossfade_frames"] < total, "render/loop", "循环末端或交叠帧数不符合约定")
        expected = "wrap_into_loop" if loop["boundary_method"] == "tail_wrap_crossfade" else "none"
        require(r["tail_policy"] == expected, "render/loop", "循环方法与尾音策略冲突")
        require(loop["boundary_method"] != "validated_no_tail" or loop["crossfade_frames"] == 0, "render/loop", "无尾音循环不能配置交叠")
    policy = score["intent"]["reference_policy"]
    require(bool(policy["source_ids"]) == (policy["mode"] == "abstract_features_only"), "intent/reference_policy", "参考模式与来源引用不一致")
    return {"music_frames": frames, "total_frames": total,
            "render_ready": score["document_state"] != "structure_example" and all(p["resource_binding"]["state"] == "resolved" for p in score["parts"])}


def require_shaping(part, cap):
    from sound_shaping import validate_shaping
    try:
        validate_shaping(part.get("sound_shaping", {}))
    except ValueError as exc:
        require(False, "parts/sound_shaping", str(exc))
    if part.get("sound_shaping"):
        require(cap["performance"].get("sound_shaping", {}).get("state") == "supported",
                "parts/sound_shaping", "声音塑形尚未实际探测，不能静默忽略")


def articulation_preset(resource, articulation_id, bank, program):
    """奏法落到同库真实预设；resource未登记时只有normal可用。"""
    spec = resource.get("articulation_programs", {}).get(articulation_id)
    if spec is None:
        return bank, [program]
    return spec["bank"], list(spec["programs"])


def round_robin(notes, resource, part_id, score_id, seed, articulation_of, bank, program):
    """采样轮换独立随机流：按起音顺序逐音高轮转，改混音或改别的声部都不重抽。"""
    choice = {}
    counters = {}
    for n in sorted(notes, key=lambda item: (item["onset_tick"], item["note_id"])):
        articulation = articulation_of(n)
        _, programs = articulation_preset(resource, articulation, bank, program)
        if len(programs) == 1:
            choice[n["note_id"]] = programs[0]
            continue
        key = (articulation, n["pitch"].get("midi", n["pitch"].get("hit_key")))
        if key not in counters:
            counters[key] = random_int(seed, "sample_selection", f"{score_id}/{part_id}/{key[0]}/{key[1]}", len(programs))
        choice[n["note_id"]] = programs[counters[key] % len(programs)]
        counters[key] += 1
    return choice


def resolved_performance(n, on, off, timeline, rate):
    """谱面曲线转冻结控制；编译与冻结校验共用，保证两边算出同一结果。"""
    from note_expression import resolve_performance
    pitch = n["pitch"]
    cents = pitch.get("cents", 0) if pitch["kind"] == "pitched" else 0
    return resolve_performance(n.get("performance", {}), cents, n["onset_tick"], n["duration_ticks"],
                               off - on, lambda tick: tick_frame(tick, timeline, rate), rate)


def sample_start_frames(performance, sample_ids, inventory):
    """毫秒按实际采样的采样率换算，同音多采样率不一致时拒绝。"""
    if "sample_start_ms" not in performance:
        return performance
    rates = {s["sample_rate_hz"] for s in inventory["samples"] if s["sample_id"] in sample_ids}
    require(len(rates) == 1, "performance/sample_start_ms", "命中采样的采样率不一致，不能换算起点偏移")
    performance["sample_start_frames"] = round(performance["sample_start_ms"] * rates.pop() / 1000)
    return performance


def compile_score(score, cap, asset_sha256, soundfont_inventory=None, additional_inventories=None):
    summary = validate_score(score)
    validate_capabilities(cap)
    require(summary["render_ready"], "document_state", "结构示例或未绑定乐谱不能正式编译")
    require(score["randomness"]["sample_selection"] == "seeded_choice", "randomness/sample_selection", "当前SoundFont适配器不支持可控采样轮换，不能忽略round-robin请求")
    require(cap["engine"] is not None, "engine", "没有经过探测的引擎")
    engine = cap["engine"]
    require(engine["adapter_id"] == "fluid_memory", "engine/adapter_id", "当前只实现了本地FluidSynth适配器")
    for group, keys in {"execution": ["offline_render"], "export": ["wav", "exact_frame_count"], "performance": ["engine_humanize_bypass", "per_note_velocity"], "mixing": ["part_gain", "part_pan"]}.items():
        for key in keys:
            require(cap[group][key]["state"] == "supported", group + "/" + key, "必要能力尚未验证")
    r, t = score["render"], score["timeline"]
    limits = cap["format_limits"]
    for key, actual in (("sample_rates_hz", r["sample_rate_hz"]), ("sample_formats", r["sample_format"]), ("channel_counts", r["channels"])):
        require(actual in limits[key], "render/" + key, "请求格式不在已确认支持范围")
    require(summary["total_frames"] <= limits["max_render_frames"], "render", "超过当前渲染帧数预算")
    require(not r["loop"]["enabled"] or cap["export"]["loop_tail_wrap"]["state"] == "supported", "render/loop", "循环能力未验证")
    if r["loop"]["enabled"]:
        require(r["tail_policy"] == "wrap_into_loop" and r["tail_frames"] >= 64 and r["loop"]["crossfade_frames"] == 0, "render/loop", "当前循环采用三周期预热衔接，交叠长度须为零并声明尾音预算")
        require(summary["total_frames"] % 64 == 0, "render/loop", "当前循环周期必须是64帧整数倍")
    require(soundfont_inventory is not None, "assets", "缺少真实SoundFont区域映射，不能仅凭音色名称编译")
    require(soundfont_inventory["source"]["sha256"] == asset_sha256, "assets", "SoundFont映射与实际库哈希不匹配")
    inventories = {asset_sha256: soundfont_inventory}
    for sha256, extra in (additional_inventories or {}).items():
        require(extra["source"]["sha256"] == sha256, "assets", "额外音色库映射与实际文件哈希不匹配")
        require(sha256 not in inventories or inventories[sha256] == extra, "assets", "同一音色库哈希对应冲突的映射")
        inventories[sha256] = extra
    zones = {sha256: {(p["bank"], p["program"]): p for p in mapping["presets"]}
             for sha256, mapping in inventories.items()}
    resources = {x["instrument_id"]: x for x in cap["resources"]}
    seed = score["randomness"]["master_seed"]
    events, parts, adjustments = [], [], []
    starts = {s["start_tick"] for s in score["sections"]}
    for p in score["parts"]:
        binding = p["resource_binding"]
        require(binding["engine_id"] == engine["engine_id"] and binding["capability_id"] == cap["capability_id"] and binding["mapping_version"] == engine["adapter_version"], "resource_binding", "绑定的引擎、能力或映射版本已变化")
        resource = resources.get(binding["instrument_id"])
        require(resource is not None and resource["availability"] == "available", "resource_binding", "请求的真实音色不存在")
        require(not resource["sample_assets"], "resource_binding", "当前SoundFont适配器不接受独立采样映射；不能忽略显式采样选择")
        prefix, bank, program, bank_hash = resource["engine_resource_id"].split(":")
        require(prefix == "sf" and bank_hash in zones, "resource_binding", "绑定的音色库未提供或能力探测后内容已改变")
        require((int(bank), int(program)) in zones[bank_hash], "resource_binding", "指定音色库不存在所绑预设")
        preset_zones = zones[bank_hash][(int(bank), int(program))]["mappings"]
        rr = round_robin(p["notes"], resource, p["part_id"], score["score_id"], seed,
                         lambda n: p["articulation_map"][n["articulation"]].get("engine_articulation_id", "normal"), int(bank), int(program))
        parts.append({"part_id": p["part_id"], "bank": int(bank), "program": int(program), "gain_db": p["gain_db"], "pan": p["pan"], "instrument_id": resource["instrument_id"], "resource_sha256": bank_hash})
        require_shaping(p, cap)
        if "sound_shaping" in p:
            parts[-1]["sound_shaping"] = copy.deepcopy(p["sound_shaping"])
        h = p["humanize"]
        locked_groups = {n.get("humanize_group_id", str(n["onset_tick"])) for n in p["notes"] if n["timing_locked"] or n["onset_tick"] in starts}
        part_events = []
        for n in p["notes"]:
            eid = "/".join([score["score_id"], p["part_id"], n["note_id"]])
            articulation = p["articulation_map"][n["articulation"]]
            require(articulation["state"] == "resolved" and articulation["engine_articulation_id"] in resource["articulation_ids"], "articulation_map", "演奏法没有真实绑定")
            articulation_id = articulation["engine_articulation_id"]
            require(articulation_id == "normal" or articulation_id in resource.get("articulation_programs", {}), "articulation", "奏法没有登记真实预设，不能冒充")
            event_bank, _ = articulation_preset(resource, articulation_id, int(bank), int(program))
            event_program = rr[n["note_id"]]
            needs_dedicated = "performance" in n or event_program != int(program) or articulation_id != "normal" or (n["pitch"]["kind"] == "pitched" and n["pitch"]["cents"] != 0)
            if needs_dedicated:
                require(cap["performance"].get("note_expression", {}).get("state") == "supported", "performance/note_expression", "逐音演奏控制尚未实际探测")
                require((event_bank, event_program) in zones[bank_hash], "articulation", "奏法预设在音色库中不存在")
            note_zones = zones[bank_hash][(event_bank, event_program)]["mappings"]
            pitch = n["pitch"]
            if pitch["kind"] == "pitched":
                midi = pitch["midi"]
                require(resource["pitch_range"]["min_midi"] <= midi <= resource["pitch_range"]["max_midi"], "pitch/midi", "音符超出已绑定音色的可用音区")
                require(int(bank) != 128, "pitch", "鼓组必须使用明确的打击乐键映射")
            else:
                require(pitch["hit_key"] in resource["unpitched_hits"], "pitch/hit_key", "鼓键没有真实映射")
                midi = int(resource["unpitched_hits"][pitch["hit_key"]])
            require(needs_dedicated or "playable_midi_keys" not in resource or midi in resource["playable_midi_keys"], "pitch/midi", "音符落在能力报告明确未开放的采样键空洞")
            require("gain_db" not in n and "pan" not in n, "notes", "当前版本只支持声部级增益和声像，不能忽略逐音符设置")
            group = n.get("humanize_group_id", str(n["onset_tick"]))
            offset_us = delta = 0
            if h["enabled"]:
                if group not in locked_groups:
                    timing_id = score["score_id"] + "/" + p["part_id"] + "/" + (group if h["timing"]["chord_policy"] == "shared_offset" else n["note_id"])
                    offset_us = triangular(seed, "humanize_timing", timing_id, h["timing"]["max_offset_us"])
                # 同声部共享力度偏差，保持乐谱重音相对次序。
                delta = triangular(seed, "humanize_velocity", score["score_id"] + "/" + p["part_id"], h["velocity"]["max_delta"])
            shift = round_fraction(Fraction(offset_us * r["sample_rate_hz"], 1000000))
            logical = tick_frame(n["onset_tick"], t, r["sample_rate_hz"])
            logical_end = tick_frame(n["onset_tick"] + n["duration_ticks"], t, r["sample_rate_hz"])
            on = round_fraction(Fraction(logical + shift, 64)) * 64
            off = round_fraction(Fraction(logical_end + shift, 64)) * 64
            require(on >= 0 and off > on and off <= summary["music_frames"], "notes/" + n["note_id"], "人性化或64帧调度产生越界/零时长；请调整乐谱")
            final_velocity = max(1, min(127, n["velocity"] + delta))
            matching_zones = [z for z in note_zones if z["available_in_file"] and z["key_range"][0] <= midi <= z["key_range"][1] and z["velocity_range"][0] <= final_velocity <= z["velocity_range"][1]]
            require(bool(matching_zones), "notes/" + n["note_id"], "最终音高与力度在真实采样映射中没有可发声区域")
            event = {"event_id": eid, "note_id": n["note_id"], "part_id": p["part_id"], "onset_frame": on, "gate_frames": off - on, "midi": midi, "velocity": final_velocity, "articulation": "normal", "timing_offset_frames": shift, "quantization_frames": on - logical - shift, "notated_onset_tick": n["onset_tick"], "soundfont_zone_ids": [z["instrument_zone_id"] for z in matching_zones], "soundfont_sample_ids": sorted({z["sample_id"] for z in matching_zones})}
            if needs_dedicated:
                event.update(articulation=articulation_id, bank=event_bank, program=event_program)
                performance = resolved_performance(n, on, off, t, r["sample_rate_hz"])
                if "sample_start_ms" in performance:
                    performance = sample_start_frames(performance, set(event["soundfont_sample_ids"]), inventories[bank_hash])
                event["performance"] = performance
            part_events.append(event)
        ordered = sorted(part_events, key=lambda x: (x["notated_onset_tick"], x["event_id"]))
        for a, b in zip(ordered, ordered[1:]):
            require(a["onset_frame"] <= b["onset_frame"], "humanize", "微偏移造成声部顺序翻转")
        for midi in {e["midi"] for e in part_events}:
            previous_end = -1
            for e in sorted((e for e in part_events if e["midi"] == midi and "performance" not in e), key=lambda e: e["onset_frame"]):
                require(e["onset_frame"] >= previous_end, "notes", "同声部同音高门控重叠，当前引擎无法独立停止")
                previous_end = e["onset_frame"] + e["gate_frames"]
        events.extend(part_events)
    from delay_effect import validate_effects
    try:
        validate_effects(score["mix"]["effects"], {p["part_id"] for p in score["parts"]}, r["loop"]["enabled"],
                         sample_rate=r['sample_rate_hz'])
    except ValueError as exc:
        require(False, "mix/effects", str(exc))
    for effect in score["mix"]["effects"]:
        require(cap["mixing"].get(effect["kind"], {}).get("state") == "supported", "mix/effects", "效果没有对应的已验证引擎能力")
        if effect['kind'] == 'eq':
            require('ffmpeg_peaking_cut_runtime' in cap['mixing']['eq']['evidence_ids'], 'mix/effects', 'eq缺少真实频响探测证据')
        if effect["kind"] == "duck" and "trigger_part_ids" in effect["parameters"]:
            require("explicit_duck_multi_trigger" in cap["mixing"]["duck"]["evidence_ids"], "mix/effects", "多触发duck缺少已验证能力证据，不能降级为单触发")
            require(all(any(e["part_id"] == p for e in events) for p in effect["parameters"]["trigger_part_ids"]), "mix/effects", "多触发duck每个来源须有真实触发事件")
    compiled = {"schema_version": "1.0.0", "compiler_version": VERSION, "score_id": score["score_id"], "score_sha256": digest(score), "capability_sha256": digest(cap), "engine": engine, "asset_sha256": asset_sha256, "seed": seed, "sample_rate_hz": r["sample_rate_hz"], "total_frames": summary["total_frames"], "parts": parts, "events": sorted(events, key=lambda x: (x["onset_frame"], x["event_id"])), "mix": copy.deepcopy(score["mix"]), "render": copy.deepcopy(r), "diagnostics": {"schedule_grid_frames": 64, "maximum_rounding_error_frames": 32, "sample_selection": "SoundFont内部固定键域与力度层；无独立轮换控制", "repairs": adjustments}}
    used = sorted({p["resource_sha256"] for p in parts})
    if used != [asset_sha256]:
        compiled["used_asset_sha256"] = used
    return compiled


def validate_frozen(frozen, score, cap, asset_sha256, available_asset_sha256=None, soundfont_inventories=None):
    """重用事件、不重抽随机流；CLI另提供实际库映射以核区域/力度/采样身份。"""
    summary = validate_score(score)
    validate_capabilities(cap)
    require(summary["render_ready"], "score", "冻结事件的原乐谱尚未解析资源")
    require(frozen["score_sha256"] == digest(score) and frozen["asset_sha256"] == asset_sha256, "events", "冻结事件的乐谱或音色库已改变")
    require(frozen["seed"] == score["randomness"]["master_seed"] and frozen["score_id"] == score["score_id"], "events", "冻结事件的种子或作品身份不匹配")
    require(frozen["compiler_version"] == VERSION and frozen["schema_version"] == "1.0.0", "events", "冻结事件版本不受当前运行器支持")
    for key in ("engine_id", "version", "adapter_id", "adapter_version"):
        require(frozen["engine"][key] == cap["engine"][key], "events/engine", "冻结事件的音频引擎或适配版本已变化")
    require(frozen["render"] == score["render"] and frozen["mix"] == score["mix"] and frozen["sample_rate_hz"] == score["render"]["sample_rate_hz"] and frozen["total_frames"] == summary["total_frames"], "events", "冻结事件的输出或混音与乐谱不一致")
    from delay_effect import validate_effects
    try:
        validate_effects(score["mix"]["effects"], {p["part_id"] for p in score["parts"]}, score["render"]["loop"]["enabled"],
                         sample_rate=score['render']['sample_rate_hz'])
    except ValueError as exc:
        require(False, "events/mix/effects", str(exc))
    for effect in score["mix"]["effects"]:
        require(cap["mixing"].get(effect["kind"], {}).get("state") == "supported", "events/mix/effects", "冻结事件的效果当前能力未验证，不能绕过编译门禁")
        if effect['kind'] == 'eq':
            require('ffmpeg_peaking_cut_runtime' in cap['mixing']['eq']['evidence_ids'], 'events/mix/effects', '冻结eq缺少真实频响探测证据')
        if effect["kind"] == "duck" and "trigger_part_ids" in effect["parameters"]:
            require("explicit_duck_multi_trigger" in cap["mixing"]["duck"]["evidence_ids"], "events/mix/effects", "冻结事件的多触发duck缺少已验证能力证据")
            require(all(any(e["part_id"] == p for e in frozen["events"]) for p in effect["parameters"]["trigger_part_ids"]), "events/mix/effects", "冻结多触发duck每个来源须有真实触发事件")
    original_parts = {p["part_id"]: p for p in score["parts"]}
    available = {asset_sha256} | set(available_asset_sha256 or [])
    used = sorted({part["resource_sha256"] for part in frozen["parts"]})
    require(set(used) <= available, "events/assets", "冻结事件需要的额外音色库尚未提供或哈希已改变")
    require(frozen.get("used_asset_sha256", [asset_sha256]) == used, "events/assets", "冻结事件的音色库摘要清单与声部绑定不一致")
    require(len(frozen["parts"]) == len(original_parts), "events/parts", "冻结声部数量不匹配")
    unique([p["part_id"] for p in frozen["parts"]], "events/parts")
    resources = {r["instrument_id"]: r for r in cap["resources"]}
    actual_presets = {}
    for sha, mapping in (soundfont_inventories or {}).items():
        require(mapping["source"]["sha256"] == sha, "events/assets", "冻结事件校验的采样映射摘要不符")
        actual_presets[sha] = {(p["bank"], p["program"]): p["mappings"] for p in mapping["presets"]}
    for part in frozen["parts"]:
        p = original_parts[part["part_id"]]
        resource = resources[p["resource_binding"]["instrument_id"]]
        expected_resource = f"sf:{part['bank']}:{part['program']}:{part['resource_sha256']}"
        require(resource["availability"] == "available" and resource["engine_resource_id"] == expected_resource, "events/parts", "冻结声部资源已变化或不可用")
        require(part["gain_db"] == p["gain_db"] and part["pan"] == p["pan"], "events/parts", "冻结声部混音与乐谱不一致")
        require_shaping(p, cap)
        require(part.get("sound_shaping", {}) == p.get("sound_shaping", {}), "events/parts", "冻结声音塑形与原谱不一致")
    source_notes = {score["score_id"] + "/" + p["part_id"] + "/" + n["note_id"]: (p, n) for p in score["parts"] for n in p["notes"]}
    expected_ids = set(source_notes)
    actual_ids = [e["event_id"] for e in frozen["events"]]
    require(len(actual_ids) == len(expected_ids) and set(actual_ids) == expected_ids, "events", "冻结事件与谱面音符身份不匹配")
    for event in frozen["events"]:
        require(event["event_id"] == score["score_id"] + "/" + event["part_id"] + "/" + event["note_id"], "events", "冻结事件声部或音符引用不匹配")
        p, n = source_notes[event["event_id"]]
        pitch = n["pitch"]
        resource = resources[p["resource_binding"]["instrument_id"]]
        expected_midi = pitch["midi"] if pitch["kind"] == "pitched" else int(resource["unpitched_hits"][pitch["hit_key"]])
        articulation_id = p["articulation_map"][n["articulation"]].get("engine_articulation_id", "normal")
        require(event["midi"] == expected_midi and event["articulation"] == ("normal" if "performance" not in event else articulation_id), "events", "冻结音高或演奏法与谱面不符")
        _, resource_bank, resource_program, _ = resource["engine_resource_id"].split(":")
        event_bank, event_programs = articulation_preset(resource, articulation_id, int(resource_bank), int(resource_program))
        event_program = event.get("program", int(resource_program))
        require(event_program in event_programs, "events", "冻结奏法预设不在登记范围")
        require(resource["pitch_range"]["min_midi"] <= expected_midi <= resource["pitch_range"]["max_midi"] and
                ("performance" in event or "playable_midi_keys" not in resource or expected_midi in resource["playable_midi_keys"]),
                "events/pitch", "冻结音高超出真实键域或落在采样空洞")
        if soundfont_inventories is not None:
            _, bank, program, sha = resource["engine_resource_id"].split(":")
            require(sha in actual_presets and (event_bank, event_program) in actual_presets[sha], "events/assets", "冻结声部没有提供真实音色区域映射")
            matching = [z for z in actual_presets[sha][(event_bank, event_program)] if z["available_in_file"]
                        and z["key_range"][0] <= expected_midi <= z["key_range"][1]
                        and z["velocity_range"][0] <= event["velocity"] <= z["velocity_range"][1]]
            require(bool(matching) and event.get("soundfont_zone_ids") == [z["instrument_zone_id"] for z in matching]
                    and event.get("soundfont_sample_ids") == sorted({z["sample_id"] for z in matching}),
                    "events/samples", "冻结采样/区域身份或力度层与真实库不符")
        require(event.get("notated_onset_tick") == n["onset_tick"], "events", "冻结谱面时间记录与原谱不符")
        h = p["humanize"]
        max_delta = h["velocity"]["max_delta"] if h["enabled"] else 0
        require(abs(event["velocity"] - n["velocity"]) <= max_delta, "events", "冻结力度超出谱面人性化范围")
        shift = event["timing_offset_frames"]
        maximum = round_fraction(Fraction(h["timing"]["max_offset_us"] * frozen["sample_rate_hz"], 1000000)) if h["enabled"] else 0
        require(type(shift) is int and abs(shift) <= maximum, "events", "冻结微偏移超出谱面范围")
        if n["timing_locked"] or n["onset_tick"] in {s["start_tick"] for s in score["sections"]}:
            require(shift == 0, "events", "冻结事件改变了锁定起音")
        logical_on = tick_frame(n["onset_tick"], score["timeline"], frozen["sample_rate_hz"])
        logical_off = tick_frame(n["onset_tick"] + n["duration_ticks"], score["timeline"], frozen["sample_rate_hz"])
        expected_on = round_fraction(Fraction(logical_on + shift, 64)) * 64
        expected_off = round_fraction(Fraction(logical_off + shift, 64)) * 64
        require(event["onset_frame"] == expected_on and event["gate_frames"] == expected_off - expected_on and event["quantization_frames"] == expected_on - logical_on - shift, "events", "冻结起止或量化记录与谱面时值不符")
        require(event["onset_frame"] + event["gate_frames"] <= summary["music_frames"], "events", "冻结门控超出主体范围")
        if "performance" in event:
            performance = resolved_performance(n, event["onset_frame"], event["onset_frame"] + event["gate_frames"], score["timeline"], frozen["sample_rate_hz"])
            if "sample_start_frames" in event["performance"]:
                performance["sample_start_frames"] = event["performance"]["sample_start_frames"]
            require(performance == event["performance"], "events/performance", "冻结演奏控制与谱面不符")
    return frozen


def revise_gain(score, part_id, delta_db, preserve):
    validate_score(score)
    revised = copy.deepcopy(score)
    parts = {p["part_id"]: p for p in revised["parts"]}
    require(part_id in parts, "part_id", "待修改声部不存在")
    require(set(preserve) <= set(parts), "preserve", "要求保留的声部不存在")
    require(part_id not in preserve, "preserve", "修改目标与完整声部保留要求冲突")
    before = parts[part_id]["gain_db"]
    parts[part_id]["gain_db"] += delta_db
    validate_score(revised)
    original = {p["part_id"]: p for p in score["parts"]}
    checks = {pid: digest(original[pid]) == digest(parts[pid]) for pid in preserve}
    return revised, {"parent_score_sha256": digest(score), "score_sha256": digest(revised), "changes": [{"field": "parts/" + part_id + "/gain_db", "before": before, "after": parts[part_id]["gain_db"]}], "preserved_parts": checks, "seed_preserved": score["randomness"] == revised["randomness"], "score_id_preserved_for_random_streams": True}
