"""编排检查：按显式编排计划核对声部关系、作曲结构与认可锚点的谱面数值。

只读谱面，不碰音频。计划是独立 JSON（schema_version 1.0.0），不改乐谱 schema：
{
  "schema_version": "1.0.0",
  "profile": "ensemble" | "sparse_solo",          # sparse_solo 只报告锚点数值，不提示
  "roles": {"<part_id>": "foreground|answer|bass|harmony|rhythm|color"},
  "sections": {"<section_id>": {"foreground": "<part_id>", "roles": {...覆盖}}},   # 可选，写交接
  "peak_section": "<section_id>",                  # 可选，唯一高峰所在段
  "motifs": [{"motif_id": "a", "part_id": "<part_id>", "intervals": [2, 2, -4], "min_occurrences": 2}]   # 可选
}
判定分两级：fail（结构冲突，计划自己声明了却没做到）与 warn（经验阈值，需人核）。
锚点阈值来自 3 首认可、2 首否决作品的谱面对照（work/Claude交付_代码乐团_20261103/evidence/anchor_analysis/音乐对照结论.md），
样本少且风格混杂，所以只作 warn，不作 fail。
"""
from fractions import Fraction

ROLES = {"foreground", "answer", "bass", "harmony", "rhythm", "color"}
REGISTERS = [(-1, 48), (48, 60), (60, 72), (72, 84), (84, 200)]  # 与锚点分析同一划分
# 阈值取认可锚点（折线 / 灯火 / 向光）最小值与否决作品（江暮 / 山间）最大值之间；数值是本模块按谱面口径重算的
# （work/主流乐团完善_20261103/编排检查/evidence/calibration.json）。分不开的只报告，不提示。
ANCHOR = {"onset_density_per_s": (None, "≥", "按谱面写定起音：认可 2.90–7.97，否决 1.02–2.92，分不开；锚点分析的 3.80–8.10 含人性化偏移"),
          "registers_ge3_share": (.5, "≥", "认可 0.60–0.94，否决 0.10–0.14"),
          "polyphony_mean": (3.5, "≥", "认可 3.99–5.50，否决 2.12–2.19"),
          "foreground_long_note_share": (.5, "≤", "认可 0.01–0.33，否决 1.00"),
          "velocity_std": (None, "≥", "认可 12.8–15.6，否决 8.9–12.1，差距小")}
ANSWER_COLLISION_MAX = .25
MASKING_MAX = .35  # 认可作品最高 0.32（灯火未眠）
HANDOFF_GAP_BEATS = 1


class PlanError(ValueError):
    pass


def _seconds_of(timeline):
    ppq, tempos = timeline["ppq"], sorted(timeline["tempo_map"], key=lambda t: t["start_tick"])

    def seconds(tick):
        total, previous = Fraction(0), None
        for index, tempo in enumerate(tempos):
            start = tempo["start_tick"]
            if start >= tick:
                break
            end = min(tick, tempos[index + 1]["start_tick"] if index + 1 < len(tempos) else tick)
            bpm = Fraction(tempo["bpm"]["numerator"], tempo["bpm"]["denominator"])
            total += Fraction(end - start, ppq) * 60 / bpm
        return float(total)
    return seconds


def _notes(score):
    rows = []
    for part in score["parts"]:
        for note in part["notes"]:
            pitch = note["pitch"]
            rows.append({"part": part["part_id"], "on": note["onset_tick"], "off": note["onset_tick"] + note["duration_ticks"],
                         "midi": pitch.get("midi") if pitch.get("kind") == "pitched" else None, "velocity": note["velocity"]})
    return rows


def _section_of(score, tick):
    for section in score["sections"]:
        if section["start_tick"] <= tick < section["end_tick"]:
            return section["section_id"]
    return score["sections"][-1]["section_id"]


def _roles(plan, section_id):
    roles = dict(plan.get("roles", {}))
    roles.update(plan.get("sections", {}).get(section_id, {}).get("roles", {}))
    foreground = plan.get("sections", {}).get(section_id, {}).get("foreground")
    if foreground:
        roles = {part: ("harmony" if role == "foreground" and part != foreground else role) for part, role in roles.items()}
        roles[foreground] = "foreground"
    return roles


def validate_plan(score, plan):
    if plan.get("schema_version") != "1.0.0":
        raise PlanError("编排计划 schema_version 必须是 1.0.0")
    parts = {p["part_id"] for p in score["parts"]}
    sections = {s["section_id"] for s in score["sections"]}
    if plan.get("profile", "ensemble") not in ("ensemble", "sparse_solo"):
        raise PlanError("profile 只能是 ensemble 或 sparse_solo")
    for scope, roles in [("roles", plan.get("roles", {}))] + [(f"sections.{k}.roles", v.get("roles", {})) for k, v in plan.get("sections", {}).items()]:
        for part, role in roles.items():
            if part not in parts:
                raise PlanError(f"{scope} 引用了不存在的声部：{part}")
            if role not in ROLES:
                raise PlanError(f"{scope}.{part} 的职责不认识：{role}")
    for section_id, spec in plan.get("sections", {}).items():
        if section_id not in sections:
            raise PlanError(f"sections 引用了不存在的段落：{section_id}")
        if spec.get("foreground") and spec["foreground"] not in parts:
            raise PlanError(f"sections.{section_id}.foreground 引用了不存在的声部")
    if plan.get("peak_section") and plan["peak_section"] not in sections:
        raise PlanError("peak_section 引用了不存在的段落")
    for motif in plan.get("motifs", []):
        if motif.get("part_id") not in parts or not motif.get("intervals"):
            raise PlanError(f"动机 {motif.get('motif_id')} 须给存在的 part_id 与非空 intervals")
    for section_id in sections:
        if "foreground" not in _roles(plan, section_id).values():
            raise PlanError(f"段落 {section_id} 没有前景声部")


def _result(check_id, level, ok, value, threshold, detail):
    return {"check_id": check_id, "level": level, "status": "通过" if ok else ("不通过" if level == "fail" else "需核"),
            "value": value, "threshold": threshold, "detail": detail}


def _overlap(a_on, a_off, b_on, b_off):
    return max(0, min(a_off, b_off) - max(a_on, b_on))


def relation_checks(score, plan, notes):
    ppq = score["timeline"]["ppq"]
    results = []
    by_section = {}
    for note in notes:
        by_section.setdefault(_section_of(score, note["on"]), []).append(note)
    # R1 回答句不该和主句同时起音：呼应要错开
    hits = total = 0
    for section_id, items in by_section.items():
        roles = _roles(plan, section_id)
        fore_onsets = sorted({n["on"] for n in items if roles.get(n["part"]) == "foreground"})
        for note in items:
            if roles.get(note["part"]) == "answer":
                total += 1
                hits += any(abs(note["on"] - on) <= ppq // 8 for on in fore_onsets)
    if total:
        share = round(hits / total, 3)
        results.append(_result("answer_onset_collision", "fail", share <= ANSWER_COLLISION_MAX, share, f"≤{ANSWER_COLLISION_MAX}",
                               f"回答声部 {total} 个起音里 {hits} 个与前景起音相距不到 1/8 拍；呼应应落在前景的长音或休止里"))
    # R2 音区遮蔽：非前景有音高声部在前景音 ±2 半音内同时发声的时长占前景时长的比例（低音、节奏不计）
    masked = fore_time = 0
    for section_id, items in by_section.items():
        roles = _roles(plan, section_id)
        others = [n for n in items if n["midi"] is not None and roles.get(n["part"]) in ("answer", "harmony", "color")]
        for fore in (n for n in items if n["midi"] is not None and roles.get(n["part"]) == "foreground"):
            fore_time += fore["off"] - fore["on"]
            spans = sorted((max(o["on"], fore["on"]), min(o["off"], fore["off"])) for o in others
                           if abs(o["midi"] - fore["midi"]) <= 2 and _overlap(o["on"], o["off"], fore["on"], fore["off"]))
            covered, cursor = 0, fore["on"]
            for start, end in spans:
                start = max(start, cursor)
                if end > start:
                    covered += end - start
                    cursor = end
            masked += covered
    if fore_time:
        share = round(masked / fore_time, 3)
        results.append(_result("register_masking", "warn", share <= MASKING_MAX, share, f"≤{MASKING_MAX}",
                               "前景发声时，回答/和声/色彩声部在前景音 ±2 半音内同时发声的时长占比；高了前景会被盖住"))
    # R3 低音在场：只报告。按谱面发声时长算，认可 0.43–0.995、否决（山间）0.475，分不开；
    # 真正分开的是音频 <120 Hz 能量占比（认可 0.27–0.73，否决 0.02–0.11），要看渲染后的 check
    bass_parts = {p for p, r in plan.get("roles", {}).items() if r == "bass"}
    if bass_parts:
        end = score["timeline"]["music_end_tick"]
        spans = sorted((n["on"], min(n["off"], end)) for n in notes if n["part"] in bass_parts)
        covered, cursor = 0, 0
        for start, stop in spans:
            start = max(start, cursor)
            if stop > start:
                covered += stop - start
                cursor = stop
        results.append({"check_id": "bass_presence", "level": "report", "status": "仅报告", "value": round(covered / end, 3), "threshold": None,
                        "detail": "低音声部发声时长占音乐时长；谱面口径分不开认可与否决，低频是否足够以渲染后的 <120 Hz 能量为准"})
    # R4 交接：相邻段前景换人时，新前景应在旧前景收尾 1 拍内接上（可以重叠咬合）
    sections = score["sections"]
    for left, right in zip(sections, sections[1:]):
        old = next(p for p, r in _roles(plan, left["section_id"]).items() if r == "foreground")
        new = next(p for p, r in _roles(plan, right["section_id"]).items() if r == "foreground")
        if old == new:
            continue
        old_end = max((n["off"] for n in by_section.get(left["section_id"], []) if n["part"] == old), default=left["start_tick"])
        new_start = min((n["on"] for n in by_section.get(right["section_id"], []) if n["part"] == new), default=right["end_tick"])
        gap = round((new_start - old_end) / ppq, 2)
        results.append(_result(f"handoff:{left['section_id']}->{right['section_id']}", "warn", gap <= HANDOFF_GAP_BEATS, gap,
                               f"≤{HANDOFF_GAP_BEATS} 拍", f"前景从 {old} 交给 {new}：旧前景收尾到新前景进入的空档（负数为咬合重叠）"))
    return results


def _foreground_notes(score, plan, notes):
    rows = []
    for note in notes:
        if note["midi"] is not None and _roles(plan, _section_of(score, note["on"])).get(note["part"]) == "foreground":
            rows.append(note)
    return sorted(rows, key=lambda n: (n["on"], -n["midi"]))


def composition_checks(score, plan, notes):
    results = []
    fore = _foreground_notes(score, plan, notes)
    if plan.get("peak_section") and fore:
        top = max(n["midi"] for n in fore)
        peak_top = max((n["midi"] for n in fore if _section_of(score, n["on"]) == plan["peak_section"]), default=None)
        places = sorted({_section_of(score, n["on"]) for n in fore if n["midi"] == top})
        # 硬门：声明的高峰段必须到达全曲前景最高音，否则到达点落空
        results.append(_result("peak_reaches_top", "fail", peak_top == top, {"top_midi": top, "peak_section_top_midi": peak_top},
                               plan["peak_section"], "声明的高峰段要到达全曲前景最高音"))
        # 提示：最高音在别段也出现，高峰就不唯一。长曲副歌回归时可以重现（向光行进 v38b 七段都到 82），所以只提示
        results.append(_result("unique_peak", "warn", places == [plan["peak_section"]], {"top_midi": top, "sections": places},
                               plan["peak_section"], "前景最高音最好只在高峰段出现；短样里提前用掉最高音，到达感会变弱"))
    for motif in plan.get("motifs", []):
        line = sorted((n for n in notes if n["part"] == motif["part_id"] and n["midi"] is not None), key=lambda n: (n["on"], -n["midi"]))
        heads = []
        for index in range(len(line)):
            if index and line[index]["on"] == line[index - 1]["on"]:
                continue
            heads.append(line[index])
        steps = [b["midi"] - a["midi"] for a, b in zip(heads, heads[1:])]
        pattern, size = motif["intervals"], len(motif["intervals"])
        exact = [i for i in range(len(steps) - size + 1) if steps[i:i + size] == pattern]
        # 变尾：前 n−1 个音程相同、最后一个不同（动机回认又不原样重复）。动机至少 3 个音程才判变尾，
        # 否则「同一个音程后接任意音程」到处都是，变尾失去意义
        varied = [i for i in range(len(steps) - size + 1) if size >= 3 and steps[i:i + size - 1] == pattern[:-1] and steps[i + size - 1] != pattern[-1]]
        need = motif.get("min_occurrences", 2)
        count = len(exact) + len(varied)
        results.append(_result(f"motif:{motif.get('motif_id', '?')}", "warn", count >= need and bool(varied),
                               {"exact": len(exact), "varied_tail": len(varied)}, f"合计≥{need}，且至少一次变尾",
                               "动机要能被认出（重复）又不能原样照搬（变尾）"))
    return results


def anchor_metrics(score, plan, notes):
    seconds = _seconds_of(score["timeline"])
    end_tick = score["timeline"]["music_end_tick"]
    duration = seconds(end_tick)
    onsets = {n["on"] for n in notes}
    step = max(score["timeline"]["ppq"] // 16, 1)
    pitched = [n for n in notes if n["midi"] is not None]
    active = poly_sum = registers_ge3 = 0
    for tick in range(0, end_tick, step):
        sounding = [n for n in pitched if n["on"] <= tick < n["off"]]
        if sounding:
            active += 1
            poly_sum += len(sounding)
            registers_ge3 += len({i for n in sounding for i, (low, high) in enumerate(REGISTERS) if low <= n["midi"] < high}) >= 3
    beat = score["timeline"]["ppq"]
    fore = _foreground_notes(score, plan, notes)
    velocities = [n["velocity"] for n in notes]
    mean = sum(velocities) / len(velocities)
    values = {"onset_density_per_s": round(len(onsets) / duration, 2),
              "registers_ge3_share": round(registers_ge3 / active, 3) if active else 0.0,
              "polyphony_mean": round(poly_sum / active, 2) if active else 0.0,
              "foreground_long_note_share": round(sum(n["off"] - n["on"] >= beat for n in fore) / len(fore), 3) if fore else 0.0,
              "velocity_std": round((sum((v - mean) ** 2 for v in velocities) / len(velocities)) ** .5, 1)}
    results = []
    for key, value in values.items():
        threshold, sense, evidence = ANCHOR[key]
        if threshold is None or plan.get("profile", "ensemble") == "sparse_solo":
            reason = "认可与否决分不开" if threshold is None else "稀疏独奏不套合奏锚点"
            results.append({"check_id": f"anchor:{key}", "level": "report", "status": "仅报告", "value": value,
                            "threshold": None if threshold is None else f"{sense}{threshold}", "detail": f"{reason}；{evidence}"})
        else:
            ok = value >= threshold if sense == "≥" else value <= threshold
            results.append(_result(f"anchor:{key}", "warn", ok, value, f"{sense}{threshold}", evidence))
    return results


def check_arrangement(score, plan):
    validate_plan(score, plan)
    notes = _notes(score)
    results = relation_checks(score, plan, notes) + composition_checks(score, plan, notes) + anchor_metrics(score, plan, notes)
    failed = [r["check_id"] for r in results if r["status"] == "不通过"]
    review = [r["check_id"] for r in results if r["status"] == "需核"]
    return {"schema_version": "1.0.0", "status": "不通过" if failed else ("需核" if review else "通过"),
            "status_scope": "谱面编排结构与锚点数值；不代表听感", "failed": failed, "needs_review": review, "checks": results}
