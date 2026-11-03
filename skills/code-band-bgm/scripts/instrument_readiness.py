"""按真实映射判定键位覆盖：数据来自音色库区域，不看注册表的自述。

coverage_ok 只表示根音和音域够写前景声部，不表示音色好听、发音地道或已通过人耳验收。
"""
from statistics import median

TIERS = ("coverage_ok", "limited", "probe")
FOREGROUND_FUNCTIONS = {"melody", "countermelody"}


def _presets(resource, inventory):
    _, bank, program, _ = resource["engine_resource_id"].split(":")
    wanted = {(int(bank), int(program))}
    for spec in resource.get("articulation_programs", {}).values():
        wanted |= {(spec["bank"], candidate) for candidate in spec["programs"]}
    return [p for p in inventory["presets"] if (p["bank"], p["program"]) in wanted]


def measure(resource, inventory):
    zones = [z for preset in _presets(resource, inventory) for z in preset["mappings"] if z["available_in_file"]]
    roots = sorted({z["root_key"] for z in zones if z["root_key"] is not None})
    limits = resource.get("pitch_range") or {"min_midi": 0, "max_midi": 127}
    # GM 音色常把首尾区域拉到 0/127，声明音域也跟着写满；以真实根音外延7半音为可信核心
    low, high = (roots[0] - 7, roots[-1] + 7) if roots else (0, 127)
    declared = {key for z in zones for key in range(z["key_range"][0], z["key_range"][1] + 1)
                if limits["min_midi"] <= key <= limits["max_midi"]}
    keys = sorted(key for key in declared if low <= key <= high)
    transposition = [min(abs(key - z["root_key"]) for z in zones
                         if z["key_range"][0] <= key <= z["key_range"][1] and z["root_key"] is not None) for key in keys
                     if any(z["root_key"] is not None and z["key_range"][0] <= key <= z["key_range"][1] for z in zones)]
    # 力度层和轮换只看普通奏法；其他奏法的采样不是同一发音的替换
    normal_programs = set(resource.get("articulation_programs", {}).get("normal", {}).get("programs", []))
    _, bank, program, _ = resource["engine_resource_id"].split(":")
    normal_programs.add(int(program))
    normal = [z for preset in _presets(resource, inventory) if preset["program"] in normal_programs
              for z in preset["mappings"] if z["available_in_file"]]
    layers, takes = {}, {}
    for z in normal:
        for key in range(z["key_range"][0], z["key_range"][1] + 1):
            layers.setdefault(key, set()).add(tuple(z["velocity_range"]))
            takes.setdefault(key, set()).add(z["sample_id"])
    stereo_pairs = 2 if zones and all(z["source_sample_name"].endswith(("_L", "_R", " L", " R")) for z in zones) else 1
    span = keys[-1] - keys[0] + 1 if keys else 0
    ordered = sorted(transposition)
    return {"root_keys": len(roots), "playable_keys": len(keys), "key_span": span,
            "contiguous": len(keys) == span, "max_transposition_semitones": max(transposition, default=None),
            # 只看最坏一键会让音域边缘的一个区域决定整件乐器；按九成键位的移调判定
            "p90_transposition_semitones": ordered[int(.9 * (len(ordered) - 1))] if ordered else None,
            "median_velocity_layers": median(len(v) for v in layers.values()) if layers else 0,
            "median_takes_per_key": median(len(v) / stereo_pairs for v in takes.values()) if takes else 0,
            "stretched_keys": len(declared) - len(keys),
            "any_loop": any(z["sample_modes"] for z in zones),
            "unpitched_hits": len(resource.get("unpitched_hits", {})),
            "articulations": resource.get("articulation_ids", ["normal"])}


def classify(facts):
    gaps = []
    if facts["unpitched_hits"]:
        if facts["median_velocity_layers"] < 2:
            gaps.append("单力度层")
        if facts["median_takes_per_key"] < 2 * max(facts["median_velocity_layers"], 1):
            gaps.append("没有轮换，重复击打会有机关枪感")
        return ("coverage_ok" if not gaps else "limited"), gaps
    if facts["root_keys"] < 5:
        gaps.append(f"只有{facts['root_keys']}个真实根音")
    if facts["p90_transposition_semitones"] is None or facts["p90_transposition_semitones"] > 7:
        gaps.append("超过一成键位移调超过7半音")
    if facts["key_span"] and facts["playable_keys"] / facts["key_span"] < .5:
        gaps.append(f"音域内只有{facts['playable_keys']}/{facts['key_span']}键可发声")
    if gaps:
        return "probe", gaps
    if facts["root_keys"] < 10:
        gaps.append(f"只有{facts['root_keys']}个真实根音")
    if facts["p90_transposition_semitones"] > 4:
        gaps.append(f"九成键位移调上限{facts['p90_transposition_semitones']}半音")
    if facts["key_span"] < 24:
        gaps.append(f"音域仅{facts['key_span']}键")
    if not facts["contiguous"]:
        gaps.append("键域有空洞")
    return ("coverage_ok" if not gaps else "limited"), gaps


def assess(resource, inventory):
    facts = measure(resource, inventory)
    tier, gaps = classify(facts)
    notes = []
    if facts["median_velocity_layers"] < 2:
        notes.append("多数键只有一个力度层，强弱只靠音量和滤波")
    if facts["median_takes_per_key"] < 2:
        notes.append("多数键只有一个采样，同音连奏会完全相同")
    if facts["stretched_keys"] and not facts["unpitched_hits"]:
        notes.append(f"另有{facts['stretched_keys']}个声明键离真实根音超过7半音，只能当音效用")
    if not facts["any_loop"]:
        notes.append("无循环，长音上限为采样自身长度")
    return {"instrument_id": resource["instrument_id"], "tier": tier, "gaps": gaps, "notes": notes,
            "human_review": "未验收", "facts": facts}


def used_key_gaps(notes, inventory_zones):
    """这首谱实际用到的音：不能落在无区域处，离最近真实根音不能超过7半音。"""
    gaps = []
    for midi in sorted({n["pitch"]["midi"] for n in notes if n.get("pitch", {}).get("kind") == "pitched"}):
        covering = [z["root_key"] for z in inventory_zones
                    if z["key_range"][0] <= midi <= z["key_range"][1] and z["root_key"] is not None]
        if not covering:
            gaps.append(f"MIDI {midi} 没有可发声区域")
        elif min(abs(midi - root) for root in covering) > 7:
            gaps.append(f"MIDI {midi} 离真实根音{min(abs(midi - root) for root in covering)}半音")
    return gaps


def review_score(score, cap, inventories):
    """返回每个声部的判定；blocked 是落在探针级音色上、或用到失真键位的前景声部。"""
    resources = {r["instrument_id"]: r for r in cap["resources"]}
    rows, blocked = [], []
    for part in score["parts"]:
        binding = part.get("resource_binding", {})
        resource = resources.get(binding.get("instrument_id"))
        if resource is None or resource.get("kind") != "sampler":
            continue
        mapping = inventories.get(resource["engine_resource_id"].split(":")[3])
        if mapping is None:
            continue
        row = {"part_id": part["part_id"], "musical_function": part.get("musical_function"), **assess(resource, mapping)}
        if not resource.get("unpitched_hits"):
            zones = [z for preset in _presets(resource, mapping) for z in preset["mappings"] if z["available_in_file"]]
            row["used_key_gaps"] = used_key_gaps(part.get("notes", []), zones)
        rows.append(row)
        if part.get("musical_function") in FOREGROUND_FUNCTIONS and (row["tier"] == "probe" or row.get("used_key_gaps")):
            blocked.append(row)
    return {"parts": rows, "blocked": blocked}
