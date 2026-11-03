"""显式登记的额外SoundFont；资产及许可摘要在本机验证，路径不进入乐谱。"""
from pathlib import Path

import numpy as np

from fluid_engine import probe, render_events
from score_tools import file_hash, load, require
from soundfont_inventory import inventory


def load_packs(registry_path, perform_probe=False):
    result = {"font_paths": {}, "license_paths": {}, "inventories": {}, "resources": [], "assets": [], "probe_results": []}
    if not registry_path:
        return result
    registry = Path(registry_path).resolve()
    document = load(registry)
    require(isinstance(document, dict) and document.get("schema_version") == "1.0.0" and isinstance(document.get("packs"), list), "soundfont_packs", "额外音色登记格式无效")
    identifiers = set()
    for pack in document["packs"]:
        require(isinstance(pack, dict) and isinstance(pack.get("resources"), list), "soundfont_packs", "每个库须为含资源列表的对象")
        path = (registry.parent / pack["path"]).resolve()
        license_file = (registry.parent / pack["license_file"]).resolve()
        require(path.is_file() and license_file.is_file(), "soundfont_packs", "额外库或许可文件不存在")
        sha = file_hash(path)
        require(sha == pack["sha256"], "soundfont_packs/sha256", "额外音色库与登记摘要不匹配")
        require(sha not in result["font_paths"], "soundfont_packs", "额外库重复登记")
        mapping = inventory(path)
        presets = {(p["bank"], p["program"]): p for p in mapping["presets"]}
        if perform_probe:
            probe(path)
        for item in pack["resources"]:
            require(isinstance(item, dict), "soundfont_packs/resources", "每个资源须为JSON对象")
            identifier, bank, program = item["instrument_id"], item["bank"], item["program"]
            require(identifier not in identifiers, "soundfont_packs/resources", "额外音色ID重复")
            require((bank, program) in presets, "soundfont_packs/resources", "登记的额外预设不存在")
            identifiers.add(identifier)
            preset = presets[(bank, program)]
            playable = sorted({key for zone in preset["mappings"] if zone["available_in_file"]
                               and zone["velocity_range"][1] >= 1 and zone["velocity_range"][0] <= 127
                               for key in range(zone["key_range"][0], zone["key_range"][1] + 1)})
            if "supported_keys" in item:
                declared = item["supported_keys"]
                require(isinstance(declared, list) and all(type(k) is int for k in declared) and sorted(declared) == playable,
                        "soundfont_packs/supported_keys", "声明键集与真实可演奏区域不一致；不得用外包围音区掩盖缺键")
            hits = item.get("unpitched_hits", {})
            require(isinstance(hits, dict) and all(isinstance(key, str) for key in hits), "soundfont_packs/hits", "打击乐逻辑键须为JSON对象")
            for key in hits.values():
                require(str(key).isdigit() and any(z["available_in_file"] and z["key_range"][0] <= int(key) <= z["key_range"][1] for z in preset["mappings"]), "soundfont_packs/hits", "打击乐映射没有真实采样区域")
            resource = {"instrument_id": identifier, "label": item["label"], "kind": "sampler", "availability": "available",
                        "engine_resource_id": f"sf:{bank}:{program}:{sha}",
                        "pitch_range": {"min_midi": preset["key_range"][0], "max_midi": preset["key_range"][1]},
                        "playable_midi_keys": playable,
                        "articulation_ids": ["normal"], "unpitched_hits": hits, "sample_assets": [],
                        "provenance": {"status": "documented", "source": pack["source_url"], "license_id": pack["license_id"],
                                       "license_reference": pack["license_file"], "distribution_notes": pack["limitations"]},
                        "limitations": [pack["limitations"], "音符必须匹配真实区域；未实现的演奏法不得冒充可用。"]}
            if "articulations" in item:
                resource["articulation_ids"] = ["normal", *sorted(a for a in item["articulations"] if a != "normal")]
                resource["articulation_programs"] = item["articulations"]
                for name, spec in item["articulations"].items():
                    for candidate in spec["programs"]:
                        require((spec["bank"], candidate) in presets, "soundfont_packs/articulations", f"奏法 {name} 的预设不存在")
            if "quality" in item:
                resource["quality"] = item["quality"]
            result["resources"].append(resource)
            if perform_probe:
                zone = next(z for z in preset["mappings"] if z["available_in_file"])
                midi = next(iter(hits.values()), max(zone["key_range"][0], min(60, zone["key_range"][1])))
                velocity = max(zone["velocity_range"][0], min(90, zone["velocity_range"][1]))
                doc = {"sample_rate_hz": 48000, "total_frames": 48000, "parts": [{"part_id": "probe", "bank": bank, "program": program, "gain_db": 0, "pan": 0}],
                       "events": [{"event_id": "probe", "part_id": "probe", "onset_frame": 0, "gate_frames": 12032, "midi": int(midi), "velocity": velocity}],
                       "mix": {"master_gain_db": 0, "effects": []}, "render": {"loop": {"enabled": False}}}
                audio, _ = render_events(doc, path)
                require(np.isfinite(audio).all() and float(np.max(np.abs(audio))) > .0001, "soundfont_packs/probe", "额外音色实测静音或包含非有限数")
                result["probe_results"].append({"instrument_id": identifier, "midi": int(midi), "velocity": velocity,
                                               "peak_linear": float(np.max(np.abs(audio))), "frames": len(audio)})
        result["font_paths"][sha] = path
        result["license_paths"][sha] = license_file
        result["inventories"][sha] = mapping
        result["assets"].append({"pack_id": pack["pack_id"], "file_name": path.name, "sha256": sha,
                                  "license_id": pack["license_id"], "license_file_sha256": file_hash(license_file),
                                  "source_url": pack["source_url"], "limitations": pack["limitations"]})
        if pack.get("source_manifest"):
            source_manifest = (registry.parent / pack["source_manifest"]).resolve()
            require(source_manifest.is_file(), "soundfont_packs/source_manifest", "逐采样来源清单不存在")
            result["assets"][-1]["source_manifest_sha256"] = file_hash(source_manifest)
    return result
