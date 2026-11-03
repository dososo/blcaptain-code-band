"""乐器概念查询；不会演奏、替换音色或将分类资料视为采样授权。"""

import copy
from collections import Counter
import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "assets/instrument-catalog.json"


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _object(value, label):
    if not isinstance(value, dict):
        raise ValueError(f"{label}必须是JSON对象")
    return value


def _records(value, label):
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise ValueError(f"{label}必须是JSON对象数组")
    return value


def _hash(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def _relative_file(base, value, label):
    if not isinstance(value, str) or not value or '\\' in value or '\0' in value:
        raise ValueError(f"{label}路径无效")
    relative = Path(value)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError(f"{label}路径不可逃逸")
    resolved_base = base.resolve()
    resolved = (resolved_base / relative).resolve()
    if not resolved.is_relative_to(resolved_base):
        raise ValueError(f"{label}路径不可逃逸")
    return resolved


def _plugin_resource(resource):
    """仅登记固定VST3资源描述；不加载插件或声称运行可用。"""
    resource = _object(resource, "插件资源")
    required = {"instrument_id", "concept_id", "backend", "label", "availability",
                "state_path", "state_sha256", "performance_path", "performance_sha256",
                "model_path", "model_sha256", "plugin", "tested_midi_keys", "render_route",
                "source_url", "software_license", "source_notes", "limitations"}
    if set(resource) != required or resource.get("backend") != "vst3_single_part" or resource.get("availability") != "partial":
        raise ValueError("插件资源字段、后端或有限状态无效")
    for key in ("instrument_id", "concept_id", "label", "source_url", "software_license", "source_notes"):
        if not isinstance(resource[key], str) or not resource[key].strip() or '\0' in resource[key]:
            raise ValueError(f"插件资源{key}无效")
    if resource["render_route"] != "render-plugin":
        raise ValueError("插件资源须显式使用render-plugin路线")
    paths = {}
    for key in ("state", "performance", "model"):
        if not isinstance(resource[key + "_sha256"], str) or not re.fullmatch(r"[a-f0-9]{64}", resource[key + "_sha256"]):
            raise ValueError(f"插件资源{key}摘要无效")
        paths[key] = _relative_file(ROOT / "assets", resource[key + "_path"], f"插件{key}")
    plugin = _object(resource["plugin"], "插件身份")
    if set(plugin) != {"name", "version", "identifier", "binary_relative_path", "binary_sha256"}:
        raise ValueError("插件身份字段无效")
    if any(not isinstance(plugin[k], str) or not plugin[k].strip() or '\0' in plugin[k]
           for k in ("name", "version", "identifier", "binary_relative_path")):
        raise ValueError("插件身份无效")
    _relative_file(Path("/"), plugin["binary_relative_path"], "插件二进制相对路径")
    if not isinstance(plugin["binary_sha256"], str) or not re.fullmatch(r"[a-f0-9]{64}", plugin["binary_sha256"]):
        raise ValueError("插件二进制摘要无效")
    keys = resource["tested_midi_keys"]
    if (not isinstance(keys, list) or not keys or any(type(k) is not int or not 0 <= k <= 127 for k in keys)
            or keys != sorted(set(keys))):
        raise ValueError("插件已测键集无效")
    if (not isinstance(resource["limitations"], list) or not resource["limitations"]
            or any(not isinstance(x, str) or not x.strip() for x in resource["limitations"])):
        raise ValueError("插件资源限制缺失")
    return dict(resource, resource_kind="plugin", asset_path=paths["state"],
                asset_sha256=resource["state_sha256"], performance_asset_path=paths["performance"],
                model_asset_path=paths["model"])


def _registry(catalog):
    registry = {}
    metadata = _object(catalog["registry"], "目录注册信息")
    primary_asset = ROOT / "assets/soundfonts/MuseScore_General.sf3"
    for resource in _records(_read(ROOT / metadata["path"]), "基础资源注册表"):
        registry[resource["instrument_id"]] = dict(
            resource, asset_path=primary_asset,
            asset_sha256=catalog["registry"]["soundfont_sha256"])
    supplementary = metadata.get("supplementary_path")
    if supplementary:
        packs = _object(_read(ROOT / supplementary), "补充资源注册表")
        for pack in _records(packs["packs"], "补充资源包"):
            for resource in _records(pack["resources"], "补充资源"):
                resource_id = resource["instrument_id"]
                if resource_id in registry:
                    raise ValueError(f"资源注册重复：{resource_id}")
                registry[resource_id] = dict(
                    resource, asset_path=ROOT / "assets" / pack["path"],
                    asset_sha256=pack["sha256"])
    plugin_path = metadata.get("plugin_resources_path")
    if plugin_path:
        path = _relative_file(ROOT, plugin_path, "插件资源注册")
        expected_hash = metadata.get("plugin_resources_sha256")
        if (not isinstance(expected_hash, str) or not re.fullmatch(r"[a-f0-9]{64}", expected_hash)
                or not path.is_file() or _hash(path) != expected_hash):
            raise ValueError("插件资源注册缺失或摘要不匹配")
        document = _object(_read(path), "插件资源注册表")
        if set(document) != {"schema_version", "resources"} or document["schema_version"] != "1.0.0":
            raise ValueError("插件资源注册版本无效")
        for item in _records(document["resources"], "插件资源"):
            resource = _plugin_resource(item)
            identifier = resource["instrument_id"]
            if identifier in registry:
                raise ValueError(f"资源注册重复：{identifier}")
            registry[identifier] = resource
    return registry


def validate_catalog(catalog=None):
    """校验概念、来源和真实注册引用；不会把结构校验报告成音频实测。"""
    catalog = _read(CATALOG_PATH) if catalog is None else catalog
    _object(catalog, "乐器目录")
    registry = _registry(catalog)
    entries = _records(catalog["instruments"], "乐器条目")
    _records(catalog["families"], "家族条目")
    _records(catalog["sources"], "来源条目")
    _object(catalog["summary"], "目录摘要")
    ids = {entry["instrument_id"] for entry in entries}
    families = {family["family_id"] for family in catalog["families"]}
    sources = {source["source_id"] for source in catalog["sources"]}
    if len(ids) != len(entries):
        raise ValueError("乐器概念ID重复")
    if len(families) != len(catalog["families"]) or len(sources) != len(catalog["sources"]):
        raise ValueError("家族或来源ID重复")
    for entry in entries:
        name = entry["instrument_id"]
        if entry["family_id"] not in families:
            raise ValueError(f"家族引用不存在：{name}")
        if not entry["source_refs"] or not set(entry["source_refs"]) <= sources:
            raise ValueError(f"来源引用不存在：{name}")
        if not set(entry.get("not_equivalent_to", [])) <= ids:
            raise ValueError(f"非等价乐器引用不存在：{name}")
        state = entry["availability"]
        resources = entry["available_resource_ids"]
        if state not in {"available", "partial", "missing", "planned"}:
            raise ValueError(f"未知可用状态：{name}")
        if state in {"missing", "planned"}:
            if resources or entry["resource_id"] is not None:
                raise ValueError(f"缺失音色不能自动绑定：{name}")
            continue
        if not resources or entry["resource_id"] not in resources:
            raise ValueError(f"可用音色没有首选资源：{name}")
        if "required_hit_notes" in entry:
            hits = _object(entry["required_hit_notes"], f"{name}逻辑鼓键")
            if (set(hits) != set(entry.get("required_hit_keys", [])) or
                    any(not isinstance(note, str) or not note.isdecimal() or not 0 <= int(note) <= 127
                        for note in hits.values())):
                raise ValueError(f"逻辑鼓键名称与音符声明不符：{name}")
        for resource_id in resources:
            if resource_id not in registry:
                raise ValueError(f"资源未注册：{name}/{resource_id}")
            ranges = _object(entry["resource_pitch_ranges"], f"{name}资源键域")
            pitch = ranges.get(resource_id)
            _object(pitch, f"{name}/{resource_id}键域")
            if not pitch or not 0 <= pitch["min_midi"] <= pitch["max_midi"] <= 127:
                raise ValueError(f"资源键域无效：{name}/{resource_id}")
            resource = registry[resource_id]
            if resource.get("resource_kind") == "plugin":
                tested = resource["tested_midi_keys"]
                if (state != "partial" or resource["concept_id"] != name
                        or pitch != {"min_midi": tested[0], "max_midi": tested[-1]}
                        or entry.get("resource_playable_midi_keys", {}).get(resource_id) != tested):
                    raise ValueError(f"插件有限概念或已测音区声明不符：{name}/{resource_id}")
            if "resource_playable_midi_keys" in entry:
                keys = _object(entry["resource_playable_midi_keys"], f"{name}真实键集").get(resource_id)
                if (not isinstance(keys, list) or not keys or any(type(k) is not int or not pitch["min_midi"] <= k <= pitch["max_midi"] for k in keys)
                        or keys != sorted(set(keys))):
                    raise ValueError(f"真实键集无效或没有覆盖已绑定资源：{name}/{resource_id}")
        primary = registry[entry["resource_id"]]
        if primary.get("resource_kind") == "plugin":
            if entry["resource_bank"] is not None or entry["resource_program"] is not None:
                raise ValueError(f"插件资源不得声明SoundFont bank/program：{name}")
        elif (entry["resource_bank"], entry["resource_program"]) != (primary["bank"], primary["program"]):
            raise ValueError(f"资源bank/program不符：{name}")
        if state == "partial" and (
                entry.get("prototype_resource_id") not in resources or
                entry.get("full_instrument_coverage") != "missing" or
                entry.get("pitch_range") != entry["resource_pitch_ranges"][entry["prototype_resource_id"]]):
            raise ValueError(f"有限原型范围声明不完整：{name}")
    if (catalog["summary"]["catalog_entries"] != len(entries) or
            catalog["summary"]["availability_counts"] != dict(Counter(item["availability"] for item in entries))):
        raise ValueError("目录摘要计数与条目不符")
    return {"status": "通过", "status_scope": "catalog_structure_only",
            "catalog_entries": len(entries), "registered_resources": len(registry),
            "runtime_verified": False}


def _runtime(entry, capability, registry, asset_hashes):
    result = {"runtime_verified": False, "verification_scope": "未提供能力报告；查询不会演奏",
              "resources": []}
    has_plugin = any(registry[r].get("resource_kind") == "plugin" for r in entry["available_resource_ids"])
    if capability is None and not has_plugin:
        return result
    report = capability or {}
    evidence_ids = {item.get("evidence_id") for item in report.get("evidence", [])}
    offline = report.get("execution", {}).get("offline_render", {})
    probe_ok = (report.get("probe_state") in {"probed", "partially_probed"} and
                report.get("engine", {}).get("engine_id") == "fluidsynth" and
                offline.get("state") == "supported" and
                bool(offline.get("evidence_ids")) and
                set(offline["evidence_ids"]) <= evidence_ids)
    cap_resources = {item["instrument_id"]: item for item in report.get("resources", [])}
    for resource_id in entry["available_resource_ids"]:
        registered = registry[resource_id]
        if registered.get("resource_kind") == "plugin":
            checks = {}
            for label, path_key, hash_key in (("state", "asset_path", "state_sha256"),
                    ("performance", "performance_asset_path", "performance_sha256"),
                    ("model", "model_asset_path", "model_sha256")):
                path = registered[path_key]
                if path not in asset_hashes:
                    asset_hashes[path] = _hash(path) if path.is_file() else None
                checks[label] = asset_hashes[path] == registered[hash_key]
            result["resources"].append({
                "resource_id": resource_id, "backend": "vst3_single_part", "runtime_verified": False,
                "state_sha256": registered["state_sha256"], "state_file_verified": checks["state"],
                "performance_file_verified": checks["performance"], "model_file_verified": checks["model"],
                "plugin": copy.deepcopy(registered["plugin"]), "render_route": registered["render_route"],
                "state_path": "assets/" + registered["state_path"],
                "performance_path": "assets/" + registered["performance_path"],
                "path_base": "技能根目录",
                "tested_midi_keys": list(registered["tested_midi_keys"]),
                "pitch_range": entry["resource_pitch_ranges"][resource_id],
                "limitations": list(registered["limitations"]),
                "reason_zh": "仅核对固定资源文件摘要；需显式提供插件并在当前宿主实际render-plugin验证"})
            continue
        if capability is None:
            continue
        resource = cap_resources.get(resource_id, {})
        expected = f"sf:{registered['bank']}:{registered['program']}:{registered['asset_sha256']}"
        asset_path = registered["asset_path"]
        if asset_path not in asset_hashes:
            asset_hashes[asset_path] = _hash(asset_path) if asset_path.is_file() else None
        pitch = resource.get("pitch_range", {})
        catalog_pitch = entry["resource_pitch_ranges"][resource_id]
        required_hits = set(entry.get("required_hit_keys", []))
        actual_hits = resource.get("unpitched_hits", {})
        hit_notes_match = all(actual_hits.get(key) == note
                              for key, note in entry.get("required_hit_notes", {}).items())
        expected_keys = entry.get("resource_playable_midi_keys", {}).get(resource_id)
        keys_match = expected_keys is None or resource.get("playable_midi_keys") == expected_keys
        verified = (probe_ok and resource.get("availability") == "available" and
                    resource.get("engine_resource_id") == expected and
                    asset_hashes[asset_path] == registered["asset_sha256"] and
                    pitch == catalog_pitch and
                    required_hits <= set(actual_hits) and hit_notes_match and keys_match)
        result["resources"].append({
            "resource_id": resource_id, "runtime_verified": verified,
            "pitch_range": pitch if verified else catalog_pitch,
            "playable_midi_keys": resource.get("playable_midi_keys", expected_keys) if verified else expected_keys,
            "unpitched_hits": resource.get("unpitched_hits", {}) if verified else {},
            "reason_zh": "能力报告代表音证据、当前资产摘要与注册范围一致" if verified else
            "能力状态、资源、资产摘要、键域或所需鼓键未核对通过"})
    result["runtime_verified"] = any(item["runtime_verified"] for item in result["resources"])
    result["verification_scope"] = ("采样资源核对能力报告；插件只核对状态文件，需显式提供插件并实测；未重新演奏或验人工听感"
                                    if has_plugin else
                                    "读取能力报告并核对当前资产；未重新演奏，未验全部音域、奏法或人工听感")
    return result


def query_catalog(query="", capabilities_path=None):
    """名称/别名/概念ID/家族查询；缺失资源不返回替代建议。"""
    if not isinstance(query, str):
        raise ValueError("乐器查询必须是文本")
    catalog = _read(CATALOG_PATH)
    validate_catalog(catalog)
    capability = _read(capabilities_path) if capabilities_path else None
    if capabilities_path:
        from score_tools import validate_capabilities
        _object(capability, "引擎能力报告")
        validate_capabilities(capability)
    families = {item["family_id"]: item["name_zh"] for item in catalog["families"]}
    needle = query.strip().casefold()
    matches = []
    registry = _registry(catalog)
    asset_hashes = {}
    for entry in catalog["instruments"]:
        names = [entry["instrument_id"], entry["name_zh"], *entry["aliases_zh"]]
        terms = names + [entry["family_id"], families[entry["family_id"]]]
        if not needle or any(needle in term.casefold() for term in terms):
            match = copy.deepcopy(entry)
            # 目录内历史说明与本次查询核验状态分列，避免误用历史标签。
            match["runtime"] = _runtime(entry, capability, registry, asset_hashes)
            matches.append((0 if needle in [term.casefold() for term in names] else 1, match))
    matches.sort(key=lambda item: item[0])
    return {"status": "通过", "status_scope": "catalog_query", "query": query,
            "catalog_id": catalog["catalog_id"], "summary": catalog["summary"],
            "families": catalog["families"], "matches": [item[1] for item in matches],
            "capabilities_path": str(capabilities_path) if capabilities_path else None,
            "notes_zh": "分类目录是缺口清单；仅available/partial可按资源范围尝试绑定。名称匹配不证明音色、风格或听感合格。"}
