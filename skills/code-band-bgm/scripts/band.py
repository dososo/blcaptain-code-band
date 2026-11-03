#!/usr/bin/env python3
"""代码乐团的宿主无关 JSON 命令行入口。"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import tempfile
import time
import uuid

try:
    import numpy as np
    import soundfile as sf
    from audio_tools import AudioError, analyze, check_audio
    from fluid_engine import EngineError, engine_info, probe, render_events
    from score_tools import (ROOT, VERSION, ScoreError, canonical, compile_score, digest,
                             file_hash, load, require, revise_gain, validate_capabilities, validate_score, validate_frozen)
    from soundfont_inventory import inventory
except ModuleNotFoundError as exc:
    print(json.dumps({"status": "失败", "code": "dependency_missing", "message": "当前Python缺少运行依赖；请使用项目虚拟环境或按requirements.txt安装", "module": exc.name}, ensure_ascii=False))
    sys.exit(2)

DEFAULT_SF = ROOT / "assets" / "soundfonts" / "MuseScore_General.sf3"
RUNNER_VERSION = "0.3.20-reference-gate"


def write_json(path, data):
    """同一文件系统内发布完整新文件，已有文件绝不覆盖。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=".json-", dir=path.parent)
    temp = Path(temp)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write("\n")
        os.link(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def capabilities(soundfont, pack_registry=None):
    sfhash = file_hash(soundfont)
    registered = load(ROOT / "assets" / "soundfont.json")
    require(sfhash == registered["sha256"], "soundfont", "音色库未登记，不能沿用已登记库的来源、许可或映射；扩展时先核实并更新资产登记")
    measured = probe(soundfont)
    info = engine_info()
    mapping = inventory(soundfont)
    preset_mapping = {(p["bank"], p["program"]): p for p in mapping["presets"]}
    config = load(ROOT / "assets" / "instruments.json")
    present = {(p["bank"], p["program"]): p for p in measured["presets"]}
    provenance = {"status": "documented", "source": "MuseScore_General 0.2 官方托管音色库", "license_id": "MIT", "license_reference": "assets/soundfonts/MuseScore_General_License.md", "distribution_notes": "保留上游版权及许可；逐预设来源见同目录CSV，不宣称独占或零权利风险。"}
    resources, tests = [], []
    for item in config:
        bank, program = item["bank"], item["program"]
        require((bank, program) in present, "resources/" + item["instrument_id"], "音色库缺少预设")
        midi = 36 if bank == 128 else 60
        doc = {"sample_rate_hz": 48000, "total_frames": 48000, "parts": [{"part_id": "probe", "bank": bank, "program": program, "gain_db": 0, "pan": 0}], "events": [{"event_id": "probe", "part_id": "probe", "onset_frame": 0, "gate_frames": 12032, "midi": midi, "velocity": 90}], "mix": {"master_gain_db": 0, "effects": []}, "render": {"loop": {"enabled": False}}}
        audio, diagnostics = render_events(doc, soundfont)
        require(bool(np.isfinite(audio).all()) and float(np.max(np.abs(audio))) > 0.0001, "probe", "预设实测静音或非有限数")
        tests.append({"instrument_id": item["instrument_id"], "tested_midi": midi, "peak": float(np.max(np.abs(audio))), "frames": len(audio)})
        hits = {"kick": "36", "snare": "38", "rim": "37", "closed_hat": "42", "open_hat": "46", "ride": "51", "crash": "49", "low_tom": "45", "high_tom": "50", "shaker": "70"} if bank == 128 else {}
        key_range = preset_mapping[(bank, program)]["key_range"]
        resources.append({"instrument_id": item["instrument_id"], "label": item["label"], "kind": "sampler", "availability": "available", "engine_resource_id": f"sf:{bank}:{program}:{sfhash}", "pitch_range": {"min_midi": key_range[0], "max_midi": key_range[1]}, "articulation_ids": ["normal"], "unpitched_hits": hits, "sample_assets": [], "provenance": provenance, "limitations": ["SF3整体摘要绑定预设；已解析实际键域/力度层；采样解码帧未知，外部轮换控制未开放。", "逐音符需匹配实际音高和最终力度区域，重叠区域作为分层共同交给FluidSynth演奏。"]})
    def feature(state, limitation=None):
        return {"state": state, "evidence_ids": ["runtime_probe"] if state != "unknown" else [], "limitations": [limitation] if limitation else []}
    groups = {
        "execution": {"headless": "supported", "offline_render": "supported", "realtime_capture": "unsupported"},
        "timing": {"tempo_step_map": "supported", "meter_changes": "supported", "sample_frame_scheduling": "supported", "note_gate_release": "supported", "pitch_cents": "unsupported"},
        "performance": {"per_note_velocity": "supported", "per_note_pan": "unsupported", "articulations": "supported", "explicit_sample_selection": "unsupported", "seeded_sample_selection": "unsupported", "engine_humanize_bypass": "supported"},
        "mixing": {"part_gain": "supported", "part_pan": "supported", "reverb": "supported", "stem_export": "unsupported"},
        "export": {"wav": "supported", "exact_frame_count": "supported", "loop_tail_wrap": "supported"},
        "reproducibility": {"deterministic_event_resolution": "supported", "bit_exact_audio": "unknown"}}
    cap = {"schema_version": "1.0.0", "capability_id": "fluidsynth_local_v1", "probe_state": "partially_probed", "observed_at": datetime.now(timezone.utc).isoformat(), "engine": {"engine_id": info["engine_id"], "display_name": "本地FluidSynth采样乐团（新实现）", "version": info["version"], "adapter_id": info["adapter_id"], "adapter_version": info["adapter_version"], "platform_notes": "当前只实测macOS；不使用音频设备、不加载模型或云服务。"}, "format_limits": {"sample_rates_hz": [48000], "sample_formats": ["pcm_s24"], "channel_counts": [2], "max_render_frames": 48000 * 300, "notes": "固定默认规格，最长5分钟；64帧事件网格，各声部独立离线渲染；实际已测声部数见合奏验收，不以单MIDI端口16通道推断乐团上限。"}, "resources": resources, "resource_manifest_sha256": digest(resources), "evidence": [{"evidence_id": "runtime_probe", "kind": "command_output", "reference": "doctor JSON输出的probe_results及tests/真实渲染测试", "description": "已加载音色库，枚举预设并实测每个列出音色的代表音；软件能力需结合测试日志，不以此代表音乐验收。"}], "unresolved": ["SF3内部压缩样本解码帧和自主轮换不可查询；以整个库与预设版本锁定。", "所有风格、逐音符声像、微分音、分轨和复杂奏法未验证。", "真实宿主发现/触发及人工听感另行验收。"], "adapter_config_reference": "assets/soundfonts/MuseScore_General.sf3"}
    for group, entries in groups.items():
        cap[group] = {key: feature(state, "仅64帧网格，编译舍入误差不超过32帧" if key == "sample_frame_scheduling" else None) for key, state in entries.items()}
    from soundfont_packs import load_packs
    packs = load_packs(pack_registry, perform_probe=True)
    require(not ({r["instrument_id"] for r in resources} & {r["instrument_id"] for r in packs["resources"]}), "soundfont_packs/resources", "扩展音色与主库ID冲突")
    cap["resources"].extend(packs["resources"])
    cap["resource_manifest_sha256"] = digest(cap["resources"])
    validate_capabilities(cap)
    result = {"capabilities": cap, "probe_results": tests + packs["probe_results"], "additional_assets": packs["assets"], "asset_sha256": sfhash, "engine_info": info, "preset_count": len(present), "sample_mapping_sha256": digest(mapping)}
    from percussion_tools import add_percussion_checks
    return add_percussion_checks(result, soundfont)


def extend_capabilities(soundfont, base_path, pack_registry):
    """复用同版本主库的实际探测，只对新增库逐项发声；不冒称重跑主库。"""
    from soundfont_packs import load_packs
    require(bool(pack_registry), "soundfont_packs", "复用主库探测时须指定新增库登记")
    cap = load(base_path)
    validate_capabilities(cap)
    sha, info = file_hash(soundfont), engine_info()
    require(all(cap["engine"][key] == info[key] for key in ("engine_id", "version", "adapter_id", "adapter_version")), "engine", "基础探测的引擎或适配器版本已变化")
    registered = load(ROOT / "assets/soundfont.json")
    require(sha == registered["sha256"] and all(r["engine_resource_id"].split(":")[-1] == sha for r in cap["resources"]), "soundfont", "复用报告必须只含当前同摘要主库")
    probe_path = Path(str(base_path) + ".probe.json")
    require(probe_path.is_file(), "capabilities/evidence", "基础能力缺少原始发声探测记录")
    previous = load(probe_path)
    require(isinstance(previous, dict) and previous.get("capabilities") == cap and previous.get("asset_sha256") == sha, "capabilities/evidence", "原始探测记录与基础能力不一致")
    probes = previous.get("probe_results")
    require(isinstance(probes, list) and all(isinstance(p, dict) for p in probes), "capabilities/evidence", "逐资源发声记录缺失或格式错误")
    resources = {r["instrument_id"]: r for r in cap["resources"] if r["availability"] == "available"}
    identifiers = [p.get("instrument_id") for p in probes]
    require(all(isinstance(i, str) for i in identifiers) and len(set(identifiers)) == len(identifiers) and set(identifiers) == set(resources), "capabilities/evidence", "原发声记录未完整且唯一地覆盖全部可用主库资源")
    for recorded in probes:
        midi, peak = recorded.get("tested_midi", recorded.get("midi")), recorded.get("peak", recorded.get("peak_linear"))
        pitch = resources[recorded["instrument_id"]]["pitch_range"]
        require(type(midi) is int and pitch["min_midi"] <= midi <= pitch["max_midi"] and recorded.get("midi", midi) == midi, "capabilities/evidence", "原探测音高无效或不在资源键域")
        require(type(peak) in (int, float) and np.isfinite(peak) and peak > .0001 and type(recorded.get("frames")) is int and recorded["frames"] == 48000, "capabilities/evidence", "原探测帧数、有限峰值或有效发声证据无效")
    from percussion_tools import validate_cached_percussion
    validate_cached_percussion(cap, previous)
    packs = load_packs(pack_registry, perform_probe=True)
    require(not ({r["instrument_id"] for r in cap["resources"]} & {r["instrument_id"] for r in packs["resources"]}), "soundfont_packs/resources", "扩展音色与主库ID冲突")
    cap["resources"].extend(packs["resources"])
    cap["resource_manifest_sha256"] = digest(cap["resources"])
    cap["observed_at"] = datetime.now(timezone.utc).isoformat()
    cap["format_limits"]["notes"] = "固定默认规格，最长5分钟；64帧事件网格，各声部独立离线渲染；实际已测声部数见合奏验收，不以单MIDI端口16通道推断乐团上限。"
    validate_capabilities(cap)
    result = {"capabilities": cap, "probe_results": previous["probe_results"] + packs["probe_results"],
            "reused_primary_probe": {"file_sha256": file_hash(probe_path), "notes": "主库与引擎版本一致，复用既有逐预设真实发声证据；新增库本次实测。"},
            "additional_assets": packs["assets"], "asset_sha256": sha, "engine_info": info, "preset_count": previous["preset_count"]}
    if "percussion_probe_results" in previous:
        result["percussion_probe_results"] = copy.deepcopy(previous["percussion_probe_results"])
    return result


def published_reference_history(events_path, score):
    """预编译事件不等于历史作品；历史例外只认已交付且摘要完整匹配的文件。"""
    if not events_path:
        return None
    parent = Path(events_path).resolve().parent
    if not all((parent / name).is_file() for name in ("manifest.json", "score.json", "events.json", "final.wav")):
        return None
    manifest = load(parent / "manifest.json")
    require(isinstance(manifest, dict), "reference/history", "历史交付清单不是JSON对象")
    require(manifest.get("technical_check") == "通过", "reference/history", "历史作品没有技术交付证据")
    require(digest(score) == manifest.get("score_sha256") == digest(load(parent / "score.json")), "reference/history", "历史乐谱摘要不匹配")
    require(digest(load(events_path)) == manifest.get("events_sha256") == digest(load(parent / "events.json")), "reference/history", "历史事件摘要不匹配")
    require(file_hash(parent / "final.wav") == manifest.get("wav_file_sha256"), "reference/history", "历史WAV文件摘要不匹配")
    if manifest.get("score_file_sha256"):
        require(file_hash(parent / "score.json") == manifest["score_file_sha256"], "reference/history", "历史乐谱文件摘要不匹配")
    return {"manifest": manifest, "manifest_sha256": file_hash(parent / "manifest.json"), "directory": parent}


def reference_context(args, score, additional_part_ids=()):
    """只验证参考证据链与编排映射；不替代听辨、审美判断或用户风格验收。"""
    policy = score["intent"]["reference_policy"]
    context = {"analysis": None, "evidence_path": None, "target_evidence_path": None, "history": None,
               "understanding": {"status": "不适用"}, "style_review": {"status": "待人工验收"}}
    if policy["mode"] == "none":
        require(not args.reference_analysis, "reference", "无参考乐谱不能附加参考分析后宣称已参考")
        return context
    history = published_reference_history(args.events, score)
    analysis_path = Path(args.reference_analysis).resolve() if args.reference_analysis else None
    if analysis_path is None and history and (history["directory"] / "reference_analysis.json").is_file():
        analysis_path = history["directory"] / "reference_analysis.json"
    analysis = load(analysis_path) if analysis_path else None
    if analysis is not None:
        require(isinstance(analysis, dict) and isinstance(analysis.get("source"), dict), "reference", "参考分析必须是含来源的JSON对象")
        require(analysis.get("status") == "通过" and analysis.get("measurement_status", "通过") == "通过", "reference/measurements", "参考音频测量未完成；不能作为首次参考创作依据")
        require(policy["source_ids"] == [analysis.get("source", {}).get("source_id")], "reference/source_ids", "参考分析来源ID与乐谱引用不一致；当前每首支持一个已登记参考")
        context["analysis"] = analysis
        binding = analysis["source"].get("target_binding", {})
        require(isinstance(binding, dict), "reference/target_binding", "参考目标绑定须为JSON对象")
        if history is None:
            relation = binding.get("relationship")
            require(relation in ("user_supplied_audio", "target_media_audio", "associated_audio_verified"), "reference/target_binding", "参考音频与用户目标尚未核实；网页关联音乐不能自动视为视频实际配乐")
            require(isinstance(binding.get("target_id"), str) and binding["target_id"].strip() and binding.get("input_file_sha256") == analysis["source"].get("file_sha256"), "reference/target_binding", "目标身份或实际输入摘要缺失、不一致")
            if relation != "user_supplied_audio":
                name = binding.get("evidence_file")
                require(isinstance(name, str) and name.strip(), "reference/target_evidence", "网页目标须保留实际音轨取得或同录音匹配证据")
                target_evidence = Path(name)
                target_evidence = target_evidence if target_evidence.is_absolute() else analysis_path.parent / target_evidence
                require(target_evidence.is_file() and target_evidence.stat().st_size > 0 and file_hash(target_evidence) == binding.get("evidence_sha256"), "reference/target_evidence", "目标音轨证据缺失或摘要不符")
                match = load(target_evidence)
                require(isinstance(match, dict) and match.get("source_file_sha256") == binding["input_file_sha256"] and match.get("target_id") == binding["target_id"], "reference/target_evidence", "取得或匹配证据内部的实际音轨摘要/目标身份与本次参考不符")
                if relation == "target_media_audio":
                    require(match.get("status") == "acquired" and isinstance(match.get("method"), str) and match["method"].strip() and isinstance(match.get("record"), dict) and bool(match["record"]), "reference/target_evidence", "实际媒体声轨须保留非空原始取得记录，不能仅贴已取得标签")
                if relation == "associated_audio_verified":
                    require(match.get("status") == "matched" and match.get("method") in ("audio_fingerprint_alignment", "waveform_alignment"), "reference/target_match", "关联音频须有绑定实际目标的同录音时间匹配；语义或风格相似度不能证明身份")
                    target_hash, alignment = match.get("target_file_sha256"), match.get("alignment")
                    require(isinstance(target_hash, str) and len(target_hash) == 64 and all(c in "0123456789abcdef" for c in target_hash) and isinstance(alignment, list) and bool(alignment), "reference/target_match", "缺少目标实际声轨摘要或非空时间对齐结果")
                    for span in alignment:
                        require(isinstance(span, dict) and all(type(span.get(k)) in (int, float) and np.isfinite(span[k]) and span[k] >= 0 for k in ("source_start_seconds", "target_start_seconds", "duration_seconds")) and span["duration_seconds"] > 0, "reference/target_match", "对齐须有有限非负的源/目标起点与正匹配时长")
                context["target_evidence_path"] = target_evidence
        elif binding.get('evidence_file'):
            # 历史重演不重新取得源音频，但必须把已声明的目标证据继续交付。
            name = binding['evidence_file']
            require(isinstance(name, str) and name.strip(), 'reference/target_evidence', '历史目标证据路径无效')
            target_evidence = Path(name)
            target_evidence = target_evidence if target_evidence.is_absolute() else analysis_path.parent / target_evidence
            require(target_evidence.is_file() and target_evidence.stat().st_size > 0
                    and file_hash(target_evidence) == binding.get('evidence_sha256'),
                    'reference/target_evidence', '历史目标证据缺失或摘要改变，不能发布断链引用')
            context['target_evidence_path'] = target_evidence.resolve()
    understanding = analysis.get("reference_understanding", {}) if analysis else {}
    require(isinstance(understanding, dict), "reference/understanding", "参考理解必须是结构化证据对象")
    if understanding.get("status") == "证据已记录":
        require(understanding.get("method") in ("host_audio_perception", "local_audio_model", "external_audio_model", "user_description", "local_role_analysis"), "reference/understanding/method", "缺少支持的音频感知、本地角色分析或用户音乐描述证据类型")
        source_hash = analysis.get("source", {}).get("file_sha256")
        require(isinstance(source_hash, str) and len(source_hash) == 64 and all(c in "0123456789abcdef" for c in source_hash), "reference/source_file_sha256", "参考录音摘要格式无效")
        require(understanding.get("source_file_sha256") == source_hash, "reference/source_file_sha256", "语义证据绑定的录音与实际测量源不同")
        evidence_name = understanding.get("evidence_file")
        require(isinstance(evidence_name, str) and bool(evidence_name.strip()), "reference/evidence_file", "缺少可读取的语义证据文件")
        evidence = Path(evidence_name)
        evidence = evidence if evidence.is_absolute() else analysis_path.parent / evidence
        require(evidence.is_file(), "reference/evidence_file", "语义证据文件不存在")
        require(evidence.stat().st_size > 0, "reference/evidence_file", "语义证据文件为空")
        require(file_hash(evidence) == understanding.get("evidence_sha256"), "reference/evidence_sha256", "语义证据文件内容已变化或摘要不匹配")
        legacy_local_evidence = False
        if understanding['method'] == 'local_role_analysis':
            from role_evidence import validate_role_evidence
            try:
                validate_role_evidence(evidence, source_hash)
            except ValueError as exc:
                require(False, 'reference/role_evidence', str(exc))
        if understanding["method"] == "local_audio_model":
            local_report = load(evidence)
            require(isinstance(local_report, dict), "reference/local_model_report", "本地模型证据须为结构化原始JSON报告")
            if local_report.get("analysis_scope") == "local_audio_language_model_estimates":
                require(isinstance(local_report.get("source"), dict) and local_report["source"].get("file_sha256") == source_hash, "reference/local_model_source", "本地模型原始报告中的录音与参考测量不一致")
                segments = local_report.get("segments")
                require(local_report.get("status") == "通过" and isinstance(segments, list) and segments and all(isinstance(s, dict) and isinstance(s.get("description"), str) and s["description"].strip() for s in segments), "reference/local_model_report", "本地模型原始描述未完成")
            else:
                require(history is not None, "reference/local_model_report", "首次参考创作须提供本地模型原始报告，不能仅用人工包装的证据文件")
                legacy_local_evidence = True
        dimensions = understanding.get("dimensions", {})
        require(isinstance(dimensions, dict), "reference/dimensions", "风格观察与实现必须按维度记录")
        part_ids = {p["part_id"] for p in score["parts"]} | set(additional_part_ids)
        for dimension in ("timbre_and_space", "rhythm", "texture", "structure_and_mood"):
            item = dimensions.get(dimension, {})
            require(isinstance(item, dict), "reference/dimensions/" + dimension, "维度记录必须是JSON对象")
            require(all(isinstance(item.get(k), str) and item[k].strip() for k in ("observation", "implementation")), "reference/dimensions/" + dimension, "必须分别记录参考观察及新曲实现，不能用空字段代替证据")
            mapped = item.get("part_ids")
            require(isinstance(mapped, list) and all(isinstance(p, str) for p in mapped) and set(mapped) <= part_ids and (bool(mapped) or dimension == "structure_and_mood"), "reference/dimensions/" + dimension, "风格映射引用不存在的声部，或缺少必要的声部映射")
        context["understanding"] = copy.deepcopy(understanding)
        if legacy_local_evidence:
            context["understanding"]["status"] = "历史未核准"
            context["understanding"]["notes"] = "已有冻结作品的旧包装证据；仅重演，不作为首次参考理解证据"
        context["understanding"]["validation_scope"] = "仅核对来源绑定、证据文件摘要及声部映射；不证明描述真实、音乐同风格或用户已接受"
        context["evidence_path"] = evidence
    else:
        require(history is not None, "reference/understanding", "首次参考创作缺少语义或用户音乐描述证据；已测量不等于已理解风格。补充可靠证据后再渲染，不需要重做已有音频测量")
        context["understanding"] = {"status": "历史未核准", "notes": "仅重演已交付的冻结乐谱与事件；未补造参考语义或将其提升为风格通过"}
    if history:
        context["history"] = {"manifest_sha256": history["manifest_sha256"], "wav_file_sha256": history["manifest"]["wav_file_sha256"]}
        previous = history["manifest"].get("style_review", history["manifest"].get("human_review"))
        verdict = previous.get("status") if isinstance(previous, dict) else previous
        if verdict in ("需修改", "用户否决", "否决", "失败"):
            context["style_review"] = {"status": verdict, "previous_review": previous, "notes": "保留历史风格否决；重演或新增语义说明不等于用户重新接受"}
    return context


def publish_role_evidence(reference, stage, holder):
    """将角色小报告随作品打包，保留输入摘要与发布后可解析的相对引用。"""
    from role_evidence import copy_role_evidence
    evidence = reference['evidence_path']
    expected = holder['evidence_sha256']
    require(file_hash(evidence) == expected, 'reference/evidence_sha256', '渲染期间角色证据包发生变化')
    try:
        published = copy_role_evidence(evidence, reference['analysis']['source']['file_sha256'], stage)
    except ValueError as exc:
        require(False, 'reference/role_evidence', str(exc))
    require(file_hash(evidence) == expected, 'reference/evidence_sha256', '打包期间角色证据包发生变化')
    holder.update(evidence_file=published.name, evidence_sha256=file_hash(published),
                  input_evidence_sha256=expected)
    return [published.name, *[str(p.relative_to(stage)) for p in sorted((stage / 'reference_role_sources').iterdir())]]


def render_work(args):
    out = Path(args.out).resolve()
    require(not out.exists(), "out", "输出目录已存在；请用新的版本目录，禁止覆盖作品")
    score, cap = load(args.score), load(args.capabilities)
    reference = reference_context(args, score)
    asset_hash = file_hash(args.soundfont)
    from soundfont_packs import load_packs
    packs = load_packs(getattr(args, "soundfont_packs", None))
    frozen_controls = None
    frozen_has_controls = False
    if args.events:
        frozen = load(args.events)
        frozen_has_controls = "control_events" in frozen
        frozen_controls = frozen.pop("control_events", None)
        compiled = validate_frozen(frozen, score, cap, asset_hash,
            available_asset_sha256={asset_hash, *packs["font_paths"]},
            soundfont_inventories={asset_hash: inventory(args.soundfont), **packs["inventories"]})
    else:
        compiled = compile_score(score, cap, asset_hash, inventory(args.soundfont), additional_inventories=packs["inventories"]) if packs["inventories"] else compile_score(score, cap, asset_hash, inventory(args.soundfont))
    from instrument_readiness import review_score
    readiness = review_score(score, cap, {asset_hash: inventory(args.soundfont), **packs["inventories"]})
    accepted = getattr(args, "accept_limited_instrument", False)
    if readiness["blocked"] and not accepted:
        raise ScoreError("instrument_not_ready", "前景声部落在探针级音色上；换音色，或确认接受后加 --accept-limited-instrument",
                         {"parts": [{"part_id": r["part_id"], "instrument_id": r["instrument_id"], "gaps": r["gaps"] + r.get("used_key_gaps", [])} for r in readiness["blocked"]]})
    controls_path = getattr(args, "control_events", None)
    require(not (controls_path and frozen_has_controls), "control_events", "冻结事件已有控制时不能另外指定控制文件")
    controls = load(controls_path) if controls_path else frozen_controls
    if controls_path or frozen_has_controls:
        require(isinstance(controls, list), "control_events", "控制文件必须是事件数组")
        compiled["control_events"] = controls
    require(engine_info()["version"] == compiled["engine"]["version"], "engine/version", "本地引擎版本已变化，请重新探测")
    out.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".render-", dir=out.parent))
    started = time.monotonic()
    try:
        audio, diagnostic = render_events(compiled, args.soundfont, font_paths=packs["font_paths"]) if packs["font_paths"] else render_events(compiled, args.soundfont)
        fade = score["render"]["end_fade_frames"]
        if fade:
            audio[-fade:] *= np.linspace(1.0, 0.0, fade)[:, None]
        require(bool(np.isfinite(audio).all()), "audio", "量化前浮点包含非有限数")
        require(float(np.max(np.abs(audio))) < 1.0, "mix", "量化前峰值超出满幅，请降低混音增益")
        sf.write(str(stage / "final.wav"), audio, compiled["sample_rate_hz"], subtype="PCM_24", format="WAV")
        expected = copy.deepcopy(score["render"])
        expected["target_file_frames"] = compiled["total_frames"]
        if getattr(args, "continuous_tail_max_seconds", None) is not None:
            expected["continuous_tail_max_seconds"] = args.continuous_tail_max_seconds
        report = check_audio(stage / "final.wav", expected)
        if report["status"] != "通过":
            write_json(stage / "audio_check_failed.json", report)
        require(report["status"] == "通过", "audio_check", "最终音频检查失败，未发布作品")
        report["measurements"]["quantization_input_finite"] = "通过：本次渲染写盘前检查全部浮点样本"
        report["engine_diagnostics"] = diagnostic
        if score["render"]["loop"]["enabled"]:
            sf.write(str(stage / "loop_three_cycles.wav"), np.tile(audio, (3, 1)), compiled["sample_rate_hz"], subtype="PCM_24")
            report["measurements"]["loop_boundary"]["three_cycle_audio"] = "loop_three_cycles.wav"
            report["measurements"]["loop_boundary"]["state_initialization"] = "连续演奏3周期后捕获第3周期，保留效果状态；未经人工接缝签收"
            report["measurements"]["loop_boundary"]["notes"] = "已导出三周期拼接试听文件；音乐结构和接缝听感仍待人工验收"
        reference_files = []
        if reference["target_evidence_path"] is not None:
            binding = reference["analysis"]["source"]["target_binding"]
            name = "reference_target_evidence" + reference["target_evidence_path"].suffix
            shutil.copyfile(reference["target_evidence_path"], stage / name)
            require(file_hash(stage / name) == binding["evidence_sha256"], "reference/target_evidence", "渲染期间目标音轨证据发生变化")
            binding["evidence_file"] = name
            reference_files.append(name)
        if reference["evidence_path"] is not None:
            if reference['understanding'].get('method') == 'local_role_analysis':
                reference_files.extend(publish_role_evidence(reference, stage, reference['understanding']))
                reference['analysis']['reference_understanding'].update(reference['understanding'])
            else:
                evidence_name = "reference_evidence" + reference["evidence_path"].suffix
                shutil.copyfile(reference["evidence_path"], stage / evidence_name)
                require(file_hash(stage / evidence_name) == reference["understanding"]["evidence_sha256"], "reference/evidence_sha256", "渲染期间语义证据文件已改变，不能发布过期绑定")
                reference["understanding"]["evidence_file"] = evidence_name
                reference["analysis"]["reference_understanding"]["evidence_file"] = evidence_name
                reference_files.append(evidence_name)
        if reference["analysis"] is not None:
            write_json(stage / "reference_analysis.json", reference["analysis"])
            reference_files.append("reference_analysis.json")
        write_json(stage / "score.json", score)
        write_json(stage / "events.json", compiled)
        write_json(stage / "audio_check.json", report)
        write_json(stage / "brief.json", {"intent": score["intent"], "sections": score["sections"], "randomness": score["randomness"], "defaults": "无人声，最终时长含尾音，48kHz/24位/双声道", "reference_understanding": reference["understanding"], "style_review": reference["style_review"]})
        parent = None
        if args.change:
            change = load(args.change)
            require(change["score_sha256"] == digest(score), "change", "变更记录与乐谱不匹配")
            write_json(stage / "change.json", change)
            parent = change["parent_score_sha256"]
        manifest = {"work_id": score["score_id"], "version": out.name, "created_at": datetime.now(timezone.utc).isoformat(), "status": "技术通过", "input_summary": score["intent"]["brief"], "parent_score_sha256": parent, "seed": compiled["seed"], "engine": compiled["engine"], "asset_sha256": asset_hash, "score_sha256": digest(score), "score_file_sha256": file_hash(stage / "score.json"), "events_sha256": digest(compiled), "capability_sha256": digest(cap), "output": expected, "wav_file_sha256": file_hash(stage / "final.wav"), "decoded_float64_sha256": report["measurements"]["decoded_float64_sha256"], "command_argv": [sys.executable, *sys.argv], "render_elapsed_seconds": round(time.monotonic() - started, 3), "platform": platform.platform(), "runner_version": VERSION, "technical_check": "通过", "human_review": "未试听，等待人工审核", "originality_review": "宿主新写主题；未作全曲库唯一性或权利担保", "asset_license": "MIT，见技能内上游原文及来源CSV", "files": ["final.wav", "score.json", "events.json", "brief.json", "audio_check.json", "manifest.json", "使用说明.md"]}
        manifest.update(runner_version=RUNNER_VERSION, status_scope="audio_render_and_technical_checks", reference_understanding=reference["understanding"],
                        style_review=reference["style_review"], reference_history=reference["history"])
        manifest["activity_review"] = report.get("activity_review", "未评估")
        manifest["program_source_sha256"] = {name: file_hash(ROOT / "scripts" / name) for name in
            ("band.py", "score_tools.py", "fluid_engine.py", "soundfont_inventory.py", "soundfont_packs.py", "audio_tools.py", "delay_effect.py", "ducking_effect.py", "sound_shaping.py", "chorus_effect.py", "role_evidence.py", "eq_effect.py", "note_expression.py", "instrument_readiness.py")}
        manifest["instrument_readiness"] = {"parts": readiness["parts"], "limited_instrument_accepted": bool(readiness["blocked"] and accepted)}
        if packs["assets"]:
            used = {p["resource_sha256"] for p in compiled["parts"]}
            manifest["additional_assets"] = [a for a in packs["assets"] if a["sha256"] in used]
            if manifest["additional_assets"]:
                manifest["asset_license"] = "主库MIT；额外音色逐项许可与质量限制见additional_assets"
                for asset in manifest["additional_assets"]:
                    name = "asset_license_" + asset["sha256"] + ".txt"
                    shutil.copyfile(packs["license_paths"][asset["sha256"]], stage / name)
                    require(file_hash(stage / name) == asset["license_file_sha256"], "asset/license", "音色许可与署名在渲染期间发生变化，不能发布缺失署名的作品")
                    asset["license_file"] = name
                    manifest["files"].append(name)
        manifest["files"].extend(reference_files)
        if reference["analysis"] is not None:
            manifest["reference_source"] = copy.deepcopy(reference["analysis"]["source"])
            manifest["reference_analysis_file_sha256"] = file_hash(stage / "reference_analysis.json")
        write_json(stage / "manifest.json", manifest)
        (stage / "使用说明.md").write_text("# 作品使用说明\n\n本目录包含真实采样引擎生成的无人声音乐。技术检查通过，人工听感未试听。\n\n修改以score.json为准，保留原目录；混音修改保持score_id和音符标识，随机流不重抽。重渲染使用events.json锁定事件，同时提供同版乐谱、能力与音色库。真实命令见manifest.json。\n\n音色许可原文随技能提供，参考作品只提取抽象风格，不使用原录音切片。没有算法保证完全原创。\n", encoding="utf-8")
        with (stage / "使用说明.md").open("a", encoding="utf-8") as guide:
            guide.write("\n参考理解：" + reference["understanding"]["status"] + "；风格验收：" + reference["style_review"]["status"] + "。证据链检查不能证明风格匹配或代替人工接受。\n")
        os.rename(stage, out)
        return {"status": "通过", "status_scope": "audio_render_and_technical_checks", "work": str(out), "wav": str(out / "final.wav"), "frames": len(audio), "technical_check": "通过", "activity_review": report.get("activity_review", "未评估"), "reference_understanding": reference["understanding"], "style_review": reference["style_review"], "human_review": "未试听，等待人工审核"}
    except BaseException as exc:
        write_json(stage / "失败记录.json", {"status": "失败", "reason": str(exc), "note": "临时目录未交付为成品；可保留诊断，不可当成作品"})
        raise


def main():
    p = argparse.ArgumentParser(description="代码乐团BGM：确定性乐谱、真实本地采样、参考分析与WAV交付")
    sub = p.add_subparsers(dest="command", required=True)
    knowledge = sub.add_parser("knowledge", help="查询带来源、反例和验证方法的本地音乐决策卡")
    knowledge.add_argument("--topic", action="append", help="主题ID，可重复；不填时返回索引")
    knowledge.add_argument("--rule", action="append", help="规则ID，可重复；与主题取并集")
    knowledge.add_argument("--out", help="新JSON输出路径，不覆盖")
    phrase = sub.add_parser("phrase", help="按明确乐句计划定向调整声部起点和力度")
    phrase.add_argument("--score", required=True, help="原乐谱JSON")
    phrase.add_argument("--plan", required=True, help="显式乐句计划JSON")
    phrase.add_argument("--strength", type=float, default=1, help="计划强度，0至1")
    phrase.add_argument("--out", required=True, help="新乐谱JSON路径，不覆盖")
    arrange = sub.add_parser("arrange-check", help="按编排计划核对声部关系、唯一高峰、动机与锚点谱面数值；只读谱面")
    arrange.add_argument("--score", required=True, help="乐谱JSON")
    arrange.add_argument("--plan", required=True, help="编排计划JSON（schema_version 1.0.0）")
    arrange.add_argument("--out", help="新报告JSON路径，不覆盖")
    rotation = sub.add_parser("rotate-samples", help="为已登记双录音预设生成确定性采样轮换")
    rotation.add_argument("--score", required=True, help="原乐谱JSON")
    rotation.add_argument("--events", required=True, help="匹配原谱的原生冻结事件；不支持合奏插件文档")
    rotation.add_argument("--part", required=True, help="目标声部ID")
    rotation.add_argument("--resources", nargs=2, required=True, metavar=("RR1", "RR2"), help="登记的两种轮换预设")
    rotation.add_argument("--capabilities", required=True, help="匹配当前资产的能力JSON")
    rotation.add_argument("--soundfont", default=str(DEFAULT_SF), help="主SoundFont路径")
    rotation.add_argument("--soundfont-packs", required=True, help="扩展SoundFont登记JSON")
    rotation.add_argument("--out", required=True, help="不存在的新输出目录")
    harmonics = sub.add_parser("inspect-harmonics", help="检查显式多音候选的局部谐波与共频歧义，不认证干声或原器")
    harmonics.add_argument("--request", required=True, help="含音频SHA、候选基频及观察窗的本地JSON")
    harmonics.add_argument("--out", required=True, help="新JSON输出路径，不覆盖")
    for command in ("doctor", "catalog", "validate", "compile", "render", "analyze", "analyze-notes", "review-reference", "separate-reference", "analyze-instruments", "describe", "compare-style", "audit-reference", "check", "revise", "express"):
        c = sub.add_parser(command, help={"doctor": "环境和真实音色探测", "catalog": "查询乐器目录及真实可用资源", "validate": "乐谱结构及语义校验", "compile": "编译确定性事件", "render": "渲染并交付新作品目录", "analyze": "分析本地或公开直接媒体", "analyze-notes": "隔离运行本地多乐器音符候选分析，不核准原配器", "review-reference": "固定窗口本地语义候选、原文和真实接线核验，不自动签收参考", "separate-reference": "本地CPU六源估计、重构与残差，轨名不核准原乐器", "analyze-instruments": "分窗本地多标签乐器家族候选，不自动核准原配器", "describe": "本地音频语义描述（可选环境）", "compare-style": "校准后的本地声音对照；不自动验收风格", "audit-reference": "固定窗口声学差异，供风格缺陷定位", "check": "检查最终音频", "revise": "修改声部增益并核验保留内容", "express": "按署名拍位统计定向修改闭镲表现，不补删击打"}[command])
        if command in ("doctor", "compile", "render"):
            c.add_argument("--soundfont", default=str(DEFAULT_SF), help="显式音色库路径")
            c.add_argument("--soundfont-packs", help="可选额外SoundFont登记JSON；相对路径按登记文件定位，逐库核验摘要")
        if command == "doctor":
            c.add_argument("--reuse-capabilities", help="同引擎同摘要的主库能力及.probe.json；复用基础探测，仅实测新增库")
            c.add_argument("--verify-percussion", action="store_true", help="复用旧主库记录时补测全部标准打击键；首次完整探测会自动检查")
        if command in ("validate", "compile", "render", "revise", "express"):
            c.add_argument("--score", required=True, help="乐谱JSON")
        if command == "render":
            c.add_argument("--accept-limited-instrument", action="store_true", help="明确接受前景声部使用探针级音色；写入清单")
        if command in ("compile", "render", "catalog"):
            c.add_argument("--capabilities", required=command != "catalog", help="已探测能力JSON")
        if command == "catalog":
            c.add_argument("--query", default="", help="中文乐器名称或别名；留空返回完整目录")
        if command in ("doctor", "compile", "analyze", "describe", "audit-reference", "check", "revise", "express"):
            c.add_argument("--out", help="新JSON输出路径，不覆盖")
        if command == "render":
            c.add_argument("--out", required=True, help="新的作品版本目录")
            c.add_argument("--events", help="冻结事件，验证匹配后直接重演奏")
            c.add_argument("--control-events", help="可选CC11控制事件JSON数组；不能与已有控制的冻结事件同时提供")
            c.add_argument("--continuous-tail-max-seconds", type=float, help="显式要求持续短样时允许的最长低活动尾音；不适合有意长休止")
            c.add_argument("--change", help="定向修改证据JSON")
            c.add_argument("--reference-analysis", help="真实参考分析JSON")
        if command in ("check", "analyze", "describe", "analyze-notes", "review-reference", "separate-reference", "analyze-instruments"):
            c.add_argument("--input", required=True, help="本地音频；分析额外支持公开直接音频链接")
        if command == "analyze":
            c.add_argument("--network-mode", choices=["direct", "system_proxy"], default="direct", help="默认严格直连；显式system_proxy使用操作者信任的现有系统HTTPS代理")
            c.add_argument("--rhythm-backend", choices=["basic", "librosa", "beat_this"], default="basic", help="可选节拍后端惰性加载，不自动安装或下载")
            c.add_argument("--rhythm-model", help="beat_this必需：已取得的本地checkpoint文件")
            c.add_argument("--rhythm-model-sha256", help="beat_this必需：期望模型SHA256")
        if command == "describe":
            c.add_argument("--model", required=True, help="已在本地取得的模型目录，不自动下载")
        if command in ("analyze-notes", "review-reference", "separate-reference", "analyze-instruments"):
            c.add_argument("--model-config", required=True, help="显式本地权重/源码摘要/隔离依赖配置，无自动下载")
            c.add_argument("--backend-python", required=True, help="具备已固定可选依赖的隔离Python")
            c.add_argument("--out", required=True, help="新的候选分析目录，不覆盖")
            if command == "analyze-instruments":
                c.add_argument("--window-plan", required=True, help="绑定本地源SHA的结构化分窗JSON，不按标签自动选乐器")
            else:
                c.add_argument("--start", type=float, default=0, help="本地音频窗口起点秒")
                c.add_argument("--seconds", type=float, default=8, help="窗口长度0.5至30秒")
            if command == "analyze-notes":
                c.add_argument("--prefix-seconds", type=float, choices=[0, .25], default=.25, help="已验证的左边缘上下文预算，不改变波形增益")
            elif command == "review-reference":
                c.add_argument("--task", choices=["roles", "control"], default="roles", help="角色候选或匿名已知控制题")
                c.add_argument("--tokens", type=int, default=768, help="256至1024；达到预算即拒绝作为语义结果")
        if command == "compare-style":
            c.add_argument("--inputs", required=True, help="本地匿名音频输入清单JSON，含明确语音负对照")
            c.add_argument("--model", required=True, help="已验证的本地CLAP模型目录，不自动下载")
            c.add_argument("--out", required=True, help="新的对照输出目录，不覆盖")
            c.add_argument("--batch-size", type=int, default=4, help="推理批量，默认4")
        if command == "audit-reference":
            c.add_argument("--reference", required=True, help="真实本地参考音频")
            c.add_argument("--candidate", required=True, help="真实本地候选音频")
            c.add_argument("--reference-start", type=float, default=0, help="参考窗口起点秒")
            c.add_argument("--candidate-start", type=float, default=0, help="候选窗口起点秒")
            c.add_argument("--seconds", type=float, default=20, help="同规格对照时长，1至30秒")
        if command == "check":
            c.add_argument("--expected", help="含render字段的乐谱或输出规格JSON")
        if command == "express":
            c.add_argument("--part", required=True, help="明确要修改的闭镲声部ID")
            c.add_argument("--profile", default=str(ROOT / "assets/groove-profiles/gmd_hiphop110_closed_hat.json"), help="有来源许可的条件统计JSON")
        if command == "revise":
            c.add_argument("--part", required=True, help="要改增益的声部ID")
            c.add_argument("--gain-db", required=True, type=float, help="增益变化，单位dB")
            c.add_argument("--preserve", nargs="+", default=[], help="完整保留的声部ID")
    args = p.parse_args()
    try:
        if args.command == "doctor":
            result = extend_capabilities(args.soundfont, args.reuse_capabilities, args.soundfont_packs) if args.reuse_capabilities else capabilities(args.soundfont, args.soundfont_packs)
            if args.reuse_capabilities and args.verify_percussion:
                from percussion_tools import add_percussion_checks
                result = add_percussion_checks(result, args.soundfont)
            from delay_effect import add_delay_capability
            result = add_delay_capability(result)
            from eq_effect import add_eq_capability
            result = add_eq_capability(result)
            from sound_shaping import add_shaping_capability
            result = add_shaping_capability(result, args.soundfont)
            from note_expression import add_expression_capability
            result = add_expression_capability(result, args.soundfont)
            from ducking_effect import add_duck_capability
            result = add_duck_capability(result)
            from chorus_effect import add_chorus_capability
            result = add_chorus_capability(result, args.soundfont)
            validate_capabilities(result["capabilities"])
            if args.out:
                write_json(args.out, result["capabilities"])
                write_json(str(args.out) + ".probe.json", result)
            percussion = result.get("percussion_probe_results", [])
            failed_hits = [{"instrument_id": group["instrument_id"], "midi": p["midi"], "reason": p.get("reason", p.get("code"))}
                           for group in percussion for p in group["hits"] if p["status"] != "通过"]
            result = {"status": "通过", "status_scope": "capability_probe", "engine": result["engine_info"], "capabilities": args.out,
                      "preset_count": result["preset_count"], "available_resource_count": len(result["capabilities"]["resources"]),
                      "asset_sha256": result["asset_sha256"], "percussion_hits_tested": sum(len(p["hits"]) for p in percussion), "unavailable_percussion_hits": failed_hits}
        elif args.command == "knowledge":
            from music_knowledge import query_knowledge
            result = query_knowledge(args.topic, args.rule)
            if args.out:
                write_json(args.out, result)
        elif args.command == "inspect-harmonics":
            from harmonic_observation import inspect_harmonics
            result = inspect_harmonics(args.request)
            write_json(args.out, result)
        elif args.command == "catalog":
            from instrument_catalog import query_catalog
            result = query_catalog(args.query, args.capabilities)
        elif args.command == "arrange-check":
            from arrangement_check import check_arrangement, PlanError
            score = load(args.score)
            validate_score(score)
            try:
                result = check_arrangement(score, load(args.plan))
            except PlanError as error:
                raise ScoreError("semantic_invalid", "编排计划无效", {"field": "plan", "reason": str(error)}) from error
            result.update(score_sha256=digest(score), plan_file_sha256=file_hash(args.plan),
                          program_sha256=file_hash(Path(__file__).with_name("arrangement_check.py")))
            if args.out:
                require(not Path(args.out).exists(), "out", "报告已存在，拒绝覆盖")
                write_json(args.out, result)
        elif args.command == "rotate-samples":
            from sample_rotation import rotate_score, validate_rotation_parent, validate_rotation_output
            from soundfont_packs import load_packs
            output = Path(args.out)
            require(not output.exists(), "out", "轮换输出目录已存在，拒绝覆盖")
            soundfont = inventory(args.soundfont)
            packs = load_packs(args.soundfont_packs)
            score, cap, frozen = load(args.score), load(args.capabilities), load(args.events)
            asset = soundfont["source"]["sha256"]
            validate_rotation_parent(frozen, score, cap, asset, soundfont, packs["inventories"])
            revised, events, report = rotate_score(score, cap,
                soundfont["source"]["sha256"], soundfont, packs["inventories"], args.part, args.resources)
            validate_rotation_output(frozen, events, args.part, report["generated_part_ids"])
            output.mkdir(parents=True)
            for name, value in (("score.json", revised), ("events.json", events), ("rotation.json", report)):
                write_json(output / name, value)
            result = {"status": "通过", "output": str(output), "slots": len(report["assignments"]),
                      "scope": "已登记独立采样轮换及事件编译；音色听感未验"}
        elif args.command == "phrase":
            from phrase_expression import apply_phrase
            paths = [Path(args.out), Path(str(args.out) + ".change.json")]
            require(not any(path.exists() for path in paths), "out", "乐句输出已存在，拒绝覆盖")
            original = load(args.score)
            revised, evidence = apply_phrase(original, load(args.plan), args.strength)
            evidence["parent_score_sha256"] = digest(original)
            evidence["score_sha256"] = digest(revised)
            evidence["expression_program_sha256"] = file_hash(Path(__file__).with_name("phrase_expression.py"))
            for path, value in zip(paths, (revised, evidence)):
                write_json(path, value)
            result = {"status": "通过", "score": str(paths[0]), "change": str(paths[1]),
                      "changed_note_count": len(evidence["changes"]), "scope": evidence["scope"]}
        elif args.command == "validate":
            result = {"status": "通过", **validate_score(load(args.score)), "note": "结构与部分谱面语义通过；仍须编译检查音源、人性化和最终帧，且尚未生成音频"}
        elif args.command == "express":
            from groove_profile import apply_profile
            require(bool(args.out), "out", "表现修改需要新的乐谱输出路径")
            paths = [Path(args.out), Path(str(args.out)+".change.json"), Path(str(args.out)+".groove-profile.json")]
            require(not any(p.exists() for p in paths), "out", "表现修改输出已存在，拒绝覆盖")
            profile = load(args.profile)
            revised, evidence = apply_profile(load(args.score), profile, args.part)
            evidence["profile_file_sha256"] = file_hash(args.profile)
            for path, value in zip(paths, (revised, evidence, profile)):
                write_json(path, value)
            result = {"status": "通过", "score": args.out, "change": str(paths[1]), "profile": str(paths[2]),
                      "eligible_note_count": evidence["eligible_note_count"], "unmatched_note_count": evidence["unmatched_note_count"],
                      "scope": evidence["scope"]}
        elif args.command == "compile":
            from soundfont_packs import load_packs
            packs = load_packs(args.soundfont_packs)
            result = compile_score(load(args.score), load(args.capabilities), file_hash(args.soundfont), inventory(args.soundfont), additional_inventories=packs["inventories"]) if packs["inventories"] else compile_score(load(args.score), load(args.capabilities), file_hash(args.soundfont), inventory(args.soundfont))
            if args.out:
                write_json(args.out, result)
            result = {"status": "通过", "event_count": len(result["events"]), "events_sha256": digest(result), "out": args.out}
        elif args.command == "render":
            result = render_work(args)
        elif args.command == "audit-reference":
            from reference_audit import audit_reference
            result = audit_reference(args.reference, args.candidate, args.reference_start, args.candidate_start, args.seconds)
            if args.out:
                write_json(args.out, result)
        elif args.command == "analyze":
            options = {"network_mode": args.network_mode, "rhythm_backend": args.rhythm_backend}
            if args.rhythm_model is not None or args.rhythm_model_sha256 is not None:
                options.update(rhythm_model=args.rhythm_model, rhythm_model_sha256=args.rhythm_model_sha256)
            result = analyze(args.input, **options)
            result["source"].setdefault("source_id", "reference_" + result["source"]["file_sha256"][:16])
            if args.out:
                write_json(args.out, result)
        elif args.command == "describe":
            from local_audio_review import describe_audio
            from reference_gate import SEMANTIC_LABEL, require_mono_models
            measured, gate = require_mono_models(args.input)
            result = describe_audio(args.input, args.model)
            result.update(signal_gate=gate, gate_measurements=measured, brief_policy=SEMANTIC_LABEL)
            if args.out:
                write_json(args.out, result)
        elif args.command == "analyze-notes":
            from note_analysis import analyze_notes
            from reference_gate import require_mono_models
            require_mono_models(args.input)
            result = analyze_notes(args.input, args.model_config, args.backend_python, args.out, args.start, args.seconds, args.prefix_seconds)
        elif args.command == "review-reference":
            from semantic_review import review_reference
            from reference_gate import SEMANTIC_LABEL, require_mono_models
            measured, gate = require_mono_models(args.input)
            result = review_reference(args.input, args.model_config, args.backend_python, args.out, args.start, args.seconds, args.task, args.tokens)
            result.update(signal_gate=gate, gate_measurements=measured, brief_policy=SEMANTIC_LABEL)
        elif args.command == "analyze-instruments":
            from instrument_analysis import analyze_instruments
            from reference_gate import require_mono_models
            require_mono_models(args.input)
            result = analyze_instruments(args.input, args.model_config, args.backend_python, args.window_plan, args.out)
        elif args.command == "separate-reference":
            from separate_analysis import separate_reference
            from reference_gate import require_mono_models
            require_mono_models(args.input)
            result = separate_reference(args.input, args.model_config, args.backend_python, args.out, args.start, args.seconds)
        elif args.command == "compare-style":
            from style_compare import compare_style
            result = compare_style(args.inputs, args.model, args.out, batch_size=args.batch_size)
        elif args.command == "check":
            expected = load(args.expected) if args.expected else {}
            result = check_audio(args.input, expected.get("render", expected))
            if args.out:
                write_json(args.out, result)
        else:
            require(bool(args.out), "out", "修改需要新的乐谱输出路径")
            revised, evidence = revise_gain(load(args.score), args.part, args.gain_db, args.preserve)
            require(not Path(args.out).exists() and not Path(str(args.out) + ".change.json").exists(), "out", "修改输出已存在")
            write_json(args.out, revised)
            write_json(str(args.out) + ".change.json", evidence)
            result = {"status": "通过", "score": args.out, "change": str(args.out) + ".change.json", **evidence}
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
        return 0 if result.get("status") != "失败" else 2
    except (ScoreError, AudioError, EngineError) as exc:
        print(json.dumps({"status": "失败", "code": exc.code, "message": exc.message, "details": exc.details}, ensure_ascii=False))
    except ImportError as exc:
        print(json.dumps({"status": "失败", "code": "dependency_missing", "message": "当前Python缺少可选本地音频分析依赖；常规测量和渲染不需要此依赖", "module": exc.name}, ensure_ascii=False))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "失败", "code": "input_or_io_error", "message": "输入或文件操作失败", "details": {"reason": str(exc)}}, ensure_ascii=False))
    return 2


if __name__ == "__main__":
    sys.exit(main())
