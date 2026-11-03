"""通过 libfluidsynth 的离线内存接口演奏已解析事件；不打开音频设备。"""

from __future__ import annotations

import ctypes as ct
import ctypes.util
import hashlib
import math
import shutil
from pathlib import Path

import numpy as np


class EngineError(Exception):
    def __init__(self, code, message, details=None):
        super().__init__(message)
        self.code, self.message, self.details = code, message, details or {}


def _library():
    path = ctypes.util.find_library("fluidsynth")
    if not path:
        executable = shutil.which("fluidsynth")
        if executable:
            prefix = Path(executable).parent
            candidates = []
            for directory in (prefix, prefix.parent / "lib"):
                for pattern in ("libfluidsynth.dylib", "libfluidsynth.so*", "libfluidsynth*.dll"):
                    candidates.extend(sorted(directory.glob(pattern)))
            if candidates:
                path = str(candidates[0])
    if not path:
        raise EngineError("engine_missing", "找不到 libfluidsynth；请先安装系统运行库。")
    try:
        lib = ct.CDLL(path)
    except OSError as exc:
        raise EngineError("engine_load", "无法加载 libfluidsynth。", {"reason": str(exc)}) from exc
    signatures = {
        "fluid_version_str": (ct.c_char_p, []),
        "new_fluid_settings": (ct.c_void_p, []),
        "delete_fluid_settings": (None, [ct.c_void_p]),
        "fluid_settings_setint": (ct.c_int, [ct.c_void_p, ct.c_char_p, ct.c_int]),
        "fluid_settings_setnum": (ct.c_int, [ct.c_void_p, ct.c_char_p, ct.c_double]),
        "fluid_settings_get_type": (ct.c_int, [ct.c_void_p, ct.c_char_p]),
        "new_fluid_synth": (ct.c_void_p, [ct.c_void_p]),
        "delete_fluid_synth": (None, [ct.c_void_p]),
        "fluid_synth_sfload": (ct.c_int, [ct.c_void_p, ct.c_char_p, ct.c_int]),
        "fluid_synth_get_sfont_by_id": (ct.c_void_p, [ct.c_void_p, ct.c_int]),
        "fluid_sfont_iteration_start": (None, [ct.c_void_p]),
        "fluid_sfont_iteration_next": (ct.c_void_p, [ct.c_void_p]),
        "fluid_preset_get_name": (ct.c_char_p, [ct.c_void_p]),
        "fluid_preset_get_banknum": (ct.c_int, [ct.c_void_p]),
        "fluid_preset_get_num": (ct.c_int, [ct.c_void_p]),
        "fluid_synth_get_internal_bufsize": (ct.c_int, [ct.c_void_p]),
        "fluid_synth_program_select": (ct.c_int, [ct.c_void_p] + [ct.c_int] * 4),
        "fluid_synth_noteon": (ct.c_int, [ct.c_void_p] + [ct.c_int] * 3),
        "fluid_synth_noteoff": (ct.c_int, [ct.c_void_p] + [ct.c_int] * 2),
        "fluid_synth_get_polyphony": (ct.c_int, [ct.c_void_p]),
        "fluid_synth_count_midi_channels": (ct.c_int, [ct.c_void_p]),
        "fluid_synth_get_voicelist": (None, [ct.c_void_p, ct.POINTER(ct.c_void_p),
                                            ct.c_int, ct.c_int]),
        "fluid_voice_get_channel": (ct.c_int, [ct.c_void_p]),
        "fluid_voice_get_key": (ct.c_int, [ct.c_void_p]),
        "fluid_voice_is_on": (ct.c_int, [ct.c_void_p]),
        "fluid_synth_cc": (ct.c_int, [ct.c_void_p] + [ct.c_int] * 3),
        "fluid_synth_pitch_bend": (ct.c_int, [ct.c_void_p] + [ct.c_int] * 2),
        "fluid_synth_pitch_wheel_sens": (ct.c_int, [ct.c_void_p] + [ct.c_int] * 2),
        "fluid_synth_get_cc": (ct.c_int, [ct.c_void_p, ct.c_int, ct.c_int,
                                        ct.POINTER(ct.c_int)]),
        "fluid_synth_set_gen": (ct.c_int, [ct.c_void_p, ct.c_int, ct.c_int, ct.c_float]),
        "fluid_synth_get_gen": (ct.c_float, [ct.c_void_p, ct.c_int, ct.c_int]),
        "fluid_synth_set_interp_method": (ct.c_int, [ct.c_void_p, ct.c_int, ct.c_int]),
        "fluid_synth_write_float": (ct.c_int, [ct.c_void_p, ct.c_int, ct.c_void_p,
                                             ct.c_int, ct.c_int, ct.c_void_p,
                                             ct.c_int, ct.c_int]),
    }
    for name, (result, arguments) in signatures.items():
        fn = getattr(lib, name)
        fn.restype, fn.argtypes = result, arguments
    return lib


def _ok(value, operation):
    if value < 0:
        raise EngineError("engine_call", "音频引擎调用失败。", {"operation": operation})


class _Synth:
    def __init__(self, soundfont_path, sample_rate, reverb, midi_channels=None):
        self.lib = _library()
        self.settings = self.lib.new_fluid_settings()
        self.synth = None
        if not self.settings:
            raise EngineError("engine_memory", "无法创建音频设置。")
        try:
            integers = {"synth.cpu-cores": 1, "synth.chorus.active": 0,
                        "synth.reverb.active": int(reverb is not None),
                        "synth.threadsafe-api": 0, "synth.parallel-render": 0,
                        "synth.polyphony": 256, "synth.min-note-length": 0}
            if midi_channels is not None:
                integers["synth.midi-channels"] = midi_channels
            # parallel-render 仅在部分版本存在；有此设置时固定关闭。
            for name, value in integers.items():
                if name == "synth.parallel-render" and self.lib.fluid_settings_get_type(
                        self.settings, name.encode()) < 0:
                    continue
                _ok(self.lib.fluid_settings_setint(self.settings, name.encode(), value), name)
            if self.lib.fluid_settings_get_type(self.settings, b"synth.limiter.active") >= 0:
                _ok(self.lib.fluid_settings_setint(self.settings, b"synth.limiter.active", 0),
                    "synth.limiter.active")
            numbers = {"synth.sample-rate": sample_rate, "synth.gain": 0.4}
            if reverb is not None:
                numbers.update({"synth.reverb.room-size": reverb["room_size"],
                                "synth.reverb.damp": reverb["damp"],
                                "synth.reverb.width": reverb["width"],
                                "synth.reverb.level": reverb["level"]})
            for name, value in numbers.items():
                _ok(self.lib.fluid_settings_setnum(self.settings, name.encode(), value), name)
            self.synth = self.lib.new_fluid_synth(self.settings)
            if not self.synth:
                raise EngineError("engine_memory", "无法创建离线合成器。")
            _ok(self.lib.fluid_synth_set_interp_method(self.synth, -1, 4), "插值设置")
            self.sfont_id = self.lib.fluid_synth_sfload(
                self.synth, str(Path(soundfont_path).resolve()).encode("utf-8"), 0)
            _ok(self.sfont_id, "加载 SoundFont")
            self.block = self.lib.fluid_synth_get_internal_bufsize(self.synth)
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.synth:
            self.lib.delete_fluid_synth(self.synth)
            self.synth = None
        if self.settings:
            self.lib.delete_fluid_settings(self.settings)
            self.settings = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def samples(self, frames):
        output = np.empty((frames, 2), dtype=np.float32)
        # 两个通道共享交错数组，通过偏移与步幅指定输出。
        _ok(self.lib.fluid_synth_write_float(self.synth, frames,
                                            output.ctypes.data, 0, 2,
                                            output.ctypes.data, 1, 2), "浮点渲染")
        if not np.isfinite(output).all():
            raise EngineError("non_finite_audio", "量化前音频含 NaN 或无穷值。")
        return output

    def noteoff(self, channel, midi):
        result = self.lib.fluid_synth_noteoff(self.synth, channel, midi)
        if result >= 0:
            return False
        if result != -1:
            _ok(result, "音符释音")
        channels = self.lib.fluid_synth_count_midi_channels(self.synth)
        polyphony = self.lib.fluid_synth_get_polyphony(self.synth)
        if not (0 <= channel < channels and 0 <= midi <= 127 and polyphony > 0):
            _ok(result, "音符释音")
        # 离线单线程中，在下次合成之前读取完整voice列表，指针不会被并发回收。
        voices = (ct.c_void_p * (polyphony + 1))()
        self.lib.fluid_synth_get_voicelist(self.synth, voices, len(voices), -1)
        for voice in voices:
            if not voice:
                return True
            if (self.lib.fluid_voice_get_channel(voice) == channel
                    and self.lib.fluid_voice_get_key(voice) == midi
                    and self.lib.fluid_voice_is_on(voice)):
                _ok(result, "音符释音")
        _ok(result, "无法完整核实释音后的声部状态")


def engine_info():
    lib = _library()
    return {"engine_id": "fluidsynth", "version": lib.fluid_version_str().decode(),
            "adapter_id": "fluid_memory", "adapter_version": "1.0.0",
            "audio_device_opened": False, "cpu_cores": 1,
            "interpolation": "fluidsynth_4th_order", "synth_gain": 0.4}


def probe(soundfont_path):
    path = Path(soundfont_path)
    if not path.is_file():
        raise EngineError("soundfont_missing", "SoundFont 文件不存在。")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    with _Synth(path, 48000, None) as synth:
        font = synth.lib.fluid_synth_get_sfont_by_id(synth.synth, synth.sfont_id)
        if not font:
            raise EngineError("soundfont_invalid", "加载后没有可枚举的 SoundFont。")
        synth.lib.fluid_sfont_iteration_start(font)
        presets = []
        while True:
            preset = synth.lib.fluid_sfont_iteration_next(font)
            if not preset:
                break
            presets.append({"bank": synth.lib.fluid_preset_get_banknum(preset),
                            "program": synth.lib.fluid_preset_get_num(preset),
                            "name": synth.lib.fluid_preset_get_name(preset).decode("utf-8", "replace")})
        return {**engine_info(), "soundfont_sha256": digest.hexdigest(),
                "internal_block_frames": synth.block,
                "presets": sorted(presets, key=lambda p: (p["bank"], p["program"])),
                "sample_selection": "SoundFont 内部映射固定；不暴露单样本轮换接口。",
                "sample_metadata": "音区、力度层与样本帧须另查 SoundFont 区域表；不由预设名称推断。"}


def probe_hits(soundfont_path, bank, program, hits: dict[str, str]) -> list[dict]:
    """同一干声合成器逐键探测；每次全停并清空一块，避免前键释音污染。"""
    from soundfont_inventory import inventory

    if (type(bank) is not int or not 0 <= bank <= 16383 or
            type(program) is not int or not 0 <= program <= 127):
        raise EngineError("invalid_preset", "逐键探测的音色库或程序编号无效。")
    if not isinstance(hits, dict) or any(
            not isinstance(key, str) or not key or not isinstance(value, str) or
            not value.isdigit() or not 0 <= int(value) <= 127 for key, value in hits.items()):
        raise EngineError("invalid_hit_map", "打击键必须是非空标识到0至127整数文本的映射。")
    if not hits:
        return []
    path = Path(soundfont_path)
    if not path.is_file():
        raise EngineError("soundfont_missing", "逐键探测的音色库文件不存在。")
    mapping = inventory(path)
    preset = next((p for p in mapping["presets"]
                   if p["bank"] == bank and p["program"] == program), None)
    if preset is None:
        raise EngineError("invalid_preset", "逐键探测的真实预设不存在。")
    aliases = {}
    for key, value in hits.items():
        aliases.setdefault(int(value), []).append(key)
    results = []
    with _Synth(path, 48000, None) as synth:
        if synth.block != 64:
            raise EngineError("engine_grid_changed", "逐键探测要求已验证的64帧引擎块长。")
        channel = 9 if bank == 128 else 0
        _ok(synth.lib.fluid_synth_program_select(synth.synth, channel, synth.sfont_id,
                                                bank, program), "选择打击乐真实预设")
        _ok(synth.lib.fluid_synth_cc(synth.synth, channel, 91, 0), "关闭混响发送")
        _ok(synth.lib.fluid_synth_cc(synth.synth, channel, 93, 0), "关闭合唱发送")
        for midi in sorted(aliases):
            zones = [zone for zone in preset["mappings"] if zone["available_in_file"]
                     and zone["key_range"][0] <= midi <= zone["key_range"][1]
                     and zone["velocity_range"][0] <= 90 <= zone["velocity_range"][1]]
            result = {"hit_keys": sorted(aliases[midi]), "midi": midi, "velocity": 90,
                      "bank": bank, "program": program, "frames": 0,
                      "sample_rate_hz": 48000, "channels": 2,
                      "soundfont_sha256": mapping["source"]["sha256"],
                      "reverb_enabled": False, "chorus_enabled": False,
                      "sample_zone_ids": [zone["instrument_zone_id"] for zone in zones]}
            if not zones:
                result.update(status="失败", code="unmapped_hit", peak_linear=None,
                              reason="力度90下没有真实可发声区域；未调用音符起音。")
                results.append(result)
                continue
            _ok(synth.lib.fluid_synth_cc(synth.synth, channel, 120, 0), "逐键全停")
            cleared = synth.samples(64)
            clear_peak = float(np.max(np.abs(cleared)))
            result["cleared_frames"] = 64
            result["cleanup_peak_linear"] = clear_peak
            if clear_peak > 1e-8:
                result.update(status="失败", code="uncleared_audio", peak_linear=None,
                              reason="清空块仍有音频，不能把前键残音计入当前键。")
                results.append(result)
                continue
            _ok(synth.lib.fluid_synth_noteon(synth.synth, channel, midi, 90), "逐键起音")
            audio = synth.samples(4800)
            peak = float(np.max(np.abs(audio)))
            result.update(frames=len(audio), peak_linear=peak, finite=True,
                          window_float32_sha256=hashlib.sha256(
                              audio.astype("<f4", copy=False).tobytes()).hexdigest())
            if peak > 0.0001:
                result["status"] = "通过"
            else:
                result.update(status="失败", code="silent_hit",
                              reason="100ms干声窗口未达到0.0001线性峰值；不标为可用键。")
            results.append(result)
    return results


def _finite_number(value, minimum, maximum, field):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise EngineError("invalid_parameter", "参数必须为有限数值。", {"field": field})
    if not minimum <= value <= maximum:
        raise EngineError("invalid_parameter", "参数超出支持范围。", {"field": field})
    return value


def _reverb(mix, part_ids):
    effects = [e for e in mix.get("effects", []) if e.get("kind") != "chorus"]
    from delay_effect import validate_effects
    try:
        validate_effects(effects, part_ids)
    except ValueError as exc:
        raise EngineError("unsupported_effect", str(exc)) from exc
    reverbs = [e for e in effects if e.get("kind") == "reverb"]
    if not reverbs:
        return None
    effect = reverbs[0]
    targets = effect.get("target_part_ids", [])
    if not isinstance(targets, list) or any(target not in part_ids for target in targets):
        raise EngineError("invalid_effect_target", "混响目标必须引用现有声部。")
    values = {"room_size": 0.4, "damp": 0.3, "width": 0.8, "level": 0.2, "send": 45}
    parameters = effect.get("parameters", {})
    if set(parameters) - set(values):
        raise EngineError("unsupported_parameter", "混响参数存在未实现字段。",
                          {"fields": sorted(set(parameters) - set(values))})
    values.update(parameters)
    for name in ("room_size", "damp", "width", "level"):
        _finite_number(values[name], 0, 1, "reverb." + name)
    if type(values["send"]) is not int or not 0 <= values["send"] <= 127:
        raise EngineError("invalid_parameter", "混响 send 必须为 0 至 127 的整数。")
    return values


def _validate(document):
    rate, frames = document["sample_rate_hz"], document["total_frames"]
    if type(rate) is not int or not 8000 <= rate <= 96000:
        raise EngineError("invalid_format", "当前适配器支持 8 至 96 kHz 整数采样率。")
    if type(frames) is not int or not 1 <= frames <= rate * 600:
        raise EngineError("invalid_duration", "单次渲染时长必须为正，且不超过 600 秒。")
    parts = document["parts"]
    identifiers = [part["part_id"] for part in parts]
    if not parts or len(set(identifiers)) != len(identifiers):
        raise EngineError("invalid_parts", "声部不能为空，标识必须唯一。")
    for part in parts:
        from sound_shaping import validate_shaping
        try:
            validate_shaping(part.get("sound_shaping", {}))
        except ValueError as exc:
            raise EngineError("invalid_sound_shaping", str(exc)) from exc
        if type(part["bank"]) is not int or not 0 <= part["bank"] <= 16383:
            raise EngineError("invalid_preset", "音色库编号无效。")
        if type(part["program"]) is not int or not 0 <= part["program"] <= 127:
            raise EngineError("invalid_preset", "音色程序编号无效。")
        _finite_number(part.get("gain_db", 0), -96, 24, "part.gain_db")
        _finite_number(part.get("pan", 0), -1, 1, "part.pan")
    loop = document.get("render", {}).get("loop", {}).get("enabled", False)
    if loop and frames % 64:
        raise EngineError("loop_grid", "循环周期必须是 64 帧的整数倍，以保持引擎相位。")
    seen = set()
    intervals = {}
    for event in document["events"]:
        if event["part_id"] not in identifiers or event["event_id"] in seen:
            raise EngineError("invalid_event", "事件引用未知声部，或事件标识重复。")
        seen.add(event["event_id"])
        start, gate, midi, velocity = (event[k] for k in ("onset_frame", "gate_frames", "midi", "velocity"))
        if any(type(v) is not int for v in (start, gate, midi, velocity)):
            raise EngineError("invalid_event", "已解析事件的帧数、音高与力度必须为整数。")
        if start < 0 or gate <= 0 or start + gate > frames or not 0 <= midi <= 127 or not 1 <= velocity <= 127:
            raise EngineError("event_bounds", "事件起止、音高或力度越界。", {"event_id": event["event_id"]})
        if start % 64 or gate % 64:
            raise EngineError("event_grid", "事件起止须由编译器量化到 64 帧网格。",
                              {"event_id": event["event_id"]})
        dedicated = "performance" in event or "program" in event
        if event.get("cents", 0) != 0 or (event.get("articulation", "normal") != "normal" and not dedicated):
            raise EngineError("unsupported_articulation", "音分偏移须写入演奏曲线，非普通奏法须绑定真实预设。")
        if dedicated:
            from note_expression import validate_resolved
            try:
                validate_resolved(event)
            except ValueError as exc:
                raise EngineError("invalid_performance", str(exc), {"event_id": event["event_id"]}) from exc
            if "program" in event and (type(event["program"]) is not int or not 0 <= event["program"] <= 127
                                       or type(event.get("bank", 0)) is not int or not 0 <= event.get("bank", 0) <= 16383):
                raise EngineError("invalid_preset", "逐音奏法的预设编号无效。", {"event_id": event["event_id"]})
        if event.get("gain_db", 0) != 0 or event.get("pan") is not None:
            raise EngineError("unsupported_note_mix", "当前适配器只支持声部级增益与声像。")
        if not dedicated:
            intervals.setdefault((event["part_id"], midi), []).append((start, start + gate))
    for key, spans in intervals.items():
        spans.sort()
        if any(left[1] > right[0] for left, right in zip(spans, spans[1:])):
            raise EngineError("same_pitch_overlap", "同声部同音高的门控重叠不能可靠表达。",
                              {"part_id": key[0], "midi": key[1]})
    controls = document.get("control_events", [])
    if type(controls) is not list:
        raise EngineError("invalid_control_event", "控制事件必须是列表。")
    if controls and loop:
        raise EngineError("unsupported_control_loop", "当前帧级CC控制不支持循环；不能隐式重置跨周期持音表情。")
    positions = set()
    for control in controls:
        if (type(control) is not dict or set(control) !=
                {"event_id", "part_id", "frame", "controller", "value"}):
            raise EngineError("invalid_control_event", "控制事件字段缺失或含未知字段。")
        identifier, part_id = control["event_id"], control["part_id"]
        if (type(identifier) is not str or not identifier.strip() or "\0" in identifier
                or identifier in seen or type(part_id) is not str or part_id not in identifiers):
            raise EngineError("invalid_control_event", "控制标识须非空且与音符全局唯一，并引用现有声部。")
        seen.add(identifier)
        if type(control["controller"]) is not int or control["controller"] != 11:
            raise EngineError("unsupported_controller", "当前只核准CC11持音音量表情，不接受其他控制器。")
        frame, value = control["frame"], control["value"]
        if type(frame) is not int or not 0 <= frame < frames:
            raise EngineError("control_bounds", "控制帧须为输出范围内的非负整数，不含末帧边界。")
        if frame % 64:
            raise EngineError("control_grid", "控制事件须对齐64帧网格。")
        if type(value) is not int or not 0 <= value <= 127:
            raise EngineError("invalid_control_event", "CC11值须为0至127的整数。")
        position = (part_id, control["controller"], frame)
        if position in positions:
            raise EngineError("invalid_control_event", "同声部同控制器同帧不能重复指定值。")
        positions.add(position)
    _finite_number(document.get("mix", {}).get("master_gain_db", 0), -96, 0, "master_gain_db")
    return loop


DEDICATED_CHANNELS = 128
CHANNEL_HOLD_SECONDS = 1


def _dedicated_part(events):
    return any("performance" in e or "program" in e for e in events)


def _channel_plan(events, rate):
    """每音独占通道：音高轮和表情只影响这一音。通道在释音结束后才回收。"""
    pool = [channel for channel in range(1, DEDICATED_CHANNELS) if channel != 9]
    free_at = {channel: 0 for channel in pool}
    plan = {}
    for start, event in sorted(events, key=lambda item: (item[0], item[1]["event_id"])):
        channel = next((c for c in pool if free_at[c] <= start), None)
        if channel is None:
            raise EngineError("channel_exhausted", "同时存活音符超过独立通道上限。", {"event_id": event["event_id"]})
        plan[(start, event["event_id"])] = channel
        free_at[channel] = start + event["gate_frames"] + CHANNEL_HOLD_SECONDS * rate
    return plan


def _play_dedicated(synth, part, events, repetitions, frames, capture, rate, reverb_send, shaping_values):
    from note_expression import BEND_RANGE_SEMITONES, bend_value
    lib, handle = synth.lib, synth.synth
    expanded = [(event["onset_frame"] + repeat * frames, event) for repeat in range(repetitions) for event in events]
    plan = _channel_plan(expanded, rate)
    used = sorted(set(plan.values()))
    for channel in used:
        _ok(lib.fluid_synth_program_select(handle, channel, synth.sfont_id, part["bank"], part["program"]), "选择真实预设")
        _ok(lib.fluid_synth_pitch_wheel_sens(handle, channel, BEND_RANGE_SEMITONES), "设置音高轮范围")
        _ok(lib.fluid_synth_cc(handle, channel, 91, reverb_send), "混响发送")
        if shaping_values:
            from sound_shaping import apply_shaping
            apply_shaping(synth, channel, shaping_values)
    scheduled = []
    for sequence, (start, event) in enumerate(sorted(expanded, key=lambda item: (item[0], item[1]["event_id"]))):
        channel = plan[(start, event["event_id"])]
        performance = event.get("performance", {})
        pitch, expression = performance.get("pitch_points", []), performance.get("expression_points", [])
        setup = (event.get("bank", part["bank"]), event.get("program", part["program"]),
                 pitch[0][1] if pitch and pitch[0][0] == 0 else 0,
                 expression[0][1] if expression and expression[0][0] == 0 else 127,
                 performance.get("sample_start_frames", 0))
        scheduled.append((start, 1, sequence, "setup", channel, setup))
        scheduled.append((start, 2, sequence, "on", channel, (event["midi"], event["velocity"])))
        scheduled.append((start + event["gate_frames"], 0, sequence, "off", channel, event["midi"]))
        for offset, cents in pitch:
            if offset:
                scheduled.append((start + offset, 1, sequence, "bend", channel, cents))
        for offset, value in expression:
            if offset:
                scheduled.append((start + offset, 1, sequence, "cc", channel, value))
    scheduled.sort(key=lambda item: item[:3])
    output = np.zeros((capture, 2), dtype=np.float32)
    cursor = 0
    inactive = []
    for frame, _, _, action, channel, payload in scheduled:
        if frame >= capture:
            continue
        if frame > cursor:
            output[cursor:frame] = synth.samples(frame - cursor)
            cursor = frame
        if action == "setup":
            bank, program, cents, value, start_frames = payload
            _ok(lib.fluid_synth_program_select(handle, channel, synth.sfont_id, bank, program), "选择奏法预设")
            _ok(lib.fluid_synth_pitch_bend(handle, channel, bend_value(cents)), "起音音高")
            _ok(lib.fluid_synth_cc(handle, channel, 11, value), "起音表情")
            _ok(lib.fluid_synth_set_gen(handle, channel, 0, float(start_frames)), "采样起点")
        elif action == "on":
            _ok(lib.fluid_synth_noteon(handle, channel, *payload), "音符起音")
        elif action == "bend":
            _ok(lib.fluid_synth_pitch_bend(handle, channel, bend_value(payload)), "音高曲线")
        elif action == "cc":
            _ok(lib.fluid_synth_cc(handle, channel, 11, payload), "表情曲线")
        elif synth.noteoff(channel, payload):
            inactive.append({"frame": frame, "channel": channel, "midi": payload, "return_code": -1,
                             "confirmed_on_voice_count": 0})
    if cursor < capture:
        output[cursor:] = synth.samples(capture - cursor)
    return output, inactive, {"dedicated_events": len(events), "channels_used": len(used),
                              "control_points": sum(1 for item in scheduled if item[3] in ("bend", "cc"))}


def render_events(document, soundfont_path, font_paths=None):
    """返回最终浮点立体声及诊断；不量化、不归一化、不写文件。

    循环连续演奏三个周期，丢弃前两个预热周期，保留第三周期。
    选中声部使用独立但参数相同的线性混响，随后施加增益和立体声平衡。
    可选control_events仅支持非循环CC11，字段为event_id/part_id/frame/controller/value。
    同帧固定先noteoff，再CC，最后noteon；CC作用于该声部全部活动音符。
    """
    loop = _validate(document)
    from delay_effect import validate_effects, apply_delay
    from chorus_effect import validate_chorus_effects, chorus_parameters, apply_chorus
    part_ids = {p["part_id"] for p in document["parts"]}
    for effect in document.get("mix", {}).get("effects", []):
        targets = effect.get("target_part_ids", []) if isinstance(effect, dict) else None
        if not isinstance(targets, list) or any(type(target) is not str or target not in part_ids for target in targets):
            raise EngineError("invalid_effect_target", "效果目标必须引用现有声部。")
    try:
        effects = document.get("mix", {}).get("effects", [])
        chorus = validate_chorus_effects(effects, part_ids, loop)
        validate_effects([e for e in effects if e.get("kind") != "chorus"], part_ids, loop,
                         sample_rate=document['sample_rate_hz'])
    except ValueError as exc:
        raise EngineError("unsupported_effect", str(exc)) from exc
    rate, frames = document["sample_rate_hz"], document["total_frames"]
    reverb = _reverb(document.get("mix", {}), {p["part_id"] for p in document["parts"]})
    reverb_targets = next((e.get("target_part_ids", []) for e in effects if e.get("kind") == "reverb"), [])
    chorus_targets = chorus.get("target_part_ids", []) if chorus else []
    repetitions = 3 if loop else 1
    capture_frames = repetitions * frames
    result = np.zeros((frames, 2), dtype=np.float64)
    evidence = []
    verified = {}
    fonts = font_paths or {}

    def checked_hash(path):
        path = Path(path).resolve()
        if path not in verified:
            if not path.is_file():
                raise EngineError("soundfont_missing", "所需音色库文件不存在。")
            digest = hashlib.sha256()
            with path.open("rb") as source:
                for data in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(data)
            verified[path] = digest.hexdigest()
        return verified[path]

    def part_font(part):
        expected = part.get("resource_sha256", document.get("asset_sha256"))
        if expected is None:
            return soundfont_path
        if expected in fonts:
            path = fonts[expected]
        else:
            if checked_hash(soundfont_path) != expected:
                if expected == document.get("asset_sha256"):
                    raise EngineError("soundfont_hash_mismatch", "主音色库内容与冻结摘要不一致。")
                raise EngineError("soundfont_missing", "声部绑定的额外音色库未显式提供。",
                                  {"part_id": part["part_id"], "sha256": expected})
            path = soundfont_path
        if checked_hash(path) != expected:
            raise EngineError("soundfont_hash_mismatch", "声部音色库内容与冻结摘要不一致。",
                              {"part_id": part["part_id"], "sha256": expected})
        return path

    master = 10 ** (document.get("mix", {}).get("master_gain_db", 0) / 20)
    for part in document["parts"]:
        selected_font = part_font(part)
        part_reverb = reverb if not reverb_targets or part["part_id"] in reverb_targets else None
        events = [e for e in document["events"] if e["part_id"] == part["part_id"]]
        controls = [e for e in document.get("control_events", []) if e["part_id"] == part["part_id"]]
        dedicated = _dedicated_part(events)
        if dedicated and (controls or part["part_id"] in chorus_targets or (chorus and not chorus_targets)):
            raise EngineError("unsupported_performance_mix", "逐音演奏声部不能同时使用声部级CC11或原生合唱；请用expression_curve。",
                              {"part_id": part["part_id"]})
        control_ids = {(e["frame"], e["controller"]): e["event_id"] for e in controls}
        scheduled = []
        for repeat in range(repetitions):
            for event in events:
                start = event["onset_frame"] + repeat * frames
                scheduled.append((start, 2, event["midi"], event["velocity"]))
                scheduled.append((start + event["gate_frames"], 0, event["midi"], 0))
        scheduled.extend((e["frame"], 1, e["controller"], e["value"]) for e in controls)
        performance_diagnostics = None
        # 同帧先停止旧音，再施加表情控制，最后触发新音；同类事件顺序稳定。
        scheduled.sort()
        output = np.zeros((capture_frames, 2), dtype=np.float32)
        inactive_noteoffs = []
        chorus_diagnostics = None
        control_diagnostics = []
        with _Synth(selected_font, rate, part_reverb, DEDICATED_CHANNELS if dedicated else None) as synth:
            if synth.block != 64:
                raise EngineError("engine_grid_changed", "引擎内部块长变化，需要重新编译适配。",
                                  {"actual_block_frames": synth.block})
            channel = 9 if part["bank"] == 128 else 0
            if dedicated:
                output, inactive_noteoffs, performance_diagnostics = _play_dedicated(
                    synth, part, events, repetitions, frames, capture_frames, rate,
                    part_reverb["send"] if part_reverb else 0, part.get("sound_shaping"))
                scheduled = []
            _ok(synth.lib.fluid_synth_program_select(synth.synth, channel, synth.sfont_id,
                                                    part["bank"], part["program"]), "选择真实预设")
            shaping = {}
            if part.get("sound_shaping") and dedicated:
                shaping = dict(part["sound_shaping"])
            elif part.get("sound_shaping"):
                from sound_shaping import apply_shaping
                try:
                    shaping = apply_shaping(synth, channel, part["sound_shaping"])
                except ValueError as exc:
                    raise EngineError("invalid_sound_shaping", str(exc)) from exc
            _ok(synth.lib.fluid_synth_cc(synth.synth, channel, 91,
                                       part_reverb["send"] if part_reverb else 0), "混响发送")
            if chorus and (not chorus_targets or part["part_id"] in chorus_targets):
                try:
                    chorus_diagnostics = apply_chorus(synth, chorus_parameters(chorus), channel)
                except ValueError as exc:
                    raise EngineError("native_chorus_failed", str(exc)) from exc
            cursor = capture_frames if dedicated else 0
            for frame, action, midi, velocity in scheduled:
                if frame > cursor:
                    output[cursor:frame] = synth.samples(frame - cursor)
                    cursor = frame
                if action == 2:
                    _ok(synth.lib.fluid_synth_noteon(synth.synth, channel, midi, velocity), "音符起音")
                elif action == 1:
                    _ok(synth.lib.fluid_synth_cc(synth.synth, channel, midi, velocity), "持续音CC11控制")
                    observed = ct.c_int(-1)
                    _ok(synth.lib.fluid_synth_get_cc(synth.synth, channel, midi,
                                                   ct.byref(observed)), "读回持续音CC11控制")
                    if observed.value != velocity:
                        raise EngineError("control_readback_mismatch", "实际CC11读回与冻结控制值不一致。",
                                          {"part_id": part["part_id"], "frame": frame,
                                           "requested": velocity, "actual": observed.value})
                    control_diagnostics.append({"event_id": control_ids[frame, midi],
                                                "frame": frame, "controller": midi, "value": velocity,
                                                "channel": channel, "readback": observed.value})
                else:
                    if synth.noteoff(channel, midi):
                        inactive_noteoffs.append({"frame": frame, "channel": channel,
                                                  "midi": midi, "return_code": -1,
                                                  "confirmed_on_voice_count": 0})
            if cursor < capture_frames:
                output[cursor:] = synth.samples(capture_frames - cursor)
        stem = output[-frames:].astype(np.float64)
        delay_diagnostics = []
        duck_diagnostics = []
        eq_diagnostics = []
        for effect in document.get("mix", {}).get("effects", []):
            if effect['kind'] == 'eq' and part['part_id'] in effect['target_part_ids']:
                from eq_effect import apply_eq
                try:
                    stem, details = apply_eq(stem, rate, effect)
                except ValueError as exc:
                    raise EngineError('eq_processing_failed', str(exc)) from exc
                eq_diagnostics.append({'effect_id': effect['effect_id'], **details})
            if effect["kind"] == "delay" and (not effect.get("target_part_ids") or part["part_id"] in effect["target_part_ids"]):
                stem, details = apply_delay(stem, rate, effect)
                delay_diagnostics.append({"effect_id": effect["effect_id"], **details})
            if effect["kind"] == "duck" and part["part_id"] in effect["target_part_ids"]:
                from ducking_effect import apply_duck
                stem, details = apply_duck(stem, rate, effect, document["events"])
                duck_diagnostics.append({"effect_id": effect["effect_id"], **details})
        pan = part.get("pan", 0)
        balance = np.array([1 - max(pan, 0), 1 + min(pan, 0)])
        stem *= balance * (10 ** (part.get("gain_db", 0) / 20)) * master
        result += stem
        evidence.append({"part_id": part["part_id"], "event_count": len(events),
                         "bank": part["bank"], "program": part["program"],
                         "resource_sha256": part.get("resource_sha256", document.get("asset_sha256")),
                         "peak_linear": float(np.max(np.abs(stem))),
                         "rms_linear": float(np.sqrt(np.mean(stem ** 2))),
                         "noteoff_without_active_voice": inactive_noteoffs,
                         "includes_shared_settings_reverb": part_reverb is not None,
                         "delay_effects": delay_diagnostics,
                         "isolation_method": "独立合成器声部真实渲染，混音前测量"})
        if duck_diagnostics:
            evidence[-1]['duck_effects'] = duck_diagnostics
        if eq_diagnostics:
            evidence[-1]['eq_effects'] = eq_diagnostics
        if shaping:
            evidence[-1]["sound_shaping"] = shaping
        if chorus_diagnostics is not None:
            evidence[-1]["chorus"] = {"effect_id": chorus["effect_id"], **chorus_diagnostics}
        if control_diagnostics:
            evidence[-1]["control_events"] = control_diagnostics
        if performance_diagnostics:
            evidence[-1]["performance"] = performance_diagnostics
    if not np.isfinite(result).all():
        raise EngineError("non_finite_audio", "量化前混音包含非有限数。")
    return result, {**engine_info(), "parts": evidence, "reverb": reverb,
                    "reverb_target_part_ids": reverb_targets,
                    "verified_asset_sha256": sorted(set(verified.values())),
                    "internal_block_frames": 64, "prequantization_finite": True,
                    "peak_linear": float(np.max(np.abs(result))),
                    "automatic_normalization": False,
                    "pan_algorithm": "保留原立体声的线性左右平衡，仅衰减反侧",
                    "loop": {"enabled": loop, "continuous_cycles": repetitions,
                             "retained_cycle": repetitions if loop else None,
                             "crossfade_applied": False,
                             "seamless_claim": "需另测首尾和连续试听；预热本身不证明无缝"}}
