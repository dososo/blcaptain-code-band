"""逐音演奏控制：把谱面音高曲线、揉弦和表情曲线解析成冻结的帧级控制。

这里只计算确定性的控制点，不选采样、不抽随机数。音高轮作用于已录制的采样，
能做出按滑、吟揉这类“同一发音的音高变化”，但不等于录制过的连音过渡。
"""
import math

POINT_FRAMES = 256
BEND_RANGE_SEMITONES = 12
MAX_CURVE_POINTS = 64
SAMPLE_START_LIMIT_MS = 1000


def _number(value, low, high, field):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{field} 越界或不是有限数值")
    return value


def _curve(points, duration_ticks, key, low, high, field):
    if not isinstance(points, list) or not 1 <= len(points) <= MAX_CURVE_POINTS:
        raise ValueError(f"{field} 须为1至{MAX_CURVE_POINTS}个控制点")
    previous = -1
    for point in points:
        if not isinstance(point, dict) or set(point) != {"at_tick", key}:
            raise ValueError(f"{field} 控制点字段无效")
        tick = point["at_tick"]
        if type(tick) is not int or tick <= previous and not (tick == 0 and previous == -1) or tick > duration_ticks:
            raise ValueError(f"{field} 控制点须严格递增且不超出音符时值")
        previous = tick
        _number(point[key], low, high, field)


def validate_performance(performance, duration_ticks):
    """谱面阶段校验；未知字段直接拒绝，不静默忽略。"""
    allowed = {"technique", "pitch_curve", "vibrato", "expression_curve", "sample_start_ms"}
    if not isinstance(performance, dict) or not performance or not set(performance) <= allowed:
        raise ValueError("performance 须为含已知字段的非空对象")
    if "technique" in performance and (type(performance["technique"]) is not str or not performance["technique"].strip()):
        raise ValueError("technique 须为非空说明")
    if "pitch_curve" in performance:
        _curve(performance["pitch_curve"], duration_ticks, "cents", -BEND_RANGE_SEMITONES * 100,
               BEND_RANGE_SEMITONES * 100, "pitch_curve")
    if "expression_curve" in performance:
        _curve(performance["expression_curve"], duration_ticks, "value", 0, 127, "expression_curve")
    if "vibrato" in performance:
        vibrato = performance["vibrato"]
        if not isinstance(vibrato, dict) or set(vibrato) - {"start_tick", "rate_hz", "depth_cents", "fade_ticks", "end_depth_cents"} \
                or not {"start_tick", "rate_hz", "depth_cents", "fade_ticks"} <= set(vibrato):
            raise ValueError("vibrato 字段无效")
        for name in ("start_tick", "fade_ticks"):
            if type(vibrato[name]) is not int or not 0 <= vibrato[name] <= duration_ticks:
                raise ValueError(f"vibrato.{name} 超出音符时值")
        _number(vibrato["rate_hz"], .5, 12, "vibrato.rate_hz")
        _number(vibrato["depth_cents"], 0, 300, "vibrato.depth_cents")
        if "end_depth_cents" in vibrato:
            _number(vibrato["end_depth_cents"], 0, 300, "vibrato.end_depth_cents")
    if "sample_start_ms" in performance:
        _number(performance["sample_start_ms"], 0, SAMPLE_START_LIMIT_MS, "sample_start_ms")
    return performance


def _interpolate(points, tick, key):
    if tick <= points[0]["at_tick"]:
        return float(points[0][key])
    for left, right in zip(points, points[1:]):
        if tick <= right["at_tick"]:
            span = right["at_tick"] - left["at_tick"]
            return left[key] + (right[key] - left[key]) * (tick - left["at_tick"]) / span
    return float(points[-1][key])


def resolve_performance(performance, base_cents, onset_tick, duration_ticks, gate_frames, tick_frame, rate):
    """返回冻结控制：帧偏移相对音符起音，对齐64帧；同一输入永远得到同一结果。"""
    if performance:
        validate_performance(performance, duration_ticks)
    start = tick_frame(onset_tick)
    span = max(tick_frame(onset_tick + duration_ticks) - start, 1)
    vibrato = performance.get("vibrato")
    pitch_points, expression_points = [], []
    for offset in range(0, gate_frames + 1, POINT_FRAMES):
        tick = min(offset / span, 1.0) * duration_ticks
        cents = float(base_cents)
        if "pitch_curve" in performance:
            cents += _interpolate(performance["pitch_curve"], tick, "cents")
        if vibrato and tick >= vibrato["start_tick"]:
            age = tick - vibrato["start_tick"]
            fade = 1.0 if vibrato["fade_ticks"] == 0 else min(age / vibrato["fade_ticks"], 1.0)
            remaining = duration_ticks - vibrato["start_tick"]
            depth = vibrato["depth_cents"]
            if "end_depth_cents" in vibrato and remaining > 0:
                depth += (vibrato["end_depth_cents"] - depth) * age / remaining
            seconds = (tick_frame(onset_tick + int(tick)) - tick_frame(onset_tick + vibrato["start_tick"])) / rate
            cents += depth * fade * math.sin(2 * math.pi * vibrato["rate_hz"] * seconds)
        pitch_points.append([offset, round(cents, 2)])
        if "expression_curve" in performance:
            expression_points.append([offset, int(round(_interpolate(performance["expression_curve"], tick, "value")))])
    resolved = {}
    if "technique" in performance:
        resolved["technique"] = performance["technique"]
    if "pitch_curve" in performance or vibrato or base_cents:
        resolved["pitch_points"] = _thin(pitch_points)
    if expression_points:
        resolved["expression_points"] = _thin(expression_points)
    if performance.get("sample_start_ms"):
        resolved["sample_start_ms"] = performance["sample_start_ms"]
    return resolved


def _thin(points):
    """去掉与前一控制点相同的值，保留首点，避免无意义的控制调用。"""
    kept = [points[0]]
    for point in points[1:]:
        if point[1] != kept[-1][1]:
            kept.append(point)
    return kept


def bend_value(cents):
    value = 8192 + round(cents / (BEND_RANGE_SEMITONES * 100) * 8192)
    return max(0, min(16383, value))


def validate_resolved(event):
    """引擎入口再查一次冻结事件，手写或损坏的 events.json 不能绕过。"""
    performance = event.get("performance")
    if performance is None:
        return
    if not isinstance(performance, dict) or set(performance) - {"technique", "pitch_points", "expression_points", "sample_start_ms", "sample_start_frames"}:
        raise ValueError("冻结演奏控制含未知字段")
    for name, low, high in (("pitch_points", -1200, 1200), ("expression_points", 0, 127)):
        points = performance.get(name, [])
        if not isinstance(points, list):
            raise ValueError("冻结控制点须为列表")
        previous = -1
        for point in points:
            if (not isinstance(point, list) or len(point) != 2 or type(point[0]) is not int or point[0] <= previous
                    or point[0] % 64 or point[0] > event["gate_frames"]
                    or isinstance(point[1], bool) or not isinstance(point[1], (int, float))
                    or not math.isfinite(point[1]) or not low <= point[1] <= high):
                raise ValueError("冻结控制点越界、未对齐64帧或不递增")
            previous = point[0]
    if "sample_start_frames" in performance and (type(performance["sample_start_frames"]) is not int or performance["sample_start_frames"] < 0):
        raise ValueError("采样起点偏移必须为非负整数帧")


def add_expression_capability(result, soundfont):
    """真实渲染探测：音高轮必须真的改变基频，两次渲染必须逐样相同。"""
    import numpy as np
    from fluid_engine import render_events
    from soundfont_inventory import inventory
    cap = result["capabilities"]
    mapping = inventory(soundfont)
    preset = next((p for p in mapping["presets"] if p["bank"] == 0 and p["program"] == 73), None)
    if preset is None or not any(z["available_in_file"] and z["key_range"][0] <= 69 <= z["key_range"][1]
                                 and z["velocity_range"][0] <= 90 <= z["velocity_range"][1] for z in preset["mappings"]):
        cap["performance"]["note_expression"] = {"state": "unknown", "evidence_ids": [],
            "limitations": ["当前音色库缺少探测用的代表音色，未验证逐音控制。"]}
        result["note_expression_probe"] = {"status": "未实测"}
        return result

    def document(points):
        event = {"event_id": "bend", "part_id": "p", "onset_frame": 0, "gate_frames": 43200, "midi": 69, "velocity": 90}
        if points:
            event["performance"] = {"pitch_points": points}
        return {"sample_rate_hz": 48000, "total_frames": 48000, "parts": [{"part_id": "p", "bank": 0, "program": 73, "gain_db": 0, "pan": 0}],
                "events": [event], "mix": {"master_gain_db": 0, "effects": []}, "render": {"loop": {"enabled": False}}}

    def frequency(audio):
        window = audio[12000:36000].mean(axis=1) * np.hanning(24000)
        spectrum = np.abs(np.fft.rfft(window, 1 << 19))
        bins = np.fft.rfftfreq(1 << 19, 1 / 48000)
        band = (bins > 300) & (bins < 700)
        return float(bins[band][np.argmax(spectrum[band])])

    flat, _ = render_events(document(None), soundfont)
    bent, diagnostic = render_events(document([[0, 100.0]]), soundfont)
    again, _ = render_events(document([[0, 100.0]]), soundfont)
    measured = 1200 * math.log2(frequency(bent) / frequency(flat))
    if abs(measured - 100) > 5 or not np.array_equal(bent, again):
        raise ValueError("逐音音高控制实测失败，不能声明支持")
    identifier = "native_note_expression_probe"
    cap["performance"]["note_expression"] = {"state": "supported", "evidence_ids": [identifier],
        "limitations": ["音高轮和CC11作用于已录制采样；滑音、吟揉、轮指是控制近似或重复触发，不等于录制过的连音过渡。",
                        "每音占独立MIDI通道，单声部同时存活上限127音。"]}
    cap["evidence"] = [e for e in cap["evidence"] if e["evidence_id"] != identifier]
    cap["evidence"].append({"evidence_id": identifier, "kind": "command_output", "reference": "doctor JSON的note_expression_probe",
                            "description": "真实音色+100音分渲染，基频实测与重复渲染一致性。"})
    result["note_expression_probe"] = {"measured_cents": round(measured, 2), "float_repeat_equal": True,
                                       "dedicated_channels": diagnostic["parts"][0]["performance"]["channels_used"]}
    return result
