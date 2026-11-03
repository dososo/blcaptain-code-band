"""民族器乐语汇：把一个演奏动作展开成显式音符和演奏控制。

输出仍是普通谱面音符，可以逐个改。这里的参数是起始值：录音实测支持的写在 MEASURED，
其余是按常见演奏习惯给的假设，须经人耳确认后再当默认。
"""

# 来自 Pufermufin 古筝全量录音的三处按滑候选（evidence/guzheng/calibration.json），未经人工核对。
MEASURED = {"guzheng_slide_seconds": .7}
# 古琴吟猱与绰：GuqinSonGest（CC BY 4.0）逐条测量的中位数，见
# work/主流乐团完善_20261103/guqin/evidence/calibration_v2.json。古琴吟猱远比古筝吟揉慢而宽，
# 不能用下面 ASSUMED 里的 5 Hz/18 音分。偷吟、全扶在 9 条里只测到 1–3 条有周期摆动，样本少，只作参考。
GUQIN_MEASURED = {
    # 承声吟：弹响后立刻摆动；6/9 条可测
    "chengsheng": {"rate_hz": 2.47, "start_seconds": .14, "depth_cents": 50, "end_depth_cents": 44},
    # 声后吟：先稳住约 0.9 秒再摆；3/9 条可测，深度明显衰减
    "shenghou": {"rate_hz": 2.63, "start_seconds": .87, "depth_cents": 52, "end_depth_cents": 33},
    "quanfu": {"rate_hz": 1.75, "start_seconds": .77, "depth_cents": 19, "end_depth_cents": 16},
    "touyin": {"rate_hz": 1.45, "start_seconds": .32, "depth_cents": 15, "end_depth_cents": 12},
}
# 绰（自下滑入）：录音是徽位间一个全音的滑入；到达目标 ±25 音分内所需时间。快 20 条、慢 13 条。
GUQIN_CHUO_SECONDS = {"fast": .44, "slow": .99}
# 假设值：录音中12个可测长音都没有明显吟揉，无法校准。
ASSUMED = {"yinrou_rate_hz": 5.0, "yinrou_depth_cents": 18, "erhu_vibrato_rate_hz": 5.8, "erhu_vibrato_depth_cents": 28,
           "dizi_vibrato_rate_hz": 5.2, "dizi_vibrato_depth_cents": 16, "tremolo_rate_hz": 9.0}


def note(note_id, tick, duration, midi, velocity, performance=None, articulation="normal", cents=0, accent="none"):
    result = {"note_id": note_id, "onset_tick": round(tick), "duration_ticks": round(duration),
              "pitch": {"kind": "pitched", "midi": midi, "cents": cents}, "velocity": velocity,
              "articulation": articulation, "timing_locked": True, "accent": accent}
    if performance:
        result["performance"] = performance
    return result


def _perform(result):
    return result.setdefault("performance", {"technique": "pluck"})


def _named(result, name):
    performance = _perform(result)
    previous = performance.get("technique")
    performance["technique"] = name if previous in (None, "pluck") else f"{previous}+{name}"[:64]
    return performance


def press_up(result, cents, start_ticks, end_ticks, hold=True):
    """上滑音：弹响后按弦把音高推上去。谱面音高写弦的空弦音。"""
    performance = _named(result, "press_up")
    points = [{"at_tick": 0, "cents": 0}]
    if start_ticks > 0:
        points.append({"at_tick": round(start_ticks), "cents": 0})
    points.append({"at_tick": round(end_ticks), "cents": cents})
    if not hold:
        back = min(round(end_ticks * 2 - start_ticks), result["duration_ticks"])
        points.append({"at_tick": back, "cents": 0})
    performance["pitch_curve"] = points
    return result


def press_release(result, cents, start_ticks, end_ticks):
    """下滑音：先按好再弹，弹响后松回。"""
    performance = _named(result, "press_release")
    performance["pitch_curve"] = [{"at_tick": 0, "cents": cents}, {"at_tick": round(start_ticks), "cents": cents},
                                  {"at_tick": round(end_ticks), "cents": 0}]
    return result


def pressed(result, cents):
    """按音：全程按住，用于弦上没有的音，或用尾音更长的相邻弦代替短采样。"""
    performance = _named(result, "pressed")
    performance["pitch_curve"] = [{"at_tick": 0, "cents": cents}]
    return result


def vibrato(result, start_ticks, fade_ticks, rate_hz=None, depth_cents=None, end_depth_cents=None):
    performance = _named(result, "vibrato")
    performance["vibrato"] = {"start_tick": round(start_ticks), "fade_ticks": round(fade_ticks),
                              "rate_hz": rate_hz or ASSUMED["yinrou_rate_hz"],
                              "depth_cents": ASSUMED["yinrou_depth_cents"] if depth_cents is None else depth_cents}
    if end_depth_cents is not None:
        performance["vibrato"]["end_depth_cents"] = end_depth_cents
    return result


def guqin_yin(result, kind, ticks_per_second, fade_seconds=.15):
    """古琴吟猱（实测参数）。kind 取 GUQIN_MEASURED 的键；摆动从实测起摆时间开始，深度按实测衰减。

    摆动方向也按实测：吟（承声、声后、偷吟）从本音往上摆，下沿在本音上；全扶往下摆，上沿在本音上。
    所以正弦摆动之外再叠一条随深度变化的音高偏移。已有音高曲线（如先绰后吟）时，偏移接在曲线末端之后。
    """
    spec = GUQIN_MEASURED[kind]
    duration = result["duration_ticks"]
    start = min(round(spec["start_seconds"] * ticks_per_second), duration)
    fade = min(round(fade_seconds * ticks_per_second), duration - start)
    vibrato(result, start, fade, spec["rate_hz"], spec["depth_cents"], spec["end_depth_cents"])
    performance = result["performance"]
    performance["technique"] = performance["technique"].replace("vibrato", f"guqin_{kind}")
    sign = -1 if kind == "quanfu" else 1
    curve = performance.get("pitch_curve", [{"at_tick": 0, "cents": 0}])
    base = curve[-1]["cents"]
    begin = max(start, curve[-1]["at_tick"])
    points = [{"at_tick": begin, "cents": base}, {"at_tick": min(begin + fade, duration), "cents": base + sign * spec["depth_cents"]},
              {"at_tick": duration, "cents": base + sign * spec["end_depth_cents"]}]
    for point in points:
        if point["at_tick"] > curve[-1]["at_tick"]:
            curve.append(point)
        elif point["at_tick"] == curve[-1]["at_tick"]:
            curve[-1] = point
    performance["pitch_curve"] = curve
    return result


def guqin_chuo(result, ticks_per_second, speed="fast", from_cents=-200):
    """古琴绰：从下方 from_cents 处滑入目标音，滑行时间取实测（快 0.44 秒、慢 0.99 秒）。"""
    performance = _named(result, f"guqin_chuo_{speed}")
    end = min(round(GUQIN_CHUO_SECONDS[speed] * ticks_per_second), result["duration_ticks"])
    performance["pitch_curve"] = [{"at_tick": 0, "cents": from_cents}, {"at_tick": end, "cents": 0}]
    return result


def tremolo(prefix, tick, duration, midi, start_velocity, end_velocity, ticks_per_second, rate_hz=None, ring_ticks=None, cents=0):
    """摇指：真实重复拨弦。交替强弱模拟托抹，不是把单音做成颤音。"""
    step = ticks_per_second / (rate_hz or ASSUMED["tremolo_rate_hz"])
    count = max(int(duration / step), 2)
    notes = []
    for index in range(count):
        position = index / (count - 1)
        velocity = start_velocity + (end_velocity - start_velocity) * position
        velocity = round(velocity + (5 if index % 2 == 0 else -5))
        ring = ring_ticks if index == count - 1 and ring_ticks else step * 2.2
        item = note(f"{prefix}_{index:03d}", tick + index * step, ring, midi, max(1, min(127, velocity)), {"technique": "tremolo"})
        if cents:
            item["performance"]["pitch_curve"] = [{"at_tick": 0, "cents": cents}]
        notes.append(item)
    return notes


def run(prefix, tick, scale, step_ticks, start_velocity, end_velocity, ring_ticks, last_ring_ticks=None):
    """花指/刮奏：沿弦序连续拨过，音高由 scale 明确给出。"""
    notes = []
    for index, midi in enumerate(scale):
        position = index / max(len(scale) - 1, 1)
        velocity = round(start_velocity + (end_velocity - start_velocity) * position)
        ring = last_ring_ticks if index == len(scale) - 1 and last_ring_ticks else ring_ticks
        notes.append(note(f"{prefix}_{index:02d}", tick + index * step_ticks, ring, midi, velocity, {"technique": "glissando"}))
    return notes


def slur_to(result, previous_midi, duration_ticks, sample_start_ms=120, fade_ticks=60, slide_ticks=0):
    """合成连音：跳过采样的换弓起音并淡入。这是控制近似，不是录制的连弓过渡。"""
    performance = _named(result, "slur")
    performance["sample_start_ms"] = sample_start_ms
    performance["expression_curve"] = [{"at_tick": 0, "value": 40}, {"at_tick": round(fade_ticks), "value": 127}]
    if slide_ticks:
        performance["pitch_curve"] = [{"at_tick": 0, "cents": (previous_midi - result["pitch"]["midi"]) * 100}, {"at_tick": round(slide_ticks), "cents": 0}]
    return result


def fade_out(result, ticks):
    """给连音的上一音收尾，避免和下一音叠响。"""
    performance = _perform(result)
    curve = performance.setdefault("expression_curve", [{"at_tick": 0, "value": 127}])
    end = result["duration_ticks"]
    start = max(end - round(ticks), curve[-1]["at_tick"] + 1)
    curve += [{"at_tick": start, "value": 127}, {"at_tick": end, "value": 0}]
    return result


def grace(prefix, tick, midi, target_midi, velocity, ticks=40):
    """倚音：打音、叠音都写成一个很短的真实音。"""
    return note(f"{prefix}_g", tick, ticks, midi, velocity, {"technique": "grace"}), target_midi
