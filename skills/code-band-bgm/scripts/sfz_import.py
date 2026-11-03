"""把开放 SFZ 采样库转成 SF2 资源。映射取自 SFZ，音高逐采样实测，防止文件名或键位差八度。

SF2 只有 16 位；连续音用烘入的交叉淡化做循环，循环处是合成结果，不是录音。
"""
from collections import Counter
import json
import math
from pathlib import Path
import re

import numpy as np
from scipy.signal import correlate
import soundfile as sf

from sf2_writer import write_sf2

OPCODE = re.compile(r"(\w+)=(.*?)(?=\s+\w+=|$)")
NOTE = {"c": 0, "d": 2, "e": 4, "f": 5, "g": 7, "a": 9, "b": 11}


def _key(value):
    text = str(value).strip().lower()
    if re.fullmatch(r"-?\d+", text):
        return int(text)
    match = re.fullmatch(r"([a-g])([#b]?)(-?\d)", text)
    if not match:
        raise ValueError(f"无法解析键：{value}")
    return NOTE[match[1]] + {"": 0, "#": 1, "b": -1}[match[2]] + (int(match[3]) + 1) * 12


def _lines(path, seen=()):
    for raw in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.split("//")[0].strip()
        if line.startswith("#include"):
            target = (Path(path).parent / line.split('"')[1].replace("\\", "/")).resolve()
            if target not in seen:
                yield from _lines(target, (*seen, target))
        elif line:
            yield line


def parse_sfz(path, base=None):
    """按 global/master/group/region 继承合并；只解析导入需要的开码。base 是样本路径的基准目录。"""
    path = Path(path)
    scopes = {"global": {}, "master": {}, "group": {}}
    regions, current, default_path, active = [], None, "", None
    for line in _lines(path):
        for token in re.split(r"(?=<\w+>)", line):
            header = re.match(r"\s*<(\w+)>", token)
            if header:
                if current is not None:
                    regions.append(current)
                current, active = None, header[1]
                token = token[header.end():]
                if active == "region":
                    current = {**scopes["global"], **scopes["master"], **scopes["group"]}
                elif active == "global":
                    scopes = {"global": {}, "master": {}, "group": {}}
                elif active == "master":
                    scopes["master"], scopes["group"] = {}, {}
                elif active == "group":
                    scopes["group"] = {}
            for key, value in OPCODE.findall(token.strip()):
                if current is not None:
                    current[key] = value.strip()
                elif key == "default_path":
                    default_path = value.strip()
                elif active in scopes:
                    scopes[active][key] = value.strip()
    if current is not None:
        regions.append(current)
    result = []
    for region in regions:
        if "sample" not in region or region.get("trigger", "attack") not in ("attack", "first"):
            continue
        low = _key(region.get("lokey", region.get("key", 0)))
        high = _key(region.get("hikey", region.get("key", 127)))
        centre = _key(region.get("pitch_keycenter", region.get("key", low)))
        sample = (Path(base or path.parent) / (default_path + region["sample"]).replace("\\", "/")).resolve()
        result.append({"sample": sample, "keys": (low, high), "root": centre, "velocities": (int(region.get("lovel", 1)) or 1, int(region.get("hivel", 127))),
                       "take": int(region.get("seq_position", 1)), "takes": int(region.get("seq_length", 1)),
                       "tune": int(float(region.get("tune", 0))), "release": float(region.get("ampeg_release", .3)), "attack": float(region.get("ampeg_attack", 0))})
    return result


def yin(mono, rate, root):
    """YIN：在候选八度上取最高的可信基频；返回相对 root 的音分，失败返回 None。

    只测最响处一窗：击打乐器的起音段可能测错（VSCO 木琴 C4、G5），这类采样由调用方用 cents_override 覆盖。
    """
    level = np.abs(mono)
    if level.max() <= 0:
        return None
    window = 1 << 13
    centre = int(np.argmax(np.convolve(level, np.ones(4096) / 4096, "same")))
    start = max(min(centre, len(mono) - window * 2), 0)
    frame = mono[start:start + window * 2]
    if len(frame) < window * 2:
        return None
    spectrum = np.fft.rfft(frame, window * 4)
    correlation = np.fft.irfft(np.abs(spectrum) ** 2)[:window]
    difference = 2 * correlation[0] - 2 * correlation
    difference[0] = 1
    cumulative = np.cumsum(difference[1:])
    normalised = np.ones(window)
    normalised[1:] = difference[1:] * np.arange(1, window) / np.maximum(cumulative, 1e-12)
    for shift in (24, 12, 0, -12, -24):
        hz = 440 * 2 ** ((root + shift - 69) / 12)
        lag = rate / hz
        low, high = int(lag * .95), int(lag * 1.05) + 2
        if high >= window or low < 2:
            continue
        index = low + int(np.argmin(normalised[low:high]))
        if normalised[index] < .25:
            left, centre_value, right = normalised[index - 1:index + 2]
            bend = .5 * (left - right) / (left - 2 * centre_value + right + 1e-12)
            return 1200 * math.log2(rate / (index + bend) / (440 * 2 ** ((root - 69) / 12)))
    return None


def build_loop(data, rate, fade_seconds=.08):
    """在后段找波形最相似的两点做循环，把交叉淡化烘入样本。返回(数据, 循环起止, 相似度)或None。"""
    mono = data.mean(axis=1)
    frames = len(mono)
    if frames < rate * 1.2:
        return None
    fade = int(rate * fade_seconds)
    size = min(4096, fade)
    earliest, latest = max(int(frames * .3), fade + size), int(frames * .62)
    best = (None, None, -2.0)
    for end in np.linspace(frames * .72, frames * .94, 12).astype(int):
        target = mono[end - size:end] - mono[end - size:end].mean()
        region = mono[earliest - size:latest]
        if len(region) <= size:
            continue
        overlap = correlate(region, target, mode="valid", method="fft")
        energy = np.convolve(region ** 2, np.ones(size), "valid")
        score = overlap / np.sqrt(np.maximum(energy * float(np.sum(target ** 2)), 1e-18))
        index = int(np.argmax(score))
        if score[index] > best[2]:
            best = (earliest + index, int(end), float(score[index]))
    start, end, similarity = best
    if start is None or similarity < .6:
        return None
    blended = data.copy()
    ramp = np.sin(np.linspace(0, np.pi / 2, fade))[:, None]
    blended[end - fade:end] = data[end - fade:end] * np.flip(ramp, axis=0) + data[start - fade:start] * ramp
    return blended[:end + 64], (start, end), round(similarity, 3)


def _timecents(seconds):
    return max(-12000, round(1200 * math.log2(max(seconds, .001))))


def _measure(regions, pitched):
    """先测全部采样，再定全乐器统一的八度：个别采样的八度判断不可信，音分偏差才逐个用。"""
    measured = {}
    for region in regions:
        path = region["sample"]
        if path in measured or not pitched:
            continue
        data, rate = sf.read(path, always_2d=True, dtype="float64")
        measured[path] = yin(data.mean(axis=1)[:int(8 * rate)], rate, region["root"])
    shifts = [round(v / 1200) * 12 for v in measured.values() if v is not None]
    octave = Counter(shifts).most_common(1)[0][0] if shifts else 0
    cents, conflicts = {}, []
    for path, value in measured.items():
        if value is None:
            cents[path] = None
            continue
        offset = value - octave * 100
        wrapped = (offset + 600) % 1200 - 600
        if abs(wrapped) > 60:
            conflicts.append(path.name)
            cents[path] = None
        else:
            cents[path] = wrapped
    return octave, cents, conflicts


def convert(items, output, bank_name, report, max_seconds=8.0):
    """items:[{name,sfz,base?,program,loop,pitched,cents_override?}]；同一 program 的轮换依次占用后续 program。

    cents_override：{采样文件名: 音分}，用于自动测音已被独立测量证伪的采样（须在调用方写明证据）。
    """
    samples, presets, rows, cache, summary = [], [], [], {}, []
    for item in items:
        regions = parse_sfz(item["sfz"], item.get("base"))
        octave, cents, conflicts = _measure(regions, item.get("pitched", True))
        overrides = item.get("cents_override", {})
        for path in list(cents):
            if path.name in overrides:
                cents[path] = overrides[path.name]
        missing = set(overrides) - {path.name for path in cents}
        if missing:
            raise ValueError(f"cents_override 指向不存在的采样：{sorted(missing)}")
        takes = max((r["take"] for r in regions), default=1)
        programs = {}
        for region in regions:
            path = region["sample"]
            if path not in cache:
                data, rate = sf.read(path, always_2d=True, dtype="float64")
                data = data[:int(max_seconds * rate)]
                loop = build_loop(data, rate) if item.get("loop") else None
                loop_point = quality = None
                if loop:
                    data, loop_point, quality = loop
                else:
                    tail = min(int(.05 * rate), len(data) // 4)
                    data = data.copy()
                    data[-tail:] *= np.linspace(1, 0, tail)[:, None]
                name = f"s{len(cache):04d}"
                stereo = data.shape[1] > 1 and not np.allclose(data[:, 0], data[:, 1])
                offset = cents.get(path)
                sample = {"name": name, "left": data[:, 0], "right": data[:, 1] if stereo else None, "rate": rate,
                          "root": min(127, max(0, region["root"] + octave)),
                          "correction": int(np.clip(round(-(offset or 0)) + region["tune"], -99, 99))}
                if loop_point:
                    sample["loop"] = loop_point
                samples.append(sample)
                cache[path] = (name, loop_point)
                rows.append({"sample": path.name, "name": name, "instrument": item["name"], "sfz_key": region["root"], "octave_shift": octave,
                             "measured_cents": None if offset is None else round(offset, 1),
                             "pitch_source": "override" if path.name in overrides else "yin_peak_window", "seconds": round(len(data) / rate, 2), "rate": rate,
                             "channels": 2 if stereo else 1, "loop": loop_point, "loop_similarity": quality})
            name, loop_point = cache[path]
            keys = tuple(min(127, max(0, k + octave)) for k in region["keys"])
            generators = {"attack_vol_env": _timecents(region["attack"]) if region["attack"] > .002 else -12000,
                          "release_vol_env": _timecents(region["release"]), "sample_modes": 1 if loop_point else 0}
            program = item["program"] + (region["take"] - 1 if takes > 1 else 0)
            programs.setdefault(program, []).append({"sample": name, "keys": keys, "velocities": region["velocities"], "generators": generators})
        for program, zones in sorted(programs.items()):
            presets.append({"name": f"{item['name'][:16]}{program}", "bank": 0, "program": program, "zones": zones})
        summary.append({"name": item["name"], "programs": sorted(programs), "octave_shift": octave, "pitch_conflicts": conflicts})
    write_sf2(output, bank_name, samples, presets)
    Path(report).write_text(json.dumps({"bank": bank_name, "presets": summary, "samples": rows}, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return rows
