"""逐区域渲染根音，检查根音附近有没有比根音还响的别的音；抓「音高测量只在谐波上对上」或混入余振的错采样。

只适用于有音高的乐器。输出 JSON：每个区域的主峰、最近谐波和偏差，偏差超阈值记为 suspect。
用法：python zone_pitch_check.py <sf2> <输出json> [--programs 0,1] [--margin 0]
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from fluid_engine import render_events
from soundfont_inventory import inventory

RATE = 48000


def fundamental_error(audio, start, key):
    """在基频和二倍频两处，各比较根音上下 650 音分内（避开三次谐波 +702）的「根音峰」和「别的音峰」。

    高音弦泛音常比基频强，低音乐器基频又常常弱到被噪声盖过，单看一处都会误报；
    录成别的音、混入前一个音的余振时，别的音在两处都会比根音响。
    """
    # 低音区 ±60 音分不足一个频点，换长窗口；1<<18 补零保证插值精度
    size = 8192 if 440 * 2 ** ((key - 69) / 12) >= 150 else 32768
    segment = audio[start:start + size]
    if len(segment) < size:
        return None
    spectrum = np.abs(np.fft.rfft(segment * np.hanning(size), 1 << 18))
    bins = np.fft.rfftfreq(1 << 18, 1 / RATE)
    overall = spectrum[(bins > 40) & (bins < 12000)].max()
    if overall <= 0:
        return None
    result = {}
    for octave in (1, 2):
        f0 = 440 * 2 ** ((key - 69) / 12) * octave
        cents = 1200 * np.log2(np.maximum(bins, 1e-9) / f0)
        root, other = np.abs(cents) <= 60, (np.abs(cents) > 100) & (np.abs(cents) <= 650)
        index = np.flatnonzero(other)[np.argmax(spectrum[other])]
        result[f"octave{octave}"] = {"root_db": round(20 * math.log10(max(spectrum[root].max(), 1e-12) / overall), 1),
                                     "other_hz": round(float(bins[index]), 1), "other_cents": round(float(cents[index]), 1),
                                     "other_minus_root_db": round(20 * math.log10(spectrum[index] / max(spectrum[root].max(), 1e-12)), 1)}
    first, second = result["octave1"], result["octave2"]
    # 基频清晰时看基频一处；基频弱（低音乐器常见，噪声会冒充别的音）时要二倍频也确认
    result["other_minus_root_db"] = (first["other_minus_root_db"] if first["root_db"] >= -25
                                     else min(first["other_minus_root_db"], second["other_minus_root_db"]))
    return result


def check(path, programs=None, margin=0.0):
    inv = inventory(str(path))
    rows = []
    for preset in inv["presets"]:
        if programs is not None and preset["program"] not in programs:
            continue
        seen = set()
        for zone in preset["mappings"]:
            if not zone["available_in_file"] or zone["root_key"] is None:
                continue
            root = zone["root_key"]
            key = root if zone["key_range"][0] <= root <= zone["key_range"][1] else zone["key_range"][0]
            velocity = (zone["velocity_range"][0] + zone["velocity_range"][1]) // 2
            if (key, velocity) in seen:
                continue
            seen.add((key, velocity))
            doc = {"sample_rate_hz": RATE, "total_frames": 2 * RATE, "mix": {"master_gain_db": 0, "effects": []},
                   "parts": [{"part_id": "p", "bank": preset["bank"], "program": preset["program"], "gain_db": 0, "pan": 0}],
                   "events": [{"event_id": "e", "part_id": "p", "onset_frame": 0, "gate_frames": 2 * RATE - 9600,
                               "midi": key, "velocity": velocity}], "render": {"loop": {"enabled": False}}}
            audio, _ = render_events(doc, str(path))
            mono = audio.mean(axis=1)
            judged = [j for j in (fundamental_error(mono, int(t * RATE), key) for t in (.03, .15, .35)) if j]
            suspect = len(judged) >= 2 and sum(j["other_minus_root_db"] > margin for j in judged) >= 2
            rows.append({"bank": preset["bank"], "program": preset["program"], "key": key, "velocity": velocity,
                         "sample": zone["source_sample_name"], "peaks": judged, "suspect": suspect})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("soundfont", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--programs")
    parser.add_argument("--margin", type=float, default=0.0, help="别的音比根音响多少 dB 算可疑")
    args = parser.parse_args()
    programs = {int(p) for p in args.programs.split(",")} if args.programs else None
    rows = check(args.soundfont, programs, args.margin)
    suspects = [r for r in rows if r["suspect"]]
    args.output.write_text(json.dumps({"schema_version": "1.0", "soundfont": str(args.soundfont), "margin_db": args.margin,
                                       "zones_checked": len(rows), "suspects": suspects, "rows": rows},
                                      ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"zones_checked": len(rows), "suspects": [(r["program"], r["key"], r["sample"], r["peaks"]) for r in suspects]},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
