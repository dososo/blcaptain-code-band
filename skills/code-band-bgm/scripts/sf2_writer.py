"""最小SF2写入器：把已切好的真实采样封装成立体声链接的SoundFont。只写容器，不做音频处理。"""
import os
from pathlib import Path
import struct
import tempfile

import numpy as np

GUARD_FRAMES = 46
GENERATORS = {"pan": 17, "attack_vol_env": 34, "hold_vol_env": 35, "decay_vol_env": 36, "sustain_vol_env": 37,
              "release_vol_env": 38, "initial_filter_fc": 8, "initial_attenuation": 48, "sample_modes": 54}


def _name(text):
    encoded = text.encode("ascii", "replace")[:19]
    return encoded.ljust(20, b"\0")


def _chunk(tag, body):
    return tag + struct.pack("<I", len(body)) + body + (b"\0" if len(body) % 2 else b"")


def _list(tag, chunks):
    body = tag + b"".join(chunks)
    return b"LIST" + struct.pack("<I", len(body)) + body


def _pcm(channel):
    values = np.rint(np.clip(channel, -1, 1) * 32767).astype("<i2")
    return values.tobytes() + b"\0" * (2 * GUARD_FRAMES)


def _amount(value):
    return struct.pack("<h", value) if value < 0 else struct.pack("<H", value)


def write_sf2(path, bank_name, samples, presets):
    """samples:{name,left,right|None,rate,root,correction,loop?}；presets:{name,bank,program,zones:[{sample,keys,velocities,generators}]}"""
    pcm, headers, index = [], [], {}
    position = 0
    for sample in samples:
        frames = len(sample["left"])
        stereo = sample.get("right") is not None
        if frames < 64 or (stereo and len(sample["right"]) != frames):
            raise ValueError(f"采样长度无效：{sample['name']}")
        pair = len(headers)
        sides = (("L", sample["left"], pair + 1, 4), ("R", sample["right"], pair, 2)) if stereo else (("M", sample["left"], 0, 1),)
        loop_start, loop_end = sample.get("loop", (0, frames))
        for side, channel, link, kind in sides:
            pcm.append(_pcm(channel))
            headers.append(struct.pack("<20sIIIIIBbHH", _name(f"{sample['name'][:17]}_{side}"), position, position + frames,
                                       position + loop_start, position + loop_end, sample["rate"], sample["root"], sample["correction"], link, kind))
            position += frames + GUARD_FRAMES
        index[sample["name"]] = (pair, stereo)
    headers.append(struct.pack("<20sIIIIIBbHH", _name("EOS"), 0, 0, 0, 0, 0, 0, 0, 0, 0))
    instruments, instrument_bags, instrument_generators = [], [], []
    preset_rows, preset_bags, preset_generators = [], [], []
    for number, preset in enumerate(sorted(presets, key=lambda p: (p["bank"], p["program"]))):
        instruments.append(struct.pack("<20sH", _name(preset["name"]), len(instrument_bags)))
        for zone in preset["zones"]:
            pair, stereo = index[zone["sample"]]
            for side, pan in ((0, -500), (1, 500)) if stereo else ((0, 0),):
                instrument_bags.append(struct.pack("<HH", len(instrument_generators), 0))
                instrument_generators.append(struct.pack("<HBB", 43, *zone["keys"]))
                instrument_generators.append(struct.pack("<HBB", 44, *zone["velocities"]))
                instrument_generators.append(struct.pack("<H", GENERATORS["pan"]) + _amount(pan))
                for name, value in sorted(zone.get("generators", {}).items()):
                    instrument_generators.append(struct.pack("<H", GENERATORS[name]) + _amount(value))
                instrument_generators.append(struct.pack("<HH", 53, pair + side))
        preset_rows.append(struct.pack("<20sHHHIII", _name(preset["name"]), preset["program"], preset["bank"], len(preset_bags), 0, 0, 0))
        preset_bags.append(struct.pack("<HH", len(preset_generators), 0))
        preset_generators.append(struct.pack("<HH", 41, number))
    preset_rows.append(struct.pack("<20sHHHIII", _name("EOP"), 0, 0, len(preset_bags), 0, 0, 0))
    preset_bags.append(struct.pack("<HH", len(preset_generators), 0))
    preset_generators.append(struct.pack("<HH", 0, 0))
    instruments.append(struct.pack("<20sH", _name("EOI"), len(instrument_bags)))
    instrument_bags.append(struct.pack("<HH", len(instrument_generators), 0))
    instrument_generators.append(struct.pack("<HH", 0, 0))
    title = bank_name.encode("ascii", "replace") + b"\0"
    title += b"\0" * (len(title) % 2)  # FluidSynth 拒绝奇数长度的 INFO 子块
    info = _list(b"INFO", [_chunk(b"ifil", struct.pack("<HH", 2, 1)), _chunk(b"isng", b"EMU8000\0"), _chunk(b"INAM", title)])
    sdta = _list(b"sdta", [_chunk(b"smpl", b"".join(pcm))])
    pdta = _list(b"pdta", [_chunk(b"phdr", b"".join(preset_rows)), _chunk(b"pbag", b"".join(preset_bags)),
                           _chunk(b"pmod", b"\0" * 10), _chunk(b"pgen", b"".join(preset_generators)),
                           _chunk(b"inst", b"".join(instruments)), _chunk(b"ibag", b"".join(instrument_bags)),
                           _chunk(b"imod", b"\0" * 10), _chunk(b"igen", b"".join(instrument_generators)),
                           _chunk(b"shdr", b"".join(headers))])
    body = b"sfbk" + info + sdta + pdta
    target = Path(path)
    descriptor, temporary = tempfile.mkstemp(dir=target.parent, suffix=".tmp")
    with os.fdopen(descriptor, "wb") as output:
        output.write(b"RIFF" + struct.pack("<I", len(body)) + body)
    os.replace(temporary, target)
