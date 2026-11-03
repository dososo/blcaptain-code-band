#!/usr/bin/env python3
"""只读解析SoundFont资源表；不解码或演奏采样。"""

import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys


FORMATS = {
    "phdr": "<20sHHHIII", "pbag": "<HH", "pmod": "<HHhHH",
    "pgen": "<HH", "inst": "<20sH", "ibag": "<HH",
    "imod": "<HHhHH", "igen": "<HH", "shdr": "<20sIIIIIBbHH",
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _read(file, count):
    data = file.read(count)
    _require(len(data) == count, "文件截断，无法读取完整资源表。")
    return data


def _chunks(file, start, end):
    offset = start
    while offset < end:
        _require(offset + 8 <= end, "RIFF块头不完整。")
        file.seek(offset)
        tag, size = struct.unpack("<4sI", _read(file, 8))
        data_start = offset + 8
        _require(data_start + size <= end, "RIFF子块超出父块边界。")
        yield tag, data_start, size
        offset = data_start + size + (size & 1)
    _require(offset == end, "RIFF偶数字节对齐错误。")


def _name(raw):
    return raw.split(b"\0", 1)[0].decode("cp1252", errors="replace")


def _signed(value):
    return value if value < 32768 else value - 65536


def _range(generators, operator):
    value = generators.get(operator, 127 << 8)
    pair = [value & 255, value >> 8]
    _require(0 <= pair[0] <= pair[1] <= 127, "音区或力度区间无效。")
    return pair


def _zones(headers, bags, generators, link_operator, prefix):
    result = []
    for index, header in enumerate(headers[:-1]):
        first, limit = header[-1], headers[index + 1][-1]
        _require(0 <= first <= limit < len(bags), "区域索引越界。")
        global_generators, global_id, locals_ = {}, None, []
        for bag_index in range(first, limit):
            start, stop = bags[bag_index][0], bags[bag_index + 1][0]
            _require(0 <= start <= stop <= len(generators), "生成器索引越界。")
            values = {}
            for gen_index in range(start, stop):
                op, amount = generators[gen_index]
                _require(op not in values, "同一区域含重复生成器，不推测引擎修复行为。")
                values[op] = amount
                if op == link_operator:
                    _require(gen_index == stop - 1, "资源引用不是区域末尾生成器。")
            zone_id = f"{prefix}_bag_{bag_index}"
            if link_operator not in values:
                _require(bag_index == first, "全局区域必须是首个区域。")
                global_generators, global_id = values, zone_id
            else:
                merged = {**global_generators, **values}
                locals_.append({
                    "zone_id": zone_id, "global_zone_id": global_id,
                    "linked_index": values[link_operator],
                    "key_range": _range(merged, 43),
                    "velocity_range": _range(merged, 44),
                    "generators": merged,
                })
        result.append({"global_zone_id": global_id, "zones": locals_})
    return result


def inventory(path):
    """返回预设、乐器区域与采样资源表；SF3解码帧数保持未知。"""
    path = Path(path)
    size = path.stat().st_size
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while block := file.read(1024 * 1024):
            digest.update(block)
        file.seek(0)
        tag, riff_size, form = struct.unpack("<4sI4s", _read(file, 12))
        _require(tag == b"RIFF" and form == b"sfbk", "文件不是SoundFont RIFF sfbk。")
        _require(riff_size + 8 == size, "RIFF声明长度与真实文件不一致。")
        tables, info, sample_data_bytes = {}, {}, None
        for tag, offset, length in _chunks(file, 12, size):
            if tag != b"LIST":
                continue
            _require(length >= 4, "LIST块缺少类型。")
            file.seek(offset)
            kind = _read(file, 4)
            for child, pos, count in _chunks(file, offset + 4, offset + length):
                if kind == b"pdta" and child.decode("ascii", errors="replace") in FORMATS:
                    key = child.decode("ascii")
                    _require(key not in tables, "重复的pdta表。")
                    _require(count <= 64 * 1024 * 1024, "资源表超出本工具的64MiB读取上限。")
                    file.seek(pos)
                    raw = _read(file, count)
                    unit = struct.calcsize(FORMATS[key])
                    _require(count >= unit and count % unit == 0, "pdta表尺寸不符合记录结构。")
                    tables[key] = list(struct.iter_unpack(FORMATS[key], raw))
                elif kind == b"INFO" and child in (b"ifil", b"INAM", b"IENG", b"ICOP"):
                    _require(count <= 65536, "INFO字段过长。")
                    file.seek(pos)
                    raw = _read(file, count)
                    if child == b"ifil":
                        _require(count == 4, "ifil版本记录尺寸无效。")
                        info["version"] = list(struct.unpack("<HH", raw))
                    else:
                        info[child.decode("ascii")] = _name(raw)
                elif kind == b"sdta" and child == b"smpl":
                    _require(sample_data_bytes is None, "重复的采样数据块。")
                    sample_data_bytes = count
    _require(all(key in tables for key in FORMATS), "缺少必须的pdta资源表。")
    _require(sample_data_bytes is not None, "缺少smpl采样数据块。")
    for bag_key, gen_key, mod_key in (("pbag", "pgen", "pmod"), ("ibag", "igen", "imod")):
        _require(tables[bag_key][-1] == (len(tables[gen_key]) - 1, len(tables[mod_key]) - 1), "区域终止索引未指向生成器/调制器终止记录。")
        _require(all(a[0] <= b[0] and a[1] <= b[1] for a, b in zip(tables[bag_key], tables[bag_key][1:])), "区域生成器/调制器索引不是递增顺序。")
    # 官方sftools生成的SF3把终止记录名称写成全零；仍须核对索引边界。
    terminal_names = {key: _name(tables[key][-1][0]) for key in ("phdr", "inst", "shdr")}
    _require(terminal_names["phdr"] in ("", "EOP") and tables["phdr"][-1][3] == len(tables["pbag"]) - 1, "预设结束记录索引无效。")
    _require(terminal_names["inst"] in ("", "EOI") and tables["inst"][-1][1] == len(tables["ibag"]) - 1, "乐器结束记录索引无效。")
    _require(terminal_names["shdr"] in ("", "EOS") and tables["shdr"][-1][-1] == 0, "采样结束记录无效。")
    samples = []
    for index, row in enumerate(tables["shdr"][:-1]):
        name, start, end, loop_start, loop_end, rate, pitch, correction, link, sample_type = row
        compressed, rom = bool(sample_type & 0x10), bool(sample_type & 0x8000)
        _require(start < end, "采样起止字段无效。")
        if not rom:
            _require(end <= (sample_data_bytes if compressed else sample_data_bytes // 2), "采样引用超出smpl数据块。")
        base_type = sample_type & 0xF
        if base_type in (2, 4, 8):
            _require(link < len(tables["shdr"]) - 1, "立体声关联采样索引越界。")
        samples.append({
            "sample_id": f"sample_{index}", "sample_index": index,
            "source_sample_name": _name(name), "sample_rate_hz": rate,
            "original_pitch": pitch, "pitch_correction_cents": correction,
            "sample_type": sample_type, "compressed_ogg": compressed,
            "rom": rom, "linked_sample_index": link if base_type in (2, 4, 8) else None,
            "raw_shdr": {"start": start, "end": end, "start_loop": loop_start, "end_loop": loop_end},
            "start_end_unit": "compressed_byte_offset" if compressed else "sample_frame_index",
            "decoded_frames": None if compressed or rom else end - start,
            "decoded_frame_status": "未知，未解码压缩音频" if compressed else ("外部ROM，未取得音频" if rom else "来自SF2采样头，未含区域偏移"),
        })
    inst_headers = tables["inst"]
    instruments = _zones(inst_headers, tables["ibag"], tables["igen"], 53, "inst")
    for index, instrument in enumerate(instruments):
        instrument.update({"instrument_index": index, "name": _name(inst_headers[index][0])})
        for zone in instrument["zones"]:
            _require(zone["linked_index"] < len(samples), "乐器引用不存在的采样。")
            sample = samples[zone["linked_index"]]
            root = _signed(zone["generators"].get(58, 65535))
            if root == -1:
                root = sample["original_pitch"]
            zone.update({"sample_id": sample["sample_id"], "root_key": root if 0 <= root <= 127 else None})
    preset_headers = tables["phdr"]
    # phdr中的区域索引位于第四项，后面三项不是索引。
    preset_zones = _zones([row[:4] for row in preset_headers], tables["pbag"], tables["pgen"], 41, "preset")
    presets = []
    for index, preset in enumerate(preset_zones):
        header = preset_headers[index]
        mappings = []
        for pzone in preset["zones"]:
            _require(pzone["linked_index"] < len(instruments), "预设引用不存在的乐器。")
            instrument = instruments[pzone["linked_index"]]
            for izone in instrument["zones"]:
                kr = [max(pzone["key_range"][0], izone["key_range"][0]), min(pzone["key_range"][1], izone["key_range"][1])]
                vr = [max(pzone["velocity_range"][0], izone["velocity_range"][0]), min(pzone["velocity_range"][1], izone["velocity_range"][1])]
                if kr[0] > kr[1] or vr[0] > vr[1]:
                    continue
                sample = samples[izone["linked_index"]]
                mappings.append({
                    "preset_zone_id": pzone["zone_id"], "preset_global_zone_id": pzone["global_zone_id"],
                    "instrument_index": instrument["instrument_index"], "instrument_name": instrument["name"],
                    "instrument_zone_id": izone["zone_id"], "instrument_global_zone_id": izone["global_zone_id"],
                    "sample_id": sample["sample_id"], "source_sample_name": sample["source_sample_name"],
                    "key_range": kr, "velocity_range": vr, "root_key": izone["root_key"],
                    "coarse_tune_semitones": _signed(izone["generators"].get(51, 0)) + _signed(pzone["generators"].get(51, 0)),
                    "fine_tune_cents": _signed(izone["generators"].get(52, 0)) + _signed(pzone["generators"].get(52, 0)),
                    "sample_modes": izone["generators"].get(54, 0), "available_in_file": not sample["rom"],
                })
        presets.append({
            "preset_index": index, "name": _name(header[0]), "program": header[1], "bank": header[2],
            "global_zone_id": preset["global_zone_id"], "mappings": mappings,
            "key_range": [min((m["key_range"][0] for m in mappings), default=0), max((m["key_range"][1] for m in mappings), default=0)] if mappings else None,
        })
    return {
        "schema_version": "1.0.0", "status": "ok",
        "source": {"file_name": path.name, "sha256": digest.hexdigest(), "bytes": size},
        "info": info, "sample_data_bytes": sample_data_bytes, "terminal_record_names": terminal_names,
        "counts": {"presets": len(presets), "instruments": len(instruments), "samples": len(samples), "mappings": sum(len(p["mappings"]) for p in presets)},
        "samples": samples, "instruments": instruments, "presets": presets,
        "limitations": [
            "仅解析静态资源表，未运行DSP或试听；不能证明音色的音乐可用性。",
            "区域全局值由局部同类生成器覆盖，预设与乐器音区/力度取交集；并存区域可能分层同时发声，不能误作随机轮换候选。",
            "保留原始生成器数值；未计算调制器、控制器和地址偏移后的实际发声范围/音频包络。",
            "SF3起止是压缩字节偏移，解码帧数未知；原始loop字段单独保留，未作解码帧边界保证。",
        ],
        "sources": [
            "https://musescore.org/sites/musescore.org/files/2023-01/sfspec24.pdf",
            "https://github.com/FluidSynth/fluidsynth/blob/master/src/sfloader/fluid_defsfont.c",
            "https://github.com/FluidSynth/fluidsynth/blob/master/src/sfloader/fluid_sffile.c",
            "https://github.com/FluidSynth/fluidsynth/blob/master/include/fluidsynth/sfont.h",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description="只读清点SF2/SF3预设、乐器区域和采样映射。")
    parser.add_argument("--input", required=True, help="SoundFont文件")
    parser.add_argument("--out", required=True, help="新的JSON清点文件；已有文件不覆盖")
    args = parser.parse_args()
    try:
        result = inventory(args.input)
        output = Path(args.out)
        with output.open("x", encoding="utf-8") as file:
            json.dump(result, file, ensure_ascii=False, indent=2)
            file.write("\n")
        print(json.dumps({"status": "ok", "output": str(output), "counts": result["counts"], "sha256": result["source"]["sha256"]}, ensure_ascii=False))
        return 0
    except (OSError, ValueError, struct.error) as exc:
        print(json.dumps({"status": "error", "message": str(exc)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    sys.exit(main())
