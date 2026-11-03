"""以小型资源表验证继承、引用和SF3未知帧边界，不模拟声音。"""

from pathlib import Path
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "skills/code-band-bgm" / "scripts"))
from soundfont_inventory import inventory


def _chunk(tag, data):
    return tag + struct.pack("<I", len(data)) + data + (b"\0" if len(data) % 2 else b"")


def _name(value):
    return value.encode("ascii").ljust(20, b"\0")


def _fixture(compressed=False, sample_index=0, root_override=60, terminal_gen_index=8):
    pgen = [(43, 100 << 8), (44, 20 | (127 << 8)), (43, 60 | (90 << 8)), (51, 2), (41, 0), (0, 0)]
    igen = [(43, 40 | (80 << 8)), (44, 100 << 8), (58, root_override), (51, 65524), (54, 1), (43, 50 | (70 << 8)), (52, 4), (53, sample_index), (0, 0)]
    tables = [
        (b"phdr", "<20sHHHIII", [(_name("fixture_preset"), 0, 0, 0, 0, 0, 0), (_name("EOP"), 0, 0, 2, 0, 0, 0)]),
        (b"pbag", "<HH", [(0, 0), (2, 0), (5, 0)]),
        (b"pmod", "<HHhHH", [(0, 0, 0, 0, 0)]),
        (b"pgen", "<HH", pgen),
        (b"inst", "<20sH", [(_name("fixture_instrument"), 0), (_name("EOI"), 2)]),
        (b"ibag", "<HH", [(0, 0), (5, 0), (terminal_gen_index, 0)]),
        (b"imod", "<HHhHH", [(0, 0, 0, 0, 0)]),
        (b"igen", "<HH", igen),
        (b"shdr", "<20sIIIIIBbHH", [
            (_name("fixture_sample"), 0, 64, 8, 56, 44100, 48, -3, 0, 17 if compressed else 1),
            (_name("EOS"), 0, 0, 0, 0, 0, 0, 0, 0, 0),
        ]),
    ]
    pdta = b"pdta" + b"".join(_chunk(tag, b"".join(struct.pack(fmt, *row) for row in rows)) for tag, fmt, rows in tables)
    body = (b"sfbk" + _chunk(b"LIST", b"INFO" + _chunk(b"ifil", struct.pack("<HH", 3 if compressed else 2, 1)))
            + _chunk(b"LIST", b"sdta" + _chunk(b"smpl", bytes(128))) + _chunk(b"LIST", pdta))
    return b"RIFF" + struct.pack("<I", len(body)) + body


class SoundFontInventoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 保留自有临时夹具便于复现，不对文件或目录执行批量删除。
        cls.scratch = Path(tempfile.mkdtemp(prefix="code-band-sf-test-")).resolve()
        cls.valid = cls.scratch / "valid.sf2"
        cls.valid.write_bytes(_fixture())
        cls.result = inventory(cls.valid)
        cls.mapping = cls.result["presets"][0]["mappings"][0]

    def test_local_ranges_replace_global_then_intersect(self):
        self.assertEqual(self.mapping["key_range"], [60, 70])

    def test_missing_velocity_inherits_global(self):
        self.assertEqual(self.mapping["velocity_range"], [20, 100])

    def test_root_override_keeps_header_separate(self):
        self.assertEqual(self.mapping["root_key"], 60)
        self.assertEqual(self.result["samples"][0]["original_pitch"], 48)

    def test_minus_one_root_falls_back_to_header(self):
        path = self.scratch / "fallback.sf2"
        path.write_bytes(_fixture(root_override=65535))
        self.assertEqual(inventory(path)["presets"][0]["mappings"][0]["root_key"], 48)

    def test_signed_tuning_combines_preset_and_instrument(self):
        self.assertEqual(self.mapping["coarse_tune_semitones"], -10)
        self.assertEqual(self.mapping["fine_tune_cents"], 4)

    def test_sf2_frame_count_and_raw_loop(self):
        sample = self.result["samples"][0]
        self.assertEqual(sample["decoded_frames"], 64)
        self.assertEqual(sample["raw_shdr"]["end_loop"], 56)

    def test_zone_identity_is_traceable(self):
        self.assertEqual(self.mapping["preset_global_zone_id"], "preset_bag_0")
        self.assertEqual(self.mapping["instrument_zone_id"], "inst_bag_1")

    def test_sf3_byte_offsets_never_become_decoded_frames(self):
        path = self.scratch / "compressed.sf3"
        path.write_bytes(_fixture(compressed=True))
        sample = inventory(path)["samples"][0]
        self.assertIsNone(sample["decoded_frames"])
        self.assertEqual(sample["start_end_unit"], "compressed_byte_offset")

    def test_missing_sample_reference_fails(self):
        path = self.scratch / "missing_sample.sf2"
        path.write_bytes(_fixture(sample_index=9))
        with self.assertRaises(ValueError):
            inventory(path)

    def test_truncated_riff_fails(self):
        path = self.scratch / "truncated.sf2"
        path.write_bytes(_fixture()[:-1])
        with self.assertRaises(ValueError):
            inventory(path)

    def test_terminal_generator_reference_fails(self):
        path = self.scratch / "terminal.sf2"
        path.write_bytes(_fixture(terminal_gen_index=9))
        with self.assertRaises(ValueError):
            inventory(path)

    def test_actual_official_sf3_references_and_unknown_frames(self):
        path = Path(__file__).resolve().parents[1] / "skills/code-band-bgm/assets/soundfonts/MuseScore_General.sf3"
        if not path.is_file():
            self.skipTest("本机未安装官方音色文件，实际库解析未执行。")
        actual = inventory(path)
        self.assertGreater(actual["counts"]["presets"], 0)
        self.assertTrue(all(sample["decoded_frames"] is None for sample in actual["samples"]))
        ids = {sample["sample_id"] for sample in actual["samples"]}
        self.assertTrue(all(mapping["sample_id"] in ids for preset in actual["presets"] for mapping in preset["mappings"]))


if __name__ == "__main__":
    unittest.main()
