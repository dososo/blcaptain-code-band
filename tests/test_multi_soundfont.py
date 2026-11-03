"""多音色库回归；仅在隔离副本移动钢琴预设，绝不改原始音色。"""

import copy
from pathlib import Path
import struct
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "skills/code-band-bgm/scripts"))
from fluid_engine import EngineError, render_events
from score_tools import ScoreError, compile_score, digest, file_hash, load, validate_frozen
from soundfont_inventory import inventory


def relocated_piano(source, destination):
    """把副本中的钢琴从bank0移动到原库不存在的bank127，以证明选库正确。"""
    raw = bytearray(source.read_bytes())
    changed = 0
    offset = 12
    while offset < len(raw):
        tag, size = struct.unpack_from("<4sI", raw, offset)
        start = offset + 8
        if tag == b"LIST" and raw[start:start + 4] == b"pdta":
            child = start + 4
            while child < start + size:
                name, length = struct.unpack_from("<4sI", raw, child)
                if name == b"phdr":
                    for pos in range(child + 8, child + 8 + length - 38, 38):
                        program, bank = struct.unpack_from("<HH", raw, pos + 20)
                        if bank == 0 and program == 0:
                            struct.pack_into("<H", raw, pos + 22, 127)
                            changed += 1
                child += 8 + length + (length & 1)
        offset = start + size + (size & 1)
    assert changed == 1
    destination.write_bytes(raw)


class MultiSoundFontTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_path = ROOT / "skills/code-band-bgm/assets/soundfonts/MuseScore_General.sf3"
        cls.base_mapping = inventory(cls.base_path)
        cls.base_hash = cls.base_mapping["source"]["sha256"]
        assert not any(p["bank"] == 127 and p["program"] == 0 for p in cls.base_mapping["presets"])
        cls.temporary = tempfile.TemporaryDirectory()
        cls.extra_path = Path(cls.temporary.name) / "isolated_test_font.sf3"
        relocated_piano(cls.base_path, cls.extra_path)
        cls.extra_mapping = inventory(cls.extra_path)
        cls.extra_hash = cls.extra_mapping["source"]["sha256"]
        cls.score = load(ROOT / "tests/fixtures/音频闭环乐谱.json")
        cls.cap = load(ROOT / "tests/fixtures/engine-capabilities-v1.json")
        resource = copy.deepcopy(next(r for r in cls.cap["resources"] if r["instrument_id"] == "piano"))
        resource.update(instrument_id="isolated_extra_piano", label="隔离测试钢琴",
                        engine_resource_id=f"sf:127:0:{cls.extra_hash}")
        cls.cap["resources"].append(resource)
        cls.cap["resource_manifest_sha256"] = digest(cls.cap["resources"])

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def mixed_score(self):
        score = copy.deepcopy(self.score)
        extra = copy.deepcopy(score["parts"][0])
        extra["part_id"] = "extra_piano"
        extra["resource_binding"]["instrument_id"] = "isolated_extra_piano"
        score["parts"].append(extra)
        return score

    def compile(self, score, extras=None):
        return compile_score(score, self.cap, self.base_hash, self.base_mapping,
                             additional_inventories=extras)

    def test_old_single_font_document_keeps_legacy_fields(self):
        compiled = self.compile(self.score)
        self.assertNotIn("used_asset_sha256", compiled)
        self.assertEqual({p["resource_sha256"] for p in compiled["parts"]}, {self.base_hash})

    def test_compile_and_frozen_require_each_real_font_hash(self):
        score = self.mixed_score()
        with self.assertRaises(ScoreError):
            self.compile(score)
        compiled = self.compile(score, {self.extra_hash: self.extra_mapping})
        self.assertEqual(compiled["used_asset_sha256"], sorted([self.base_hash, self.extra_hash]))
        self.assertEqual(compiled["parts"][1]["bank"], 127)
        self.assertEqual(compiled["parts"][1]["resource_sha256"], self.extra_hash)
        with self.assertRaises(ScoreError):
            validate_frozen(compiled, score, self.cap, self.base_hash)
        self.assertIs(validate_frozen(compiled, score, self.cap, self.base_hash,
                                      available_asset_sha256={self.extra_hash}), compiled)
        with self.assertRaises(ScoreError):
            self.compile(score, {self.extra_hash: self.base_mapping})
        changed = copy.deepcopy(compiled)
        changed["used_asset_sha256"] = [self.base_hash]
        with self.assertRaises(ScoreError):
            validate_frozen(changed, score, self.cap, self.base_hash,
                            available_asset_sha256={self.extra_hash})

    def test_real_renderer_selects_extra_font_and_rejects_missing_or_changed_file(self):
        score = self.mixed_score()
        compiled = self.compile(score, {self.extra_hash: self.extra_mapping})
        with self.assertRaises(EngineError) as raised:
            render_events(compiled, self.base_path)
        self.assertEqual(raised.exception.code, "soundfont_missing")
        with self.assertRaises(EngineError) as raised:
            render_events(compiled, self.base_path, {self.extra_hash: self.base_path})
        self.assertEqual(raised.exception.code, "soundfont_hash_mismatch")

        audio, diagnostics = render_events(compiled, self.base_path,
                                           {self.extra_hash: self.extra_path})
        self.assertEqual(audio.shape, (compiled["total_frames"], 2))
        self.assertTrue(np.isfinite(audio).all())
        self.assertGreater(float(np.max(np.abs(audio))), 1e-4)
        evidence = {p["part_id"]: p for p in diagnostics["parts"]}
        self.assertEqual(evidence["extra_piano"]["resource_sha256"], self.extra_hash)
        self.assertGreater(evidence["extra_piano"]["rms_linear"], 1e-5)
        self.assertEqual(file_hash(self.base_path), self.base_hash)
        single = self.compile(self.score)
        with self.assertRaises(EngineError) as raised:
            render_events(single, self.extra_path)
        self.assertEqual(raised.exception.code, "soundfont_hash_mismatch")

    def test_frozen_zone_and_sample_metadata_must_match_real_files(self):
        score = self.mixed_score()
        frozen = self.compile(score, {self.extra_hash: self.extra_mapping})
        mappings = {self.base_hash: self.base_mapping, self.extra_hash: self.extra_mapping}
        validate_frozen(frozen, score, self.cap, self.base_hash,
                        available_asset_sha256={self.extra_hash}, soundfont_inventories=mappings)
        for field in ("soundfont_zone_ids", "soundfont_sample_ids"):
            changed = copy.deepcopy(frozen)
            changed["events"][0][field] = ["nonexistent"]
            with self.subTest(field=field), self.assertRaises(ScoreError) as raised:
                validate_frozen(changed, score, self.cap, self.base_hash,
                    available_asset_sha256={self.extra_hash}, soundfont_inventories=mappings)
            self.assertEqual(raised.exception.details["field"], "events/samples")

    def test_current_sparse_key_declaration_also_restricts_frozen_events(self):
        score = self.mixed_score()
        frozen = self.compile(score, {self.extra_hash: self.extra_mapping})
        limited = copy.deepcopy(self.cap)
        resource = next(r for r in limited["resources"] if r["instrument_id"] == "isolated_extra_piano")
        resource["playable_midi_keys"] = [60, 67]
        limited["resource_manifest_sha256"] = digest(limited["resources"])
        with self.assertRaises(ScoreError):
            compile_score(score, limited, self.base_hash, self.base_mapping,
                          additional_inventories={self.extra_hash: self.extra_mapping})
        with self.assertRaises(ScoreError) as raised:
            validate_frozen(frozen, score, limited, self.base_hash, available_asset_sha256={self.extra_hash})
        self.assertEqual(raised.exception.details["field"], "events/pitch")


if __name__ == "__main__":
    unittest.main()
