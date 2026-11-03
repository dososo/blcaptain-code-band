"""真实离线引擎回归；音色未安装时明确跳过，不伪造通过。"""

import copy
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "skills/code-band-bgm" / "scripts"))
import fluid_engine as engine


def document():
    return {"sample_rate_hz": 48000, "total_frames": 96000,
            "parts": [{"part_id": "keys", "bank": 0, "program": 0,
                       "gain_db": 0, "pan": 0}],
            "events": [{"part_id": "keys", "event_id": "note_a", "onset_frame": 0,
                        "gate_frames": 24000, "midi": 60, "velocity": 90}],
            "mix": {"master_gain_db": -6, "effects": []},
            "render": {"loop": {"enabled": False}}}


class FluidEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.font = ROOT / "skills/code-band-bgm/assets/soundfonts/MuseScore_General.sf3"
        if not cls.font.is_file():
            raise unittest.SkipTest("未安装官方音色；真实引擎测试未执行")
        try:
            cls.info = engine.probe(cls.font)
        except engine.EngineError as exc:
            raise unittest.SkipTest("音频引擎不可用：" + exc.message)

    def test_real_presets_and_block_size(self):
        self.assertEqual(self.info["internal_block_frames"], 64)
        presets = {(p["bank"], p["program"]): p["name"] for p in self.info["presets"]}
        self.assertIn((0, 0), presets)
        self.assertIn((128, 0), presets)
        self.assertGreater(len(presets), 128)
        self.assertEqual(len(self.info["soundfont_sha256"]), 64)

    def test_part_diagnostics_already_include_native_master(self):
        doc = document()
        quiet, report = engine.render_events(doc, self.font)
        self.assertAlmostEqual(report['parts'][0]['rms_linear'], float(np.sqrt(np.mean(quiet ** 2))), places=13)
        self.assertAlmostEqual(report['parts'][0]['peak_linear'], float(np.max(np.abs(quiet))), places=13)
        doc['mix']['master_gain_db'] = 0
        dry, dry_report = engine.render_events(doc, self.font)
        gain = 10 ** (-6 / 20)
        np.testing.assert_allclose(quiet, dry * gain, rtol=1e-13, atol=1e-15)
        self.assertAlmostEqual(report['parts'][0]['rms_linear'], dry_report['parts'][0]['rms_linear'] * gain, places=13)

    def test_real_render_is_finite_non_silent_and_repeatable(self):
        first, evidence = engine.render_events(document(), self.font)
        second, _ = engine.render_events(document(), self.font)
        self.assertEqual(first.shape, (96000, 2))
        self.assertEqual(first.dtype, np.float64)
        self.assertTrue(np.isfinite(first).all())
        self.assertGreater(np.max(np.abs(first)), 1e-4)
        np.testing.assert_array_equal(first, second)
        self.assertTrue(evidence["prequantization_finite"])
        self.assertEqual(evidence["parts"][0]["event_count"], 1)
        self.assertGreater(evidence["parts"][0]["rms_linear"], 1e-5)

    def test_loop_and_reverb_are_real_not_silence_padding(self):
        score = document()
        score["render"]["loop"]["enabled"] = True
        score["events"][0]["onset_frame"] = 72000
        score["events"][0]["gate_frames"] = 16000
        score["mix"]["effects"] = [{"kind": "reverb", "target_part_ids": [],
                                    "parameters": {"room_size": 0.35, "damp": 0.45,
                                                   "width": 0.8, "level": 0.18}}]
        first, evidence = engine.render_events(score, self.font)
        second, _ = engine.render_events(score, self.font)
        self.assertEqual(first.shape, (96000, 2))
        np.testing.assert_array_equal(first, second)
        # 音符只在周期后部起音，开头有上一周期的真实释放与混响。
        self.assertGreater(float(np.max(np.abs(first[:4800]))), 1e-5)
        self.assertEqual(evidence["loop"]["continuous_cycles"], 3)
        self.assertEqual(evidence["loop"]["retained_cycle"], 3)
        self.assertFalse(evidence["loop"]["crossfade_applied"])

    def test_mixed_parts_and_pan_preserve_real_contributions(self):
        score = document()
        score["parts"][0]["pan"] = -1
        score["parts"].append({"part_id": "bass", "bank": 0, "program": 33,
                               "gain_db": -6, "pan": 1})
        score["events"].append({"part_id": "bass", "event_id": "note_b",
                                 "onset_frame": 0, "gate_frames": 24000,
                                 "midi": 36, "velocity": 90})
        rendered, evidence = engine.render_events(score, self.font)
        self.assertTrue(np.any(rendered[:, 0] != rendered[:, 1]))
        self.assertEqual(len(evidence["parts"]), 2)
        self.assertTrue(all(p["rms_linear"] > 0 for p in evidence["parts"]))

    def test_unsupported_effect_and_bad_event_fail_before_render(self):
        mutations = [
            (lambda d: d["mix"].update(effects=[{"kind": "eq"}]), "unsupported_effect"),
            (lambda d: d["events"][0].update(onset_frame=1), "event_grid"),
            (lambda d: d["events"][0].update(cents=20), "unsupported_articulation"),
            (lambda d: d["events"][0].update(pan=0.5), "unsupported_note_mix"),
            (lambda d: d["events"][0].update(velocity=0), "event_bounds"),
            (lambda d: d["events"][0].update(part_id="missing"), "invalid_event"),
        ]
        for mutation, code in mutations:
            with self.subTest(code=code):
                score = document()
                mutation(score)
                with self.assertRaises(engine.EngineError) as raised:
                    engine.render_events(score, self.font)
                self.assertEqual(raised.exception.code, code)

    def test_same_pitch_overlap_rejected(self):
        score = document()
        other = copy.deepcopy(score["events"][0])
        other.update(event_id="note_b", onset_frame=12032)
        score["events"].append(other)
        with self.assertRaises(engine.EngineError) as raised:
            engine.render_events(score, self.font)
        self.assertEqual(raised.exception.code, "same_pitch_overlap")

    def test_partial_reverb_changes_selected_part_and_keeps_other_part_dry(self):
        score = document()
        score["parts"][0]["pan"] = -1
        score["parts"].append({"part_id": "bass", "bank": 0, "program": 33,
                               "gain_db": -6, "pan": 1})
        score["events"].append({"part_id": "bass", "event_id": "bass_note",
                                 "onset_frame": 0, "gate_frames": 24000,
                                 "midi": 36, "velocity": 90})
        dry, _ = engine.render_events(score, self.font)
        score["mix"]["effects"] = [{"kind": "reverb", "target_part_ids": ["keys"],
                                    "parameters": {"level": 0.3, "send": 80}}]
        wet, diagnostics = engine.render_events(score, self.font)
        # 左右隔离使每声部的真实音频贡献可直接比较。
        np.testing.assert_array_equal(dry[:, 1], wet[:, 1])
        self.assertGreater(float(np.max(np.abs(dry[:, 0] - wet[:, 0]))), 1e-5)
        evidence = {p["part_id"]: p for p in diagnostics["parts"]}
        self.assertTrue(evidence["keys"]["includes_shared_settings_reverb"])
        self.assertFalse(evidence["bass"]["includes_shared_settings_reverb"])

    def test_unknown_reverb_target_is_rejected(self):
        score = document()
        score["mix"]["effects"] = [{"kind": "reverb", "target_part_ids": ["missing"]}]
        with self.assertRaises(engine.EngineError) as raised:
            engine.render_events(score, self.font)
        self.assertEqual(raised.exception.code, "invalid_effect_target")


if __name__ == "__main__":
    unittest.main()
