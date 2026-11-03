"""真实GM标准鼓逐键发声与释音隔离，不把残音当作下一键成功。"""

import hashlib
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "skills/code-band-bgm/scripts"))
import fluid_engine as engine


class PercussionProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.font = ROOT / "skills/code-band-bgm/assets/soundfonts/MuseScore_General.sf3"

    def test_all_standard_keys_use_one_synth_and_real_dry_windows(self):
        hits = {f"note_{midi}": str(midi) for midi in range(35, 82)}
        with patch.object(engine, "_Synth", wraps=engine._Synth) as constructor:
            report = engine.probe_hits(self.font, 128, 0, hits)
        self.assertEqual(constructor.call_count, 1)
        self.assertEqual([r["midi"] for r in report], list(range(35, 82)))
        self.assertTrue(all(r["status"] == "通过" for r in report))
        self.assertTrue(all(r["frames"] == 4800 and r["finite"] for r in report))
        self.assertTrue(all(r["peak_linear"] > 0.0001 for r in report))
        self.assertTrue(all(r["cleanup_peak_linear"] <= 1e-8 for r in report))
        self.assertTrue(all(not r["reverb_enabled"] and not r["chorus_enabled"] for r in report))

    def test_long_previous_cymbal_does_not_pollute_next_bongo(self):
        report = engine.probe_hits(self.font, 128, 0, {"long_ride": "59", "bongo": "60"})
        target = next(r for r in report if r["midi"] == 60)
        with engine._Synth(self.font, 48000, None) as synth:
            channel = 9
            engine._ok(synth.lib.fluid_synth_program_select(synth.synth, channel,
                                                           synth.sfont_id, 128, 0), "测试预设")
            engine._ok(synth.lib.fluid_synth_cc(synth.synth, channel, 91, 0), "测试混响")
            engine._ok(synth.lib.fluid_synth_cc(synth.synth, channel, 93, 0), "测试合唱")
            engine._ok(synth.lib.fluid_synth_cc(synth.synth, channel, 120, 0), "测试全停")
            synth.samples(64)
            # 保持相同时间线与起音次数，只把先前长镲从力度90改为1。
            # 不要求新建合成器的首音等同于已有发声历史的下一音。
            engine._ok(synth.lib.fluid_synth_noteon(synth.synth, channel, 59, 1), "测试弱前音")
            synth.samples(4800)
            engine._ok(synth.lib.fluid_synth_cc(synth.synth, channel, 120, 0), "测试前音全停")
            cleared = synth.samples(64)
            np.testing.assert_array_equal(cleared, np.zeros_like(cleared))
            engine._ok(synth.lib.fluid_synth_noteon(synth.synth, channel, 60, 90), "测试独立键")
            independent = synth.samples(4800)
        checksum = hashlib.sha256(independent.astype("<f4", copy=False).tobytes()).hexdigest()
        self.assertEqual(target["window_float32_sha256"], checksum)
        self.assertEqual(target["peak_linear"], float(np.max(np.abs(independent))))

    def test_aliases_are_deduplicated_and_unmapped_key_cannot_pass(self):
        report = engine.probe_hits(self.font, 128, 0,
                                   {"kick": "36", "same_kick": "36", "outside": "0"})
        self.assertEqual(len(report), 2)
        missing = next(r for r in report if r["midi"] == 0)
        self.assertEqual(missing["status"], "失败")
        self.assertEqual(missing["code"], "unmapped_hit")
        self.assertEqual(missing["frames"], 0)
        self.assertIsNone(missing["peak_linear"])
        valid = next(r for r in report if r["midi"] == 36)
        self.assertEqual(valid["hit_keys"], ["kick", "same_kick"])
        self.assertEqual(valid["status"], "通过")

    def test_invalid_note_or_missing_preset_is_explicit_failure(self):
        with self.assertRaises(engine.EngineError):
            engine.probe_hits(self.font, 128, 0, {"bad": "128"})
        with self.assertRaises(engine.EngineError):
            engine.probe_hits(self.font, 127, 0, {"bad": "36"})


if __name__ == "__main__":
    unittest.main()
