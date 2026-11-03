"""语汇展开的结果必须能通过谱面校验，且真实落到音高和音符上。"""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "skills/code-band-bgm/scripts"))
import idiomatic as idiom
from note_expression import validate_performance
from instrument_readiness import classify


class IdiomTest(unittest.TestCase):
    def test_press_up_then_vibrato_passes_validation(self):
        item = idiom.vibrato(idiom.press_up(idiom.note("a", 0, 1920, 67, 80), 200, 100, 400), 500, 300)
        validate_performance(item["performance"], item["duration_ticks"])
        self.assertEqual(item["performance"]["pitch_curve"][-1]["cents"], 200)
        self.assertEqual(item["performance"]["technique"], "press_up+vibrato")

    def test_guqin_yin_uses_measured_profile(self):
        # 960 tick/秒：声后吟 0.87 秒才起摆，深度 52→33 音分，频率 2.63 Hz
        item = idiom.guqin_yin(idiom.note("a", 0, 2400, 50, 80), "shenghou", 960)
        validate_performance(item["performance"], item["duration_ticks"])
        vib = item["performance"]["vibrato"]
        self.assertEqual(vib["start_tick"], round(.87 * 960))
        self.assertEqual((vib["rate_hz"], vib["depth_cents"], vib["end_depth_cents"]), (2.63, 52, 33))
        self.assertEqual(item["performance"]["technique"], "guqin_shenghou")
        self.assertLess(idiom.GUQIN_MEASURED["chengsheng"]["rate_hz"], idiom.ASSUMED["yinrou_rate_hz"])
        # 吟往上摆：起摆前在本音，摆起来后中心抬高一个深度，下沿仍在本音
        self.assertEqual(item["performance"]["pitch_curve"], [{"at_tick": 0, "cents": 0}, {"at_tick": 835, "cents": 0},
                                                              {"at_tick": 979, "cents": 52}, {"at_tick": 2400, "cents": 33}])

    def test_guqin_quanfu_swings_down_and_chuo_then_yin_chains(self):
        down = idiom.guqin_yin(idiom.note("a", 0, 2400, 50, 80), "quanfu", 960)
        self.assertEqual(down["performance"]["pitch_curve"][-1]["cents"], -16)
        chained = idiom.guqin_yin(idiom.guqin_chuo(idiom.note("b", 0, 2400, 50, 80), 960, "fast"), "chengsheng", 960)
        validate_performance(chained["performance"], chained["duration_ticks"])
        curve = chained["performance"]["pitch_curve"]
        self.assertEqual(curve[:2], [{"at_tick": 0, "cents": -200}, {"at_tick": 422, "cents": 0}])
        self.assertEqual(curve[-1], {"at_tick": 2400, "cents": 44})
        self.assertEqual([p["at_tick"] for p in curve], sorted({p["at_tick"] for p in curve}))

    def test_guqin_chuo_slides_in_from_below(self):
        item = idiom.guqin_chuo(idiom.note("a", 0, 1920, 50, 80), 960, "slow")
        validate_performance(item["performance"], item["duration_ticks"])
        self.assertEqual(item["performance"]["pitch_curve"], [{"at_tick": 0, "cents": -200}, {"at_tick": round(.99 * 960), "cents": 0}])
        short = idiom.guqin_chuo(idiom.note("b", 0, 200, 50, 80), 960, "slow")
        self.assertEqual(short["performance"]["pitch_curve"][-1]["at_tick"], 200)

    def test_press_return_comes_back(self):
        item = idiom.press_up(idiom.note("a", 0, 1920, 67, 80), 100, 100, 300, hold=False)
        self.assertEqual(item["performance"]["pitch_curve"][-1]["cents"], 0)
        validate_performance(item["performance"], item["duration_ticks"])

    def test_tremolo_is_real_repeats(self):
        notes = idiom.tremolo("t", 0, 1920, 74, 50, 90, 1000)
        self.assertGreater(len(notes), 10)
        self.assertEqual(len({n["note_id"] for n in notes}), len(notes))
        self.assertTrue(all(n["pitch"]["midi"] == 74 for n in notes))
        self.assertGreater(notes[-1]["velocity"], notes[0]["velocity"])

    def test_run_follows_scale(self):
        notes = idiom.run("r", 0, [50, 53, 55], 40, 40, 80, 400)
        self.assertEqual([n["pitch"]["midi"] for n in notes], [50, 53, 55])
        self.assertEqual([n["onset_tick"] for n in notes], [0, 40, 80])

    def test_slur_and_fade_valid(self):
        first = idiom.fade_out(idiom.note("a", 0, 960, 69, 80), 60)
        second = idiom.slur_to(idiom.note("b", 900, 960, 71, 80), 69, 960, slide_ticks=80)
        validate_performance(first["performance"], 960)
        validate_performance(second["performance"], 960)
        self.assertEqual(second["performance"]["pitch_curve"][0]["cents"], -200)


class ClassifyTest(unittest.TestCase):
    BASE = {"root_keys": 14, "playable_keys": 41, "key_span": 41, "contiguous": True, "max_transposition_semitones": 3,
            "p90_transposition_semitones": 3, "stretched_keys": 0,
            "median_velocity_layers": 1, "median_takes_per_key": 1, "any_loop": False, "unpitched_hits": 0, "articulations": ["normal"]}

    def test_rejected_six_root_guzheng_is_probe(self):
        tier, gaps = classify({**self.BASE, "root_keys": 6, "playable_keys": 6, "key_span": 32, "contiguous": False, "max_transposition_semitones": 0,
                                  "p90_transposition_semitones": 0})
        self.assertEqual(tier, "probe")
        self.assertTrue(gaps)

    def test_coverage_ok(self):
        self.assertEqual(classify(self.BASE)[0], "coverage_ok")

    def test_two_root_is_probe(self):
        self.assertEqual(classify({**self.BASE, "root_keys": 2})[0], "probe")

    def test_narrow_is_limited(self):
        self.assertEqual(classify({**self.BASE, "key_span": 13, "playable_keys": 13})[0], "limited")

    def test_percussion_without_rr_limited(self):
        self.assertEqual(classify({**self.BASE, "unpitched_hits": 2, "median_velocity_layers": 2, "median_takes_per_key": 2})[0], "limited")


if __name__ == "__main__":
    unittest.main()
