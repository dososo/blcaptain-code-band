"""逐音演奏控制：用真实渲染测频移和隔离，不把调用成功当音高生效。"""
import copy
import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "skills/code-band-bgm/scripts"))
import fluid_engine as engine
import note_expression as ne
from score_tools import ScoreError, compile_score, load, round_robin, validate_frozen, validate_score
from sf2_writer import write_sf2
from soundfont_inventory import inventory

FONT = ROOT / "skills/code-band-bgm/assets/soundfonts/MuseScore_General.sf3"


def doc(events, program=73, frames=48000):
    return {"sample_rate_hz": 48000, "total_frames": frames,
            "parts": [{"part_id": "p", "bank": 0, "program": program, "gain_db": 0, "pan": 0}],
            "events": events, "mix": {"master_gain_db": 0, "effects": []},
            "render": {"loop": {"enabled": False}}}


def note(identifier="a", onset=0, gate=43200, midi=69, velocity=90, **extra):
    return {"event_id": identifier, "part_id": "p", "onset_frame": onset, "gate_frames": gate,
            "midi": midi, "velocity": velocity, **extra}


def peak_hz(audio, start, stop, low=300, high=1200):
    length = stop - start
    window = audio[start:stop].mean(axis=1) * np.hanning(length)
    spectrum = np.abs(np.fft.rfft(window, 1 << 19))
    bins = np.fft.rfftfreq(1 << 19, 1 / 48000)
    band = (bins > low) & (bins < high)
    return float(bins[band][np.argmax(spectrum[band])])


class ValidationTest(unittest.TestCase):
    def test_unknown_field_rejected(self):
        with self.assertRaises(ValueError):
            ne.validate_performance({"pitch_curve": [{"at_tick": 0, "cents": 0}], "wobble": 1}, 480)

    def test_curve_beyond_note_rejected(self):
        with self.assertRaises(ValueError):
            ne.validate_performance({"pitch_curve": [{"at_tick": 481, "cents": 0}]}, 480)

    def test_non_monotonic_curve_rejected(self):
        with self.assertRaises(ValueError):
            ne.validate_performance({"expression_curve": [{"at_tick": 100, "value": 1}, {"at_tick": 100, "value": 2}]}, 480)

    def test_bend_range_enforced(self):
        with self.assertRaises(ValueError):
            ne.validate_performance({"pitch_curve": [{"at_tick": 0, "cents": 1300}]}, 480)

    def test_nan_rejected(self):
        with self.assertRaises(ValueError):
            ne.validate_performance({"vibrato": {"start_tick": 0, "fade_ticks": 0, "rate_hz": float("nan"), "depth_cents": 1}}, 480)

    def test_frozen_point_off_grid_rejected_by_engine(self):
        event = note(performance={"pitch_points": [[0, 0.0], [100, 50.0]]})
        with self.assertRaises(engine.EngineError) as caught:
            engine.render_events(doc([event]), FONT)
        self.assertEqual(caught.exception.code, "invalid_performance")

    def test_frozen_unknown_field_rejected_by_engine(self):
        event = note(performance={"pitch_points": [[0, 0.0]], "x": 1})
        with self.assertRaises(engine.EngineError):
            engine.render_events(doc([event]), FONT)


class ResolveTest(unittest.TestCase):
    def resolve(self, performance, duration=960, gate=48000):
        return ne.resolve_performance(performance, 0, 0, duration, gate, lambda t: t * 50, 48000)

    def test_deterministic(self):
        performance = {"vibrato": {"start_tick": 100, "fade_ticks": 200, "rate_hz": 5.5, "depth_cents": 30},
                       "pitch_curve": [{"at_tick": 0, "cents": -100}, {"at_tick": 300, "cents": 0}]}
        self.assertEqual(self.resolve(performance), self.resolve(copy.deepcopy(performance)))

    def test_points_on_engine_grid(self):
        resolved = self.resolve({"pitch_curve": [{"at_tick": 0, "cents": -200}, {"at_tick": 960, "cents": 200}]})
        self.assertTrue(all(point[0] % 64 == 0 for point in resolved["pitch_points"]))

    def test_glide_endpoints(self):
        resolved = self.resolve({"pitch_curve": [{"at_tick": 0, "cents": -200}, {"at_tick": 480, "cents": 0}]})
        self.assertEqual(resolved["pitch_points"][0], [0, -200.0])
        self.assertEqual(resolved["pitch_points"][-1][1], 0.0)

    def test_vibrato_starts_late_and_fades_in(self):
        resolved = self.resolve({"vibrato": {"start_tick": 480, "fade_ticks": 240, "rate_hz": 6, "depth_cents": 40}})
        early = [c for f, c in resolved["pitch_points"] if f < 24000]
        self.assertEqual(early, [0.0])
        self.assertGreater(max(abs(c) for _, c in resolved["pitch_points"]), 30)

    def test_bend_value_centre_and_limits(self):
        self.assertEqual(ne.bend_value(0), 8192)
        self.assertEqual(ne.bend_value(-1200), 0)
        self.assertEqual(ne.bend_value(1200), 16383)


class RenderedPitchTest(unittest.TestCase):
    def test_plain_part_unchanged_path(self):
        """无演奏控制的声部必须走旧路径，诊断中没有performance。"""
        _, diagnostic = engine.render_events(doc([note()]), FONT)
        self.assertNotIn("performance", diagnostic["parts"][0])

    def test_static_bend_measured_in_cents(self):
        flat, _ = engine.render_events(doc([note()]), FONT)
        for cents in (100, -100, 50, 700):
            bent, _ = engine.render_events(doc([note(performance={"pitch_points": [[0, float(cents)]]})]), FONT)
            measured = 1200 * math.log2(peak_hz(bent, 12000, 36000) / peak_hz(flat, 12000, 36000))
            self.assertAlmostEqual(measured, cents, delta=6, msg=f"目标{cents}音分")

    def test_upward_glide_really_rises(self):
        points = [[offset, -200 + 200 * offset / 24000] for offset in range(0, 24000, 256)] + [[24000, 0.0]]
        points = [[f, round(c, 2)] for f, c in points]
        audio, _ = engine.render_events(doc([note(performance={"pitch_points": points})]), FONT)
        start, end = peak_hz(audio, 2000, 8000), peak_hz(audio, 30000, 42000)
        self.assertGreater(1200 * math.log2(end / start), 100)

    def test_bend_is_isolated_per_note(self):
        """同声部和弦中只弯一个音，另一音的基频不能跟着动。"""
        steady = note("low", midi=60, velocity=80)
        bent = note("high", midi=69, velocity=80, performance={"pitch_points": [[0, 100.0]]})
        audio, diagnostic = engine.render_events(doc([steady, bent]), FONT)
        solo, _ = engine.render_events(doc([note("low", midi=60, velocity=80, performance={"pitch_points": [[0, 0.0]]})]), FONT)
        self.assertAlmostEqual(peak_hz(audio, 12000, 36000, 200, 300), peak_hz(solo, 12000, 36000, 200, 300), delta=1)
        self.assertEqual(diagnostic["parts"][0]["performance"]["channels_used"], 2)

    def test_same_pitch_overlap_allowed_when_dedicated(self):
        tremolo = [note(f"t{i}", onset=i * 4800, gate=9600, performance={"pitch_points": [[0, 0.0]]}) for i in range(5)]
        audio, _ = engine.render_events(doc(tremolo), FONT)
        self.assertGreater(float(np.abs(audio).max()), .001)

    def test_plain_same_pitch_overlap_still_rejected(self):
        with self.assertRaises(engine.EngineError):
            engine.render_events(doc([note("a", gate=9600), note("b", onset=4800, gate=9600)]), FONT)

    def test_repeat_render_identical(self):
        event = note(performance={"pitch_points": [[0, -50.0], [512, 0.0]], "expression_points": [[0, 90], [1024, 127]]})
        first, _ = engine.render_events(doc([event]), FONT)
        second, _ = engine.render_events(doc([event]), FONT)
        self.assertTrue(np.array_equal(first, second))

    def test_expression_curve_changes_level(self):
        quiet, _ = engine.render_events(doc([note(performance={"expression_points": [[0, 40]]})]), FONT)
        loud, _ = engine.render_events(doc([note(performance={"expression_points": [[0, 127]]})]), FONT)
        self.assertGreater(np.abs(loud[12000:36000]).max(), 2 * np.abs(quiet[12000:36000]).max())

    def test_sample_start_moves_attack(self):
        whole, _ = engine.render_events(doc([note(performance={"pitch_points": [[0, 0.0]]})]), FONT)
        skipped, _ = engine.render_events(doc([note(performance={"pitch_points": [[0, 0.0]], "sample_start_frames": 4800})]), FONT)
        self.assertFalse(np.array_equal(whole, skipped))

    def test_program_switch_per_note(self):
        """同一声部的另一音取别的预设：真实音色变了才算奏法切换。"""
        plain, _ = engine.render_events(doc([note(performance={"pitch_points": [[0, 0.0]]})]), FONT)
        other, _ = engine.render_events(doc([note(program=40, bank=0, performance={"pitch_points": [[0, 0.0]]})]), FONT)
        self.assertGreater(np.abs(plain[:48000] - other[:48000]).max(), .01)

    def test_channel_exhausted_reported(self):
        crowd = [note(f"n{i}", midi=40 + i % 40, gate=48000, performance={"pitch_points": [[0, 0.0]]}) for i in range(130)]
        with self.assertRaises(engine.EngineError) as caught:
            engine.render_events(doc(crowd), FONT)
        self.assertEqual(caught.exception.code, "channel_exhausted")

    def test_mix_conflicts_refused(self):
        document = doc([note(performance={"pitch_points": [[0, 0.0]]})])
        document["control_events"] = [{"event_id": "c", "part_id": "p", "frame": 0, "controller": 11, "value": 64}]
        with self.assertRaises(engine.EngineError) as caught:
            engine.render_events(document, FONT)
        self.assertEqual(caught.exception.code, "unsupported_performance_mix")


class CompileTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = load(ROOT / "tests/fixtures/音频闭环乐谱.json")
        cls.cap = load(ROOT / "tests/fixtures" / "engine-capabilities-v1.json")
        cls.mapping = inventory(FONT)
        cls.sha = cls.mapping["source"]["sha256"]
        cls.ready = copy.deepcopy(cls.cap)
        cls.ready["performance"]["note_expression"] = {"state": "supported", "evidence_ids": ["native_note_expression_probe"], "limitations": []}
        cls.ready["evidence"].append({"evidence_id": "native_note_expression_probe", "kind": "command_output", "reference": "测试", "description": "测试用"})

    def score(self):
        score = copy.deepcopy(self.base)
        note = score["parts"][0]["notes"][1]
        note["performance"] = {"technique": "slide_in", "pitch_curve": [{"at_tick": 0, "cents": -200}, {"at_tick": note["duration_ticks"] // 2, "cents": 0}],
                               "vibrato": {"start_tick": 0, "fade_ticks": 100, "rate_hz": 5.5, "depth_cents": 25}}
        return score

    def compile(self, score, cap=None):
        return compile_score(score, cap or self.ready, self.sha, self.mapping)

    def test_refused_without_probed_capability(self):
        with self.assertRaises(ScoreError):
            self.compile(self.score(), self.cap)

    def test_score_without_performance_compiles_as_before(self):
        events = compile_score(self.base, self.cap, self.sha, self.mapping)["events"]
        self.assertFalse(any("performance" in e or "program" in e for e in events))

    def test_compiled_note_keeps_score_curve(self):
        compiled = self.compile(self.score())
        event = next(e for e in compiled["events"] if "performance" in e)
        self.assertEqual(event["performance"]["technique"], "slide_in")
        self.assertEqual(event["performance"]["pitch_points"][0][1], -200.0)

    def test_compile_is_deterministic_and_freezes(self):
        score = self.score()
        compiled = self.compile(score)
        self.assertEqual(compiled, self.compile(score))
        frozen = validate_frozen(copy.deepcopy(compiled), score, self.ready, self.sha, soundfont_inventories={self.sha: self.mapping})
        self.assertEqual(frozen["events"], compiled["events"])

    def test_tampered_curve_rejected(self):
        score = self.score()
        compiled = self.compile(score)
        next(e for e in compiled["events"] if "performance" in e)["performance"]["pitch_points"][0][1] = 300.0
        with self.assertRaises(ScoreError):
            validate_frozen(compiled, score, self.ready, self.sha, soundfont_inventories={self.sha: self.mapping})

    def test_illegal_curve_rejected_at_validation(self):
        score = self.score()
        score["parts"][0]["notes"][1]["performance"]["pitch_curve"][0]["at_tick"] = 10 ** 7
        with self.assertRaises(ScoreError):
            validate_score(score)

    def test_compiled_score_really_renders(self):
        audio, diagnostic = engine.render_events(self.compile(self.score()), FONT)
        self.assertGreater(float(np.abs(audio).max()), .001)


class Sf2WriterTest(unittest.TestCase):
    def build(self, directory, name):
        tone = .4 * np.sin(2 * np.pi * 440 * np.arange(48000) / 44100) * np.exp(-np.arange(48000) / 22050)
        path = Path(directory) / "t.sf2"
        write_sf2(path, name, [{"name": "a4", "left": tone, "right": tone, "rate": 44100, "root": 69, "correction": -20}],
                  [{"name": "t", "bank": 0, "program": 0, "zones": [{"sample": "a4", "keys": (60, 80), "velocities": (1, 127),
                                                                      "generators": {"attack_vol_env": -12000, "release_vol_env": -2400}}]}])
        return path

    def test_loads_for_odd_and_even_bank_names(self):
        for name in ("odd", "even"):
            with tempfile.TemporaryDirectory() as directory:
                path = self.build(directory, name)
                audio, _ = engine.render_events(doc([note(gate=40000)], program=0), path)
                self.assertGreater(float(np.abs(audio).max()), .01)
                self.assertEqual(inventory(path)["presets"][0]["mappings"][0]["key_range"], [60, 80])

    def test_pitch_correction_applied(self):
        with tempfile.TemporaryDirectory() as directory:
            audio, _ = engine.render_events(doc([note(gate=40000)], program=0), self.build(directory, "name"))
        self.assertAlmostEqual(1200 * math.log2(peak_hz(audio, 4000, 30000) / 440), -20, delta=4)


class RoundRobinTest(unittest.TestCase):
    RESOURCE = {"articulation_programs": {"normal": {"bank": 0, "programs": [0, 1], "recorded": True}}}

    @staticmethod
    def notes(count, midi=60):
        return [{"note_id": f"n{i}", "onset_tick": i * 480, "pitch": {"kind": "pitched", "midi": midi}} for i in range(count)]

    def pick(self, notes, seed="7"):
        return round_robin(notes, self.RESOURCE, "p", "s", seed, lambda n: "normal", 0, 0)

    def test_same_pitch_never_repeats(self):
        chosen = self.pick(self.notes(6))
        sequence = [chosen[f"n{i}"] for i in range(6)]
        self.assertTrue(all(a != b for a, b in zip(sequence, sequence[1:])))

    def test_other_pitch_does_not_disturb(self):
        alone = self.pick(self.notes(4))
        mixed = self.pick(self.notes(4) + [{"note_id": "x", "onset_tick": 100, "pitch": {"kind": "pitched", "midi": 72}}])
        self.assertEqual({k: mixed[k] for k in alone}, alone)

    def test_single_program_is_passthrough(self):
        self.assertEqual(round_robin(self.notes(2), {}, "p", "s", "7", lambda n: "normal", 0, 5), {"n0": 5, "n1": 5})


if __name__ == "__main__":
    unittest.main()
