"""验证音乐时间、真实资源绑定、随机流和版本保留，不用音频文件冒充作品。"""
import copy
from fractions import Fraction
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "skills/code-band-bgm" / "scripts"))
from score_tools import (ScoreError, compile_score, digest, load, random_int,
                         revise_gain, round_fraction, tick_frame, validate_capabilities, validate_score, validate_frozen)
from soundfont_inventory import inventory


class ScoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = load(ROOT / "tests/fixtures/音频闭环乐谱.json")
        cls.cap = load(ROOT / "tests/fixtures" / "engine-capabilities-v1.json")
        cls.mapping = inventory(ROOT / "skills/code-band-bgm/assets/soundfonts/MuseScore_General.sf3")
        cls.sha = cls.mapping["source"]["sha256"]

    def compile(self, score, cap=None, sha=None, mapping=None):
        return compile_score(score, cap or self.cap, sha or self.sha, mapping or self.mapping)

    def test_absolute_rational_tempo_and_half_rounding(self):
        t = {"ppq": 960, "tempo_map": [{"start_tick": 0, "bpm": {"numerator": 120, "denominator": 1}}, {"start_tick": 960, "bpm": {"numerator": 60, "denominator": 1}}]}
        self.assertEqual(tick_frame(1920, t, 48000), 72000)
        self.assertEqual(round_fraction(Fraction(1, 2)), 1)
        self.assertEqual(round_fraction(Fraction(-1, 2)), -1)
        self.assertEqual(tick_frame(960 * 999, t, 48000), 47928000)

    def test_fixed_events_and_mix_independence(self):
        score = copy.deepcopy(self.base)
        part = score["parts"][0]
        part["humanize"] = {"enabled": True, "profile_id": "test", "rationale": "测试分流", "timing": {"max_offset_us": 3000, "distribution": "symmetric_triangular_integer", "chord_policy": "shared_offset", "preserve_voice_order": True, "lock_section_starts": True}, "velocity": {"max_delta": 5, "distribution": "symmetric_triangular_integer", "respect_notated_accents": True}}
        part["notes"][1]["timing_locked"] = False
        first = self.compile(score)
        self.assertEqual(first, self.compile(score))
        revised, change = revise_gain(score, "piano", -4, [])
        second = self.compile(revised)
        self.assertEqual(first["events"], second["events"])
        self.assertNotEqual(first["parts"], second["parts"])
        self.assertEqual(first["events"][0]["onset_frame"], 0)
        self.assertTrue(change["seed_preserved"])

    def test_shared_chord_and_stable_note_identity(self):
        score = copy.deepcopy(self.base)
        p = score["parts"][0]
        n = copy.deepcopy(p["notes"][1])
        n["pitch"]["midi"] = 65
        n["note_id"] = "other_chord_note"
        p["notes"].append(n)
        a = self.compile(score)
        b = copy.deepcopy(score)
        b["parts"][0]["notes"].reverse()
        self.assertEqual(a["events"], self.compile(b)["events"])
        self.assertTrue(all(e["onset_frame"] % 64 == 0 and e["gate_frames"] % 64 == 0 for e in a["events"]))
        self.assertTrue(all(e["soundfont_sample_ids"] for e in a["events"]))

    def test_uint64_and_namespaces(self):
        bad = copy.deepcopy(self.base)
        bad["randomness"]["master_seed"] = str(2**64)
        with self.assertRaises(ScoreError): validate_score(bad)
        self.assertEqual(random_int("10", "humanize_timing", "x", 100), random_int("10", "humanize_timing", "x", 100))
        a = [random_int("10", "humanize_timing", str(i), 1000) for i in range(5)]
        b = [random_int("10", "humanize_velocity", str(i), 1000) for i in range(5)]
        self.assertNotEqual(a, b)

    def test_bad_timeline_sections_and_exact_duration(self):
        for mutation in (lambda s: s["timeline"]["tempo_map"][0].update(start_tick=2), lambda s: s["sections"][0].update(start_tick=1), lambda s: s["render"].update(target_file_frames=239999), lambda s: s["parts"][0]["notes"][0].update(duration_ticks=9000)):
            score = copy.deepcopy(self.base); mutation(score)
            with self.assertRaises(ScoreError): validate_score(score)

    def test_meter_change_requires_bar_boundary(self):
        s = copy.deepcopy(self.base)
        s["timeline"]["meter_map"].append({"start_tick": 960, "numerator": 3, "denominator": 4})
        with self.assertRaises(ScoreError): validate_score(s)

    def test_pending_missing_and_changed_assets_rejected(self):
        mutations = [lambda s: s.update(document_state="structure_example"), lambda s: s["parts"][0]["resource_binding"].update(instrument_id="missing"), lambda s: s["parts"][0]["resource_binding"].update(mapping_version="changed"), lambda s: s["parts"][0]["articulation_map"]["normal"].update(engine_articulation_id="legato")]
        for mutation in mutations:
            s = copy.deepcopy(self.base); mutation(s)
            with self.assertRaises(ScoreError): self.compile(s)
        with self.assertRaises(ScoreError): self.compile(self.base, sha="0" * 64)

    def test_real_zone_holes_and_key_range(self):
        for pitch, velocity in ((0, 70), (60, 1)):
            s = copy.deepcopy(self.base)
            s["parts"][0]["notes"][0]["pitch"]["midi"] = pitch
            s["parts"][0]["notes"][0]["velocity"] = velocity
            with self.assertRaises(ScoreError): self.compile(s)

    def test_same_pitch_overlap_rejected(self):
        s = copy.deepcopy(self.base)
        p = s["parts"][0]; n = copy.deepcopy(p["notes"][0]); n["note_id"] = "overlap"; n["onset_tick"] = 120
        p["notes"].append(n)
        with self.assertRaises(ScoreError): validate_score(s)
        with self.assertRaises(ScoreError): self.compile(s)

    def test_same_pitch_adjacent_notes_are_valid(self):
        s = copy.deepcopy(self.base)
        p = s["parts"][0]; n = copy.deepcopy(p["notes"][0]); n["note_id"] = "adjacent"
        n["onset_tick"] = p["notes"][0]["onset_tick"] + p["notes"][0]["duration_ticks"]
        p["notes"] = [p["notes"][0], n]
        self.assertTrue(validate_score(s)["render_ready"])

    def test_humanized_unlocked_overlap_defers_to_compiler(self):
        s = copy.deepcopy(self.base)
        p = s["parts"][0]
        p["humanize"] = {"enabled": True, "profile_id": "test", "rationale": "测试时序不确定性", "timing": {"max_offset_us": 3000, "distribution": "symmetric_triangular_integer", "chord_policy": "shared_offset", "preserve_voice_order": True, "lock_section_starts": True}, "velocity": {"max_delta": 5, "distribution": "symmetric_triangular_integer", "respect_notated_accents": True}}
        first = copy.deepcopy(p["notes"][0]); first["onset_tick"] = 240; first["timing_locked"] = False
        second = copy.deepcopy(first); second["note_id"] = "second"; second["onset_tick"] = 360
        p["notes"] = [first, second]
        self.assertTrue(validate_score(s)["render_ready"])
        with self.assertRaises(ScoreError): self.compile(s)

    def test_unsupported_round_robin_rejected(self):
        s = copy.deepcopy(self.base)
        s["randomness"]["sample_selection"] = "seeded_round_robin"
        with self.assertRaises(ScoreError): self.compile(s)

    def test_targeted_reverb_compiles_and_rejects_unknown_part(self):
        s = copy.deepcopy(self.base)
        s["mix"]["effects"] = [{"effect_id": "room", "kind": "reverb",
                                 "target_part_ids": ["piano"], "parameters": {"level": 0.2},
                                 "required": True, "notes": "只给旋律声部混响"}]
        compiled = self.compile(s)
        self.assertEqual(compiled["mix"]["effects"][0]["target_part_ids"], ["piano"])
        s["mix"]["effects"][0]["target_part_ids"] = ["missing"]
        with self.assertRaises(ScoreError): self.compile(s)

    def test_frozen_replay_accepts_probe_date_change_without_recompile(self):
        frozen = self.compile(self.base)
        cap = copy.deepcopy(self.cap)
        cap["observed_at"] = "2026-10-31T00:00:00+00:00"
        self.assertIs(validate_frozen(frozen, self.base, cap, self.sha), frozen)
        bad = copy.deepcopy(frozen)
        bad["events"][0]["part_id"] = "not_in_score"
        with self.assertRaises(ScoreError): validate_frozen(bad, self.base, cap, self.sha)
        for change in (lambda s: s.update(seed="999"), lambda s: s["events"][0].update(midi=61), lambda s: s["events"][0].update(onset_frame=64)):
            bad = copy.deepcopy(frozen); change(bad)
            with self.assertRaises(ScoreError): validate_frozen(bad, self.base, cap, self.sha)

    def test_capability_evidence_not_claimable_when_unprobed(self):
        c = load(ROOT / "tests/fixtures" / "engine-capabilities.json")
        c["export"]["wav"] = {"state": "supported", "evidence_ids": ["invented"], "limitations": []}
        with self.assertRaises(ScoreError): validate_capabilities(c)
        c = copy.deepcopy(self.cap)
        c["resources"][0]["label"] = "changed"
        with self.assertRaises(ScoreError): validate_capabilities(c)

    def test_preserve_conflict_and_actual_content_hash(self):
        s = copy.deepcopy(self.base)
        p = copy.deepcopy(s["parts"][0]); p["part_id"] = "other"; s["parts"].append(p)
        revised, change = revise_gain(s, "other", -5, ["piano"])
        self.assertTrue(change["preserved_parts"]["piano"])
        self.assertEqual(revised["parts"][0], s["parts"][0])
        with self.assertRaises(ScoreError): revise_gain(s, "other", -5, ["other"])

    def test_loop_conflicts_and_no_pending_audio(self):
        s = copy.deepcopy(self.base)
        s["render"]["loop"] = {"enabled": True, "start_tick": 0, "end_tick": 7680, "boundary_method": "tail_wrap_crossfade", "crossfade_frames": 0, "notes": "测试"}
        with self.assertRaises(ScoreError): validate_score(s)
        example = load(ROOT / "tests/fixtures/乐谱_结构示例.json")
        self.assertFalse(validate_score(example)["render_ready"])
        with self.assertRaises(ScoreError): self.compile(example)

    def test_cli_unwritable_and_no_overwrite(self):
        runner = str(ROOT / "skills/code-band-bgm/scripts/band.py")
        with tempfile.TemporaryDirectory() as tmp:
            occupied = Path(tmp) / "occupied.json"; occupied.write_text("原内容")
            args = [sys.executable, runner, "revise", "--score", str(ROOT / "tests/fixtures/音频闭环乐谱.json"), "--part", "piano", "--gain-db", "-4", "--out", str(occupied)]
            r = subprocess.run(args, capture_output=True, text=True)
            self.assertEqual(r.returncode, 2)
            self.assertEqual(occupied.read_text(), "原内容")
            args[-1] = str(occupied / "cannot.json")
            r = subprocess.run(args, capture_output=True, text=True)
            self.assertEqual(r.returncode, 2)
            self.assertEqual(json.loads(r.stdout)["status"], "失败")


if __name__ == "__main__":
    unittest.main()
