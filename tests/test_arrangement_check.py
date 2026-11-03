"""编排检查：声部关系、唯一高峰、动机回认与锚点数值。"""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "skills/code-band-bgm/scripts"))
import idiomatic as idiom  # noqa: E402
from arrangement_check import PlanError, check_arrangement  # noqa: E402

PPQ = 960


def part(part_id, notes):
    return {"part_id": part_id, "label": part_id, "instrument_role": part_id, "musical_function": "melody",
            "resource_binding": {"state": "resolved", "engine_id": "fluidsynth", "capability_id": "fluidsynth_local_v1",
                                 "instrument_id": "piano", "mapping_version": "1.0.0", "substitution_policy": "report_only"},
            "articulation_map": {"normal": {"state": "resolved", "engine_articulation_id": "normal"}},
            "gain_db": 0, "pan": 0, "humanize": {"enabled": False, "profile_id": "written_out_expression"},
            "notes": sorted(notes, key=lambda n: (n["onset_tick"], n["note_id"]))}


def score(parts, bars=4, sections=(("a", 0, 2), ("b", 2, 4))):
    return {"schema_version": "1.0.0", "score_id": "t", "document_state": "composition_candidate",
            "intent": {"title": "t", "brief": "t", "use_case": "测试", "mood_tags": ["t"], "notes": "t",
                       "reference_policy": {"mode": "none", "source_ids": [], "reuse_reference_audio": False, "reuse_transcribed_melody": False}},
            "timeline": {"ppq": PPQ, "tempo_map": [{"start_tick": 0, "bpm": {"numerator": 60, "denominator": 1}}],
                         "meter_map": [{"start_tick": 0, "numerator": 4, "denominator": 4}], "music_end_tick": bars * 4 * PPQ},
            "randomness": {"master_seed": "1", "algorithm": "sha256_counter_v1", "namespace_version": "1", "sample_selection": "seeded_choice", "notes": "t"},
            "sections": [{"section_id": s, "label": s, "start_tick": a * 4 * PPQ, "end_tick": b * 4 * PPQ, "function": "main", "notes": s}
                         for s, a, b in sections],
            "parts": parts, "mix": {"master_gain_db": 0, "effects": [], "notes": "t"},
            "render": {"sample_rate_hz": 48000, "sample_format": "pcm_s24", "channels": 2, "duration_mode": "musical_length",
                       "tail_policy": "append", "tail_frames": 48000, "end_fade_frames": 4800, "loop": {"enabled": False}}}


def line(prefix, items):
    """items: (拍位, 拍数, MIDI)"""
    return [idiom.note(f"{prefix}{i:02d}", beat * PPQ, beats * PPQ, midi, 80 + (i % 3) * 10) for i, (beat, beats, midi) in enumerate(items)]


MELODY = line("m", [(0, 1, 60), (1, 1, 62), (2, 2, 64), (4, 1, 60), (5, 1, 62), (6, 2, 65), (8, 1, 67), (9, 1, 69), (10, 2, 72), (12, 4, 67)])


class ArrangementTest(unittest.TestCase):
    def check(self, parts, plan, **kwargs):
        return {c["check_id"]: c for c in check_arrangement(score(parts, **kwargs), {"schema_version": "1.0.0", **plan})["checks"]}

    def test_answer_on_melody_onsets_fails_interleaved_passes(self):
        roles = {"roles": {"melody": "foreground", "answer": "answer"}}
        clash = self.check([part("melody", MELODY), part("answer", line("a", [(0, 1, 55), (1, 1, 57), (4, 1, 55), (8, 1, 57)]))], roles)
        self.assertEqual(clash["answer_onset_collision"]["status"], "不通过")
        reply = self.check([part("melody", MELODY), part("answer", line("a", [(2.5, .5, 55), (3, 1, 57), (6.5, .5, 55), (14, 1, 57)]))], roles)
        self.assertEqual(reply["answer_onset_collision"]["status"], "通过")

    def test_unique_peak_in_declared_section(self):
        plan = {"roles": {"melody": "foreground"}, "peak_section": "b"}
        good = self.check([part("melody", MELODY)], plan)
        self.assertEqual((good["unique_peak"]["status"], good["peak_reaches_top"]["status"]), ("通过", "通过"))
        repeated = self.check([part("melody", line("m", [(0, 1, 72), (1, 3, 60), (8, 2, 72), (10, 2, 64)]))], plan)
        self.assertEqual(repeated["unique_peak"]["status"], "需核")
        self.assertEqual(repeated["unique_peak"]["value"]["sections"], ["a", "b"])
        self.assertEqual(repeated["peak_reaches_top"]["status"], "通过")
        missed = self.check([part("melody", line("m", [(0, 1, 74), (8, 2, 72)]))], plan)["peak_reaches_top"]
        self.assertEqual(missed["status"], "不通过")

    def test_motif_needs_recall_and_varied_tail(self):
        plan = {"roles": {"melody": "foreground"}, "motifs": [{"motif_id": "x", "part_id": "melody", "intervals": [2, 2, -4]}]}
        result = self.check([part("melody", MELODY)], plan)["motif:x"]
        # 60 62 64 | 60：原样一次；65 67 69 | 72：前两个音程同、尾音程变成 +3，算变尾
        self.assertEqual(result["value"], {"exact": 1, "varied_tail": 1})
        self.assertEqual(result["status"], "通过")
        same = line("m", [(0, 1, 60), (1, 1, 62), (2, 1, 64), (4, 1, 60), (5, 1, 62), (6, 1, 64)])
        self.assertEqual(self.check([part("melody", same)], plan)["motif:x"]["status"], "需核")

    def test_handoff_gap_and_masking(self):
        plan = {"roles": {"melody": "foreground", "strings": "harmony"}, "sections": {"b": {"foreground": "strings"}}}
        strings = line("s", [(0, 8, 61), (11, 5, 67)])
        result = self.check([part("melody", MELODY[:6]), part("strings", strings)], plan)
        self.assertEqual(result["handoff:a->b"]["value"], 3.0)
        self.assertEqual(result["handoff:a->b"]["status"], "需核")
        self.assertGreater(result["register_masking"]["value"], .3)

    def test_sparse_solo_only_reports_anchor_numbers(self):
        result = self.check([part("melody", MELODY)], {"profile": "sparse_solo", "roles": {"melody": "foreground"}})
        self.assertEqual(result["anchor:polyphony_mean"]["status"], "仅报告")
        ensemble = self.check([part("melody", MELODY)], {"roles": {"melody": "foreground"}})
        self.assertEqual(ensemble["anchor:polyphony_mean"]["status"], "需核")
        # 谱面口径下认可与否决分不开的量，任何 profile 都只报告
        self.assertEqual(ensemble["anchor:onset_density_per_s"]["status"], "仅报告")

    def test_plan_errors(self):
        with self.assertRaises(PlanError):
            self.check([part("melody", MELODY)], {"roles": {"nobody": "foreground"}})
        with self.assertRaises(PlanError):
            self.check([part("melody", MELODY)], {"roles": {"melody": "answer"}})

    def test_cli_writes_report_without_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            (folder / "s.json").write_text(json.dumps(score([part("melody", MELODY)])), encoding="utf-8")
            (folder / "p.json").write_text(json.dumps({"schema_version": "1.0.0", "roles": {"melody": "foreground"}, "peak_section": "b"}), encoding="utf-8")
            command = [sys.executable, "-B", str(ROOT / "skills/code-band-bgm/scripts/band.py"), "arrange-check", "--score", str(folder / "s.json"),
                       "--plan", str(folder / "p.json"), "--out", str(folder / "r.json")]
            first = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
            report = json.loads((folder / "r.json").read_text(encoding="utf-8"))
            self.assertIn("plan_file_sha256", report)
            self.assertNotEqual(subprocess.run(command, capture_output=True, text=True).returncode, 0)


if __name__ == "__main__":
    unittest.main()
