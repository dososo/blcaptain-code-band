import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "skills/code-band-bgm/scripts"
sys.path.insert(0, str(SCRIPTS))
import score_edit  # noqa: E402

PPQ = 960


def note(note_id, onset, duration, midi):
    return {"note_id": note_id, "onset_tick": onset, "duration_ticks": duration,
            "pitch": {"kind": "pitched", "midi": midi, "cents": 0}, "velocity": 80, "articulation": "normal",
            "performance": {"expression_curve": [{"at_tick": 0, "value": 60}, {"at_tick": duration, "value": 90}]}}


def score(bpm=84, bars=4):
    parts = [{"part_id": "lead", "gain_db": 0, "pan": 0, "notes": [note(f"l{i}", i * 4 * PPQ, 4 * PPQ - 40, 60 + i) for i in range(bars)]},
             {"part_id": "bass", "gain_db": 0, "pan": 0, "notes": [note(f"b{i}", i * 4 * PPQ, 2 * PPQ, 36) for i in range(bars)]}]
    return {"score_id": "demo", "parts": parts,
            "timeline": {"ppq": PPQ, "tempo_map": [{"start_tick": 0, "bpm": {"numerator": bpm, "denominator": 1}}],
                         "meter_map": [{"start_tick": 0, "numerator": 4, "denominator": 4}], "music_end_tick": bars * 4 * PPQ},
            "sections": [{"section_id": "s0", "start_tick": 0, "end_tick": bars * 4 * PPQ}],
            "render": {"sample_rate_hz": 48000, "duration_mode": "musical_length", "tail_policy": "append", "tail_frames": 48000,
                       "end_fade_frames": 9600, "loop": {"enabled": False}}}


class ScoreEditTest(unittest.TestCase):
    def test_shorten_keeps_earlier_notes_and_score_id(self):
        result = score_edit.shorten(score(), 2)
        report = score_edit.diff_scores(score(), result)
        self.assertEqual(result["score_id"], "demo")  # 随机流命名空间不变
        self.assertEqual(report["parts"]["lead"]["identical_notes"], 2)
        self.assertEqual(report["parts"]["lead"]["removed_notes"], 2)
        self.assertEqual(result["timeline"]["music_end_tick"], 2 * 4 * PPQ)
        for part in result["parts"]:
            for item in part["notes"]:
                self.assertLessEqual(item["onset_tick"] + item["duration_ticks"], 2 * 4 * PPQ - 30)

    def test_shorten_trims_curves_past_new_end(self):
        long = score(bars=2)
        long["parts"][0]["notes"] = [note("x", 0, 8 * PPQ - 10, 60)]
        result = score_edit.shorten(long, 1)
        item = result["parts"][0]["notes"][0]
        self.assertEqual(item["duration_ticks"], 4 * PPQ - 30)
        self.assertTrue(all(p["at_tick"] <= item["duration_ticks"] for p in item["performance"]["expression_curve"]))

    def test_loop_refuses_period_not_multiple_of_64(self):
        with self.assertRaises(SystemExit):
            score_edit.make_loop(score(bpm=84, bars=8), 8)

    def test_loop_with_tempo_that_fits_grid(self):
        result = score_edit.make_loop(score(bpm=84, bars=8), 8, bpm=90)
        self.assertTrue(result["render"]["loop"]["enabled"])
        self.assertEqual(result["render"]["tail_policy"], "wrap_into_loop")
        self.assertEqual(result["render"]["loop"]["crossfade_frames"], 0)

    def test_add_part_leaves_existing_parts_identical(self):
        donor = score()
        donor["parts"].append({"part_id": "pad", "gain_db": 0, "pan": 0, "notes": [note("p0", 0, PPQ, 55)]})
        result = score_edit.add_part(score(), donor, "pad")
        report = score_edit.diff_scores(score(), result)
        self.assertEqual(report["parts"]["lead"]["status"], "identical")
        self.assertEqual(report["parts"]["pad"]["status"], "added")
        self.assertTrue(score_edit.check_preserved(report, ["lead", "bass"])["all_identical"])

    def test_add_part_refuses_different_timeline(self):
        donor = score(bpm=90)
        donor["parts"].append({"part_id": "pad", "gain_db": 0, "pan": 0, "notes": []})
        with self.assertRaises(SystemExit):
            score_edit.add_part(score(), donor, "pad")

    def test_event_diff_catches_program_change(self):
        base = {"events": [{"part_id": "lead", "onset_frame": 0, "gate_frames": 64, "midi": 60, "velocity": 80, "articulation": "normal",
                            "bank": 0, "program": 0, "soundfont_sample_ids": ["a"]}]}
        other = json.loads(json.dumps(base))
        other["events"][0]["program"] = 1  # 轮换换了录音，起点音高都没变
        self.assertTrue(score_edit.diff_events(base, base)["lead"]["identical"])
        self.assertFalse(score_edit.diff_events(base, other)["lead"]["identical"])

    def test_cli_refuses_to_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, out = Path(tmp) / "s.json", Path(tmp) / "out.json"
            src.write_text(json.dumps(score()), encoding="utf-8")
            out.write_text("{}", encoding="utf-8")
            with self.assertRaises(SystemExit):
                score_edit.main(["shorten", "--score", str(src), "--bars", "2", "--out", str(out)])
            self.assertEqual(out.read_text(encoding="utf-8"), "{}")


if __name__ == "__main__":
    unittest.main()
