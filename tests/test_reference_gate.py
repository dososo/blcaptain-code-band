import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

SCRIPTS = Path(__file__).resolve().parents[1] / "skills/code-band-bgm/scripts"
sys.path.insert(0, str(SCRIPTS))
from reference_gate import decide, measure  # noqa: E402

spec = importlib.util.spec_from_file_location("audio_tools", SCRIPTS / "audio_tools.py")
audio_tools = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audio_tools)

RATE = 22050


def music(seconds=6):
    t = np.arange(RATE * seconds) / RATE
    tone = .3 * np.sin(2 * np.pi * 220 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 2 * t))
    return np.column_stack([tone, tone])


class ReferenceGateTest(unittest.TestCase):
    def test_low_noise_is_refused(self):
        noise = np.random.default_rng(1).normal(0, 10 ** (-80 / 20), (RATE * 4, 2))
        decision = decide(measure(noise))
        self.assertEqual(decision["signal"], "拒绝")
        self.assertFalse(decision["mono_models_allowed"])

    def test_inverted_stereo_is_phase_anomaly(self):
        frames = music()
        frames[:, 1] *= -1
        measured = measure(frames)
        self.assertLess(measured["lr_correlation"], 0)
        decision = decide(measured)
        self.assertEqual(decision["phase"], "异常")
        self.assertFalse(decision["mono_models_allowed"])

    def test_correlated_stereo_passes_and_reports_true_peak(self):
        frames = music()
        measured = measure(frames)
        self.assertAlmostEqual(measured["peak_dbfs"], 20 * np.log10(np.abs(frames).max()), places=2)
        self.assertAlmostEqual(measured["mono_folddown_loss_db"], 0.0, places=2)
        self.assertTrue(decide(measured)["mono_models_allowed"])

    def test_analyze_refuses_noise_and_inverted_and_keeps_peak_honest(self):
        with tempfile.TemporaryDirectory() as folder:
            cases = {}
            for name, frames in (("noise", np.random.default_rng(2).normal(0, 1e-4, (RATE * 6, 2))),
                                 ("inverted", music() * np.array([1, -1])), ("music", music())):
                path = Path(folder) / f"{name}.wav"
                sf.write(path, frames, RATE, subtype="PCM_24")
                cases[name] = audio_tools.analyze(path)
            for name in ("noise", "inverted"):
                self.assertEqual(cases[name]["estimates"]["status"], "拒绝")
                self.assertEqual(cases[name]["estimates"]["tempo_candidates"], [])
                self.assertEqual(cases[name]["reference_understanding"]["status"], "拒绝")
            good = cases["music"]
            self.assertTrue(good["signal_gate"]["mono_models_allowed"])
            self.assertLess(abs(good["measurements"]["analysis_sample_peak_dbfs"] - 20 * np.log10(.3)), .2)
            for key in good["estimates"]["key_candidates"]:
                self.assertIsNone(key["tonic"])
            for tempo in good["estimates"]["tempo_candidates"]:
                if tempo["bpm"]:
                    self.assertFalse(tempo["single_value_allowed"])


if __name__ == "__main__":
    unittest.main()
