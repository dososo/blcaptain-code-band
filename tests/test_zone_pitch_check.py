import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "skills/code-band-bgm/scripts"))
from sf2_writer import write_sf2  # noqa: E402
from zone_pitch_check import check  # noqa: E402


def tone(midi, seconds=1.5, rate=44100, extra=None):
    t = np.arange(int(seconds * rate)) / rate
    hz = 440 * 2 ** ((midi - 69) / 12)
    wave = sum(np.sin(2 * np.pi * hz * n * t) / n for n in (1, 2, 3)) * np.exp(-t * 2)
    if extra is not None:
        wave = wave + 2 * np.sin(2 * np.pi * 440 * 2 ** ((extra - 69) / 12) * t) * np.exp(-t * 2)
    return (0.2 * wave).astype(np.float64)


class ZonePitchCheckTest(unittest.TestCase):
    def build(self, folder, wrong):
        bad = tone(73) if wrong == "other_note" else tone(76, extra=78)
        samples = [{"name": "good", "left": tone(74), "right": None, "rate": 44100, "root": 74, "correction": 0},
                   {"name": "bad", "left": bad, "right": None, "rate": 44100, "root": 76, "correction": 0}]
        zones = [{"sample": "good", "keys": [70, 74], "velocities": [1, 127], "generators": {}},
                 {"sample": "bad", "keys": [75, 79], "velocities": [1, 127], "generators": {}}]
        path = Path(folder) / "t.sf2"
        write_sf2(path, "T", samples, [{"name": "P", "bank": 0, "program": 0, "zones": zones}])
        return path

    def test_mislabeled_sample_is_suspect_and_correct_one_is_not(self):
        with tempfile.TemporaryDirectory() as folder:
            rows = {r["key"]: r for r in check(self.build(folder, "other_note"))}
            self.assertFalse(rows[74]["suspect"])
            self.assertTrue(rows[76]["suspect"])

    def test_bleed_louder_than_root_is_suspect(self):
        with tempfile.TemporaryDirectory() as folder:
            rows = {r["key"]: r for r in check(self.build(folder, "bleed"))}
            self.assertTrue(rows[76]["suspect"])


if __name__ == "__main__":
    unittest.main()
