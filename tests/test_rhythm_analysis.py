"""成熟节拍后端用已知节拍信号验证；该夹具不计为产品音乐。"""
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'skills/code-band-bgm/scripts'))
from rhythm_analysis import analyze_rhythm


@unittest.skipUnless(importlib.util.find_spec('librosa'), '主渲染环境不依赖可选librosa；独立分析环境另行实测')
class RhythmTests(unittest.TestCase):
    def test_known_120bpm_click_train_is_not_confused_with_zero(self):
        rate = 22050
        audio = np.zeros(rate * 12)
        click = np.random.default_rng(20261031).normal(0, .1, 220) * np.linspace(1, 0, 220)
        for frame in range(0, len(audio) - len(click), rate // 2):
            audio[frame:frame + len(click)] += click
        report = analyze_rhythm(audio, rate)
        self.assertEqual(report['status_scope'], 'rhythm_measurement_candidates')
        self.assertTrue(116 <= report['windows'][0]['tempo_bpm'] <= 124)
        self.assertGreater(len(report['windows'][0]['beat_times_seconds']), 10)

    def test_silence_has_no_invented_tempo(self):
        report = analyze_rhythm(np.zeros(22050 * 5), 22050)
        self.assertIsNone(report['windows'][0]['tempo_bpm'])
        self.assertEqual(report['windows'][0]['beat_times_seconds'], [])

    def test_nonfinite_stereo_and_too_short_fail(self):
        for audio in [np.zeros((22050 * 5, 2)), np.full(22050 * 5, np.nan), np.zeros(22050)]:
            with self.assertRaises(ValueError):
                analyze_rhythm(audio, 22050)


if __name__ == '__main__':
    unittest.main()
