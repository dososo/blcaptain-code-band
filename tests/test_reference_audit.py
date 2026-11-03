"""固定窗口测量的阳性/反相/退化对照，不设虚构的风格分数线。"""
from pathlib import Path
import sys
import unittest
import json
import subprocess
import tempfile
import soundfile as sf
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'skills/code-band-bgm/scripts'))
from audio_tools import AudioError
from reference_audit import acoustic_profile, audit_reference


class AcousticTests(unittest.TestCase):
    def tone(self, hz, reverse=False):
        x = .2*np.sin(2*np.pi*hz*np.arange(48000)/48000)
        return np.column_stack([x, -x if reverse else x])

    def test_frequency_and_gain_controls(self):
        bass = acoustic_profile(self.tone(100))
        bright = acoustic_profile(self.tone(3000))
        self.assertGreater(bass['band_energy_fractions'][1], .99)
        self.assertGreater(bright['band_energy_fractions'][4], .99)
        quiet = acoustic_profile(self.tone(100)*.25)
        np.testing.assert_allclose(bass['band_energy_fractions'], quiet['band_energy_fractions'], atol=1e-12)
        self.assertAlmostEqual(bass['crest_db'], quiet['crest_db'])

    def test_nyquist_energy_is_included_in_highest_band(self):
        x=np.tile([.2,-.2],24000)
        p=acoustic_profile(np.column_stack([x,x]))
        self.assertAlmostEqual(sum(p["band_energy_fractions"]),1)
        self.assertGreater(p["band_energy_fractions"][-1],.99)

    def test_antiphase_does_not_erase_stereo_spectrum(self):
        p = acoustic_profile(self.tone(100, True))
        self.assertGreater(p['band_energy_fractions'][1], .99)
        self.assertAlmostEqual(p['mono_energy_fraction'], 0)
        self.assertAlmostEqual(p['side_energy_fraction'], 1)

    def test_same_total_width_can_hide_opposite_band_placement(self):
        low = self.tone(1000)
        high = self.tone(8000)
        high_sides = high * np.array([1, -1])
        low_sides = low * np.array([1, -1])
        a = acoustic_profile(low + high_sides)
        b = acoustic_profile(low_sides + high)
        self.assertAlmostEqual(a['side_energy_fraction'], b['side_energy_fraction'])
        self.assertLess(a['band_side_energy_fractions'][3], 1e-6)
        self.assertGreater(a['band_side_energy_fractions'][5], .999999)
        self.assertGreater(b['band_side_energy_fractions'][3], .999999)
        self.assertLess(b['band_side_energy_fractions'][5], 1e-6)
        quiet = acoustic_profile((low + high_sides) * .25)
        self.assertEqual(a['band_side_energy_fractions'], quiet['band_side_energy_fractions'])

    def test_negligible_band_width_is_unknown(self):
        # 3000Hz落在4096点Welch的整数bin，避免把窗泄漏误当无能量。
        profile = acoustic_profile(self.tone(3000))
        self.assertIsNone(profile['band_side_energy_fractions'][0])
        self.assertLess(profile['band_side_energy_fractions'][4], 1e-12)

    def test_degenerate_audio_and_windows_fail(self):
        for wave in [np.zeros((48000, 2)), np.ones((48000, 2)), np.full((48000, 2), np.nan), np.zeros((10, 1))]:
            with self.subTest(shape=wave.shape), self.assertRaises(AudioError):
                acoustic_profile(wave)
        for args in [{'seconds':float('nan')}, {'reference_start':-1}, {'seconds':31}, {'seconds':True}]:
            with self.subTest(args=args), self.assertRaises(AudioError):
                audit_reference('未读取的路径', '未读取的路径', **args)

    def test_real_cli_decodes_exact_frames_and_refuses_existing_output(self):
        with tempfile.TemporaryDirectory() as folder:
            base=Path(folder); source=base/'音频 $(无需执行);.wav'; output=base/'结果.json'
            sf.write(source,self.tone(100),48000,subtype='PCM_24')
            argv=[sys.executable,str(ROOT/'skills/code-band-bgm/scripts/band.py'),'audit-reference',
                  '--reference',str(source),'--candidate',str(source),'--seconds','1','--out',str(output)]
            first=subprocess.run(argv,capture_output=True,text=True)
            self.assertEqual(first.returncode,0,first.stdout+first.stderr)
            report=json.loads(output.read_text())
            self.assertEqual(report['reference']['window_frames'],48000)
            self.assertEqual(report['tools']['reference_audit'],'0.1.1')
            self.assertTrue(report['method']['highest_band_includes_nyquist'])
            self.assertEqual(len(report['reference']['profile']['band_side_energy_fractions']), 7)
            self.assertEqual(report['candidate_minus_reference']['band_side_energy_fraction_delta'][1], 0)
            self.assertEqual(report['reference']['decoded_f32le_sha256'],report['candidate']['decoded_f32le_sha256'])
            saved=output.read_bytes()
            again=subprocess.run(argv,capture_output=True,text=True)
            self.assertEqual(again.returncode,2)
            self.assertEqual(output.read_bytes(),saved)
            with self.assertRaises(AudioError):
                audit_reference(source,source,reference_start=.5,seconds=1)


if __name__ == '__main__':
    unittest.main()
