"""真实短采样自然结束与仍按住声部错误的门禁回归。"""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'skills/code-band-bgm/scripts'))
import fluid_engine as engine


class ShortSampleReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.font = ROOT / 'skills/code-band-bgm/assets/world-expansion-20261031/VCSL-World-Selected.sf2'
        if not cls.font.is_file():
            raise unittest.SkipTest('本批真实世界采样未安装，未运行短采样回归')
        meta = json.loads((cls.font.parent / '采样映射与转换.json').read_text())
        cls.preset = next(p for p in meta['预设'] if p['instrument_id'] == 'vcsl_balafon')
        cls.zone = min(cls.preset['zones'], key=lambda z: z['sf2_frames_per_channel'] / z['sample_rate_hz'])

    def document(self):
        return {'sample_rate_hz': 48000, 'total_frames': 144000,
                'parts': [{'part_id': 'balafon', 'bank': 0,
                           'program': self.preset['program'], 'gain_db': 0, 'pan': 0}],
                'events': [{'part_id': 'balafon', 'event_id': 'short_sample',
                            'onset_frame': 0, 'gate_frames': 96000,
                            'midi': self.zone['root_midi'],
                            'velocity': sum(self.zone['velocity_range']) // 2}],
                'mix': {'master_gain_db': 0, 'effects': []},
                'render': {'loop': {'enabled': False}}}

    def test_real_short_sample_long_gate_writes_wav_and_reports_inactive_voice(self):
        source = self.document()
        original = copy.deepcopy(source)
        first, diagnostics = engine.render_events(source, self.font)
        second, _ = engine.render_events(source, self.font)
        self.assertEqual(source, original)
        np.testing.assert_array_equal(first, second)
        self.assertTrue(np.isfinite(first).all())
        self.assertGreater(float(np.max(np.abs(first))), 1e-4)
        self.assertLess(self.zone['sf2_frames_per_channel'] / self.zone['sample_rate_hz'], 2)
        accepted = diagnostics['parts'][0]['noteoff_without_active_voice']
        self.assertEqual(len(accepted), 1)
        self.assertEqual(accepted[0]['midi'], self.zone['root_midi'])
        self.assertEqual(accepted[0]['frame'], 96000)
        self.assertEqual(accepted[0]['confirmed_on_voice_count'], 0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'short.wav'
            sf.write(path, first, 48000, subtype='PCM_24')
            info = sf.info(path)
            decoded, rate = sf.read(path, always_2d=True)
            self.assertEqual((rate, info.subtype, info.channels, info.frames),
                             (48000, 'PCM_24', 2, 144000))
            self.assertTrue(np.isfinite(decoded).all())

    def test_noteoff_error_with_still_on_matching_voice_is_not_swallowed(self):
        with engine._Synth(self.font, 48000, None) as synth:
            engine._ok(synth.lib.fluid_synth_program_select(synth.synth, 0, synth.sfont_id,
                                                           0, 0), '选择真实竖笛')
            engine._ok(synth.lib.fluid_synth_noteon(synth.synth, 0, 72, 90), '触发持续采样')
            synth.samples(256)
            with mock.patch.object(synth.lib, 'fluid_synth_noteoff', return_value=-1):
                with self.assertRaises(engine.EngineError) as raised:
                    synth.noteoff(0, 72)
            self.assertEqual(raised.exception.code, 'engine_call')

    def test_normal_successful_noteoff_does_not_report_inactive_voice(self):
        document = self.document()
        document['events'][0]['gate_frames'] = 3072
        _, evidence = engine.render_events(document, self.font)
        self.assertEqual(evidence['parts'][0]['noteoff_without_active_voice'], [])

    def test_invalid_channel_or_key_errors_are_not_accepted_as_ended_samples(self):
        with engine._Synth(self.font, 48000, None) as synth:
            for channel, midi in [(-1, 60), (999, 60), (0, -1), (0, 128)]:
                with self.subTest(channel=channel, midi=midi):
                    with self.assertRaises(engine.EngineError):
                        synth.noteoff(channel, midi)

    def test_inactive_target_is_accepted_with_other_pitch_still_on(self):
        with engine._Synth(self.font, 48000, None) as synth:
            engine._ok(synth.lib.fluid_synth_program_select(synth.synth, 0, synth.sfont_id,
                                                           0, 0), '选择真实竖笛')
            engine._ok(synth.lib.fluid_synth_noteon(synth.synth, 0, 72, 90), '触发另一音高')
            synth.samples(256)
            self.assertTrue(synth.noteoff(0, 73))
            self.assertFalse(synth.noteoff(0, 72))


if __name__ == '__main__':
    unittest.main()
