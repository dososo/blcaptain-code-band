"""帧级CC11的验证、调用顺序与真实旧行为回归，不把控制读回当听感。"""
import copy
import ctypes as ct
import hashlib
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'skills/code-band-bgm/scripts'))
import fluid_engine as engine


def document():
    return {'sample_rate_hz': 48000, 'total_frames': 96000,
            'parts': [{'part_id': 'strings', 'bank': 0, 'program': 48, 'gain_db': 0, 'pan': 0}],
            'events': [{'event_id': 'held', 'part_id': 'strings', 'onset_frame': 0,
                        'gate_frames': 72000, 'midi': 62, 'velocity': 84}],
            'mix': {'master_gain_db': 0, 'effects': []},
            'render': {'loop': {'enabled': False}}}


def control(frame=0, value=32, identifier='expression'):
    return {'event_id': identifier, 'part_id': 'strings', 'frame': frame,
            'controller': 11, 'value': value}


class FakeSynth:
    def __init__(self, readback_wrong=False, fail_cc=False):
        self.block, self.synth, self.frame, self.sfont_id = 64, object(), 0, 1
        self.calls, self.values = [], {}
        self.wrong, self.fail = readback_wrong, fail_cc
        self.lib = SimpleNamespace(fluid_synth_program_select=lambda *a: 0,
            fluid_synth_cc=self.cc, fluid_synth_get_cc=self.get,
            fluid_synth_noteon=self.noteon)

    def __enter__(self): return self
    def __exit__(self, *args): return False
    def samples(self, frames):
        self.frame += frames
        return np.full((frames, 2), .0001, dtype=np.float32)
    def cc(self, synth, channel, controller, value):
        self.calls.append((self.frame, 'cc', channel, controller, value))
        self.values[channel, controller] = value
        return -1 if controller == 11 and self.fail else 0
    def get(self, synth, channel, controller, value):
        ct.cast(value, ct.POINTER(ct.c_int))[0] = 0 if self.wrong else self.values[channel, controller]
        return 0
    def noteon(self, synth, channel, midi, velocity):
        self.calls.append((self.frame, 'on', channel, midi, velocity)); return 0
    def noteoff(self, channel, midi):
        self.calls.append((self.frame, 'off', channel, midi)); return False


class ControlValidationTests(unittest.TestCase):
    def test_bad_shape_reference_identity_types_grid_and_range_never_load_synth(self):
        invalid = [None, {}, '表达', [None], [dict(control(), extra=1)],
                   [dict(control(), part_id='missing')], [dict(control(), part_id=[])],
                   [dict(control(), event_id='')], [dict(control(), event_id='held')],
                   [dict(control(), event_id=[])], [dict(control(), controller=2)],
                   [dict(control(), controller=11.0)], [dict(control(), controller=True)],
                   [dict(control(), frame=True)], [dict(control(), frame=64.0)],
                   [dict(control(), frame=-64)], [dict(control(), frame=65)],
                   [dict(control(), frame=96000)], [dict(control(), value=-1)],
                   [dict(control(), value=128)], [dict(control(), value=True)],
                   [dict(control(), value=float('nan'))],
                   [control(), control(64, 96)],
                   [control(), control(0, 64, 'different_id')]]
        with patch.object(engine, '_Synth') as synth:
            for controls in invalid:
                d = document(); d['control_events'] = controls
                with self.subTest(controls=controls), self.assertRaises(engine.EngineError):
                    engine.render_events(d, Path('不能加载.sf3'))
            synth.assert_not_called()

    def test_nonempty_control_loop_rejected_but_empty_keeps_existing_loop(self):
        d = document(); d['render']['loop']['enabled'] = True; d['control_events'] = [control()]
        with patch.object(engine, '_Synth') as synth, self.assertRaises(engine.EngineError) as exc:
            engine.render_events(d, Path('不能加载.sf3'))
        self.assertEqual(exc.exception.code, 'unsupported_control_loop')
        synth.assert_not_called()
        d['control_events'] = []
        self.assertTrue(engine._validate(d))

    def test_same_frame_off_then_control_then_on_and_input_order_independent(self):
        d = document(); d['events'][0]['gate_frames'] = 24000
        d['events'].append({'event_id': 'answer', 'part_id': 'strings', 'onset_frame': 24000,
                            'gate_frames': 24000, 'midi': 65, 'velocity': 80})
        d['control_events'] = [control(24000, 64, 'later'), control(0, 32, 'first')]
        fake = FakeSynth()
        with patch.object(engine, '_Synth', return_value=fake), patch.object(engine, 'engine_info', return_value={}):
            _, diag = engine.render_events(d, Path('模拟.sf3'))
        at_boundary = [x[1] for x in fake.calls if x[0] == 24000]
        self.assertEqual(at_boundary, ['off', 'cc', 'on'])
        self.assertEqual([e['event_id'] for e in diag['parts'][0]['control_events']], ['first', 'later'])
        self.assertEqual(diag['parts'][0]['control_events'][1]['readback'], 64)

    def test_controller_targets_actual_drum_channel_not_melodic_channel(self):
        d = document(); d['parts'][0].update(bank=128, program=0)
        d['events'][0]['midi'] = 42; d['control_events'] = [control()]
        fake = FakeSynth()
        with patch.object(engine, '_Synth', return_value=fake), patch.object(engine, 'engine_info', return_value={}):
            _, diag = engine.render_events(d, Path('模拟.sf3'))
        self.assertIn((0, 'cc', 9, 11, 32), fake.calls)
        self.assertEqual(diag['parts'][0]['control_events'][0]['channel'], 9)

    def test_failed_call_or_wrong_readback_cannot_return_dry_fallback(self):
        d = document(); d['control_events'] = [control()]
        for fake, code in [(FakeSynth(fail_cc=True), 'engine_call'),
                           (FakeSynth(readback_wrong=True), 'control_readback_mismatch')]:
            with self.subTest(code=code), patch.object(engine, '_Synth', return_value=fake), self.assertRaises(engine.EngineError) as raised:
                engine.render_events(d, Path('模拟.sf3'))
            self.assertEqual(raised.exception.code, code)

    def test_zero_controller_value_still_requires_getter_to_write_readback(self):
        d = document(); d['control_events'] = [control(value=0)]
        fake = FakeSynth()
        fake.lib.fluid_synth_get_cc = lambda *args: 0
        with patch.object(engine, '_Synth', return_value=fake), self.assertRaises(engine.EngineError) as raised:
            engine.render_events(d, Path('模拟.sf3'))
        self.assertEqual(raised.exception.code, 'control_readback_mismatch')


class RealControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.font = ROOT / 'skills/code-band-bgm/assets/soundfonts/MuseScore_General.sf3'
        if not cls.font.is_file(): raise unittest.SkipTest('未安装真实音库')
        if hashlib.sha256(cls.font.read_bytes()).hexdigest() != '5b85b6c2c61d10b2b91cddd41efcce7b25cd31c8271d511c73afafbef20b6fa3':
            raise unittest.SkipTest('此真实回归限定固定MuseScore资产，不对替代文件宣称通过')
        cls.info = engine.engine_info()
        d = document(); cls.base, _ = engine.render_events(d, cls.font)
        empty = copy.deepcopy(d); empty['control_events'] = []
        cls.empty, _ = engine.render_events(empty, cls.font)
        d['control_events'] = [control(), control(24000, 96, 'strong'), control(48000, 32, 'weak_again')]
        cls.controlled, cls.diag = engine.render_events(d, cls.font)
        cls.repeated, _ = engine.render_events(d, cls.font)
        d['parts'][0]['pan'] = -1
        d['parts'].append({'part_id': 'untouched', 'bank': 0, 'program': 0, 'gain_db': 0, 'pan': 1})
        d['events'].append({'event_id': 'negative_note', 'part_id': 'untouched', 'onset_frame': 0,
                            'gate_frames': 72000, 'midi': 60, 'velocity': 84})
        cls.mixed_controlled, _ = engine.render_events(d, cls.font)
        del d['control_events']; cls.mixed_base, _ = engine.render_events(d, cls.font)

    def test_no_controls_and_empty_controls_match_prechange_real_hash(self):
        np.testing.assert_array_equal(self.base, self.empty)
        if sys.platform != 'darwin' or self.info['version'] != '2.6.1':
            self.skipTest('历史逐样SHA仅核本机macOS/FluidSynth2.6.1；其它平台未验')
        self.assertEqual(hashlib.sha256(self.base.astype('<f8').tobytes()).hexdigest(),
                         '3320b4931c2ab657f88e1031d55fd9e21f3a3b7b4bea110abafb64ab377ed321')

    def test_actual_held_note_weak_strong_weak_and_readbacks(self):
        deltas = []
        for start, end in [(.25, .35), (.65, .8), (1.2, 1.4)]:
            a, b = round(start*48000), round(end*48000)
            deltas.append(float(10*np.log10(np.mean(self.controlled[a:b]**2)/np.mean(self.base[a:b]**2))))
        self.assertGreater(deltas[1]-deltas[0], 6)
        self.assertGreater(deltas[1]-deltas[2], 6)
        self.assertLess(abs(deltas[0]-deltas[2]), 1.5)
        evidence = self.diag['parts'][0]
        self.assertEqual(evidence['event_count'], 1)
        self.assertEqual([e['readback'] for e in evidence['control_events']], [32, 96, 32])

    def test_real_repeat_and_other_real_part_unaffected(self):
        np.testing.assert_array_equal(self.controlled, self.repeated)
        np.testing.assert_array_equal(self.mixed_base[:, 1], self.mixed_controlled[:, 1])
        self.assertGreater(float(abs(self.mixed_base[:, 0]-self.mixed_controlled[:, 0]).max()), 1e-5)
        self.assertTrue(np.isfinite(self.controlled).all())
        self.assertGreater(float(abs(self.controlled).max()), 1e-5)


if __name__ == '__main__': unittest.main(verbosity=2)
