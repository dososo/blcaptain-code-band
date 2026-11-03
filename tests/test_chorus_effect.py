"""原生chorus的参数、失败、路由、确定性与真实空间证据；不替代听感。"""
import copy
import ctypes as ct
import importlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'skills/code-band-bgm/scripts'))
from fluid_engine import EngineError, render_events


def effect():
    return {'effect_id': 'native_chorus', 'kind': 'chorus',
            'target_part_ids': ['target'], 'parameters': {}}


def document(two_parts=False):
    parts = [{'part_id': 'target', 'bank': 0, 'program': 5, 'gain_db': 0,
              'pan': -1 if two_parts else 0}]
    events = [{'event_id': 'g4', 'part_id': 'target', 'onset_frame': 0,
               'gate_frames': 28800, 'midi': 67, 'velocity': 84}]
    if two_parts:
        parts.append({'part_id': 'untouched', 'bank': 0, 'program': 0, 'gain_db': 0, 'pan': 1})
        events.append({'event_id': 'c5', 'part_id': 'untouched', 'onset_frame': 0,
                       'gate_frames': 28800, 'midi': 72, 'velocity': 76})
    return {'sample_rate_hz': 48000, 'total_frames': 48000, 'parts': parts,
            'events': events, 'mix': {'master_gain_db': 0, 'effects': []},
            'render': {'loop': {'enabled': False}}}


class ChorusParameterTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('chorus_effect'), '缺少原生chorus模块')
        return importlib.import_module('chorus_effect')

    def test_defaults_and_explicit_numeric_type_boundaries(self):
        mod = self.module()
        self.assertEqual(mod.chorus_parameters(effect()), {
            'voices': 3, 'level': .6, 'speed_hz': .2, 'depth_ms': 4.25,
            'send': 127, 'waveform': 'sine'})
        invalid = [{'voices': True}, {'voices': 3.0}, {'voices': 0}, {'voices': 9},
                   {'level': True}, {'level': float('nan')}, {'level': 1.01},
                   {'speed_hz': .099}, {'speed_hz': 5.01}, {'speed_hz': '0.2'},
                   {'depth_ms': -1}, {'depth_ms': 20.01}, {'depth_ms': float('inf')},
                   {'send': False}, {'send': 127.0}, {'send': 128},
                   {'waveform': 'triangle'}, {'waveform': True}, {'unknown': 1}]
        for params in invalid:
            candidate = effect(); candidate['parameters'] = params
            with self.subTest(params=params), self.assertRaises(ValueError):
                mod.chorus_parameters(candidate)
        for params in [None, [], '任意文字']:
            with self.subTest(params=params), self.assertRaises(ValueError):
                mod.chorus_parameters({'parameters': params})

    def test_targets_duplicates_order_and_loop_are_rejected(self):
        mod = self.module()
        self.assertIsNone(mod.validate_chorus_effects([], {'target'}))
        self.assertEqual(mod.validate_chorus_effects([effect()], {'target'}), effect())
        for targets in [['unknown'], ['target', 'target'], [True], [[]], 'target']:
            candidate = effect(); candidate['target_part_ids'] = targets
            with self.subTest(targets=targets), self.assertRaises(ValueError):
                mod.validate_chorus_effects([candidate], {'target'})
        for effects in [[effect(), effect()], [{'kind': 'delay'}, effect()],
                        [{'kind': 'duck'}, effect()]]:
            with self.subTest(effects=effects), self.assertRaises(ValueError):
                mod.validate_chorus_effects(effects, {'target'})
        with self.assertRaisesRegex(ValueError, '循环'):
            mod.validate_chorus_effects([effect()], {'target'}, loop=True)

    def test_missing_native_api_is_error_instead_of_dry_fallback(self):
        mod = self.module()
        with self.assertRaisesRegex(ValueError, '接口'):
            mod.apply_chorus(SimpleNamespace(lib=SimpleNamespace()), mod.chorus_parameters(effect()))

    def test_drum_channel_cc93_does_not_write_melodic_channel(self):
        mod = self.module()
        from fluid_engine import _library
        lib = _library()
        settings = lib.new_fluid_settings()
        synth = lib.new_fluid_synth(settings)
        self.assertTrue(synth)
        try:
            try:
                diagnostic = mod.apply_chorus(SimpleNamespace(lib=lib, synth=synth),
                                              mod.chorus_parameters(effect()), channel=9)
            except TypeError as exc:
                self.fail('原生chorus必须能明确使用打击乐channel9：' + str(exc))
            value = ct.c_int()
            self.assertGreaterEqual(lib.fluid_synth_get_cc(synth, 9, 93, ct.byref(value)), 0)
            self.assertEqual(value.value, 127)
            self.assertGreaterEqual(lib.fluid_synth_get_cc(synth, 0, 93, ct.byref(value)), 0)
            self.assertEqual(value.value, 0)
            self.assertEqual(diagnostic['channel'], 9)
        finally:
            lib.delete_fluid_synth(synth)
            lib.delete_fluid_settings(settings)

    def test_engine_rejects_invalid_and_loop_before_synth(self):
        invalid = []
        for params in [{'send': True}, {'voices': 9}, {'level': float('nan')}]:
            doc = document(); e = effect(); e['parameters'] = params
            doc['mix']['effects'] = [e]; invalid.append(doc)
        loop = document(); loop['mix']['effects'] = [effect()]
        loop['render']['loop']['enabled'] = True
        with patch('fluid_engine._Synth') as synth:
            for doc in invalid:
                with self.subTest(doc=doc), self.assertRaises(EngineError):
                    render_events(doc, Path('未打开.sf3'))
            with self.assertRaisesRegex(EngineError, '循环'):
                render_events(loop, Path('未打开.sf3'))
            synth.assert_not_called()

    def test_engine_rejects_non_string_target_with_structured_error(self):
        doc = document(); candidate = effect(); candidate['target_part_ids'] = [[]]
        doc['mix']['effects'] = [candidate]
        with patch('fluid_engine._Synth') as synth:
            try:
                with self.assertRaises(EngineError):
                    render_events(doc, Path('未打开.sf3'))
            except TypeError as exc:
                self.fail('非法chorus目标应给EngineError而不是类型崩溃：' + str(exc))
            synth.assert_not_called()

    def test_native_api_failure_and_cc_readback_mismatch_are_rejected(self):
        mod = self.module()
        from fluid_engine import _library
        lib = _library(); settings = lib.new_fluid_settings(); synth = lib.new_fluid_synth(settings)
        self.assertTrue(synth)
        wrapper = SimpleNamespace(lib=lib, synth=synth)
        try:
            with patch.object(lib, 'fluid_synth_set_chorus_group_nr', return_value=-1):
                with self.assertRaisesRegex(ValueError, '调用失败'):
                    mod.apply_chorus(wrapper, mod.chorus_parameters(effect()))
            # getter返回成功却未写实际值，模拟读回不一致，不接受假成功。
            with patch.object(lib, 'fluid_synth_get_cc', return_value=0):
                with self.assertRaisesRegex(ValueError, '读回不一致'):
                    mod.apply_chorus(wrapper, mod.chorus_parameters(effect()))
        finally:
            lib.delete_fluid_synth(synth); lib.delete_fluid_settings(settings)


class ChorusRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.font = ROOT / 'skills/code-band-bgm/assets/soundfonts/MuseScore_General.sf3'

    def render(self, doc):
        try:
            return render_events(doc, self.font)
        except EngineError as exc:
            self.fail('应能演奏合法原生chorus，实际被拒绝：' + str(exc))

    def test_real_same_resource_dry_wet_stereo_and_repeat(self):
        dry, _ = self.render(document())
        doc = document(); doc['mix']['effects'] = [effect()]
        wet, diagnostic = self.render(doc)
        repeated, _ = self.render(doc)
        self.assertEqual(wet.shape, (48000, 2))
        self.assertTrue(np.isfinite(wet).all())
        self.assertGreater(np.max(np.abs(wet-dry)), 1e-4)
        self.assertTrue(np.array_equal(wet, repeated))
        def side_fraction(x):
            x = x - np.mean(x, axis=0)
            m, s = (x[:, 0]+x[:, 1])/2, (x[:, 0]-x[:, 1])/2
            return np.sum(s*s)/(np.sum(m*m)+np.sum(s*s))
        self.assertGreater(side_fraction(wet), side_fraction(dry) + .05)
        d = diagnostic['parts'][0]['chorus']
        self.assertEqual(d['parameters']['voices'], 3)
        self.assertEqual(d['cc93_readback'], 127)
        self.assertTrue(all(x['return_code'] >= 0 for x in d['api_calls']))
        self.assertIn('基底', d['send_semantics'])

    def test_only_target_changes_and_empty_target_means_all(self):
        base = document(two_parts=True)
        dry, _ = self.render(base)
        wet_doc = copy.deepcopy(base); wet_doc['mix']['effects'] = [effect()]
        wet, diag = self.render(wet_doc)
        self.assertTrue(np.array_equal(wet[:, 1], dry[:, 1]))
        self.assertGreater(np.max(abs(wet[:, 0]-dry[:, 0])), 1e-4)
        parts = {p['part_id']: p for p in diag['parts']}
        self.assertIn('chorus', parts['target'])
        self.assertNotIn('chorus', parts['untouched'])
        wet_doc['mix']['effects'][0]['target_part_ids'] = []
        both, both_diag = self.render(wet_doc)
        self.assertGreater(np.max(abs(both[:, 1]-dry[:, 1])), 1e-4)
        self.assertTrue(all('chorus' in p for p in both_diag['parts']))

    def test_no_effect_matches_frozen_legacy_probe(self):
        # 可移植地重现旧干声调用顺序，不启用原生chorus；不依赖研究目录。
        from fluid_engine import _Synth
        with _Synth(self.font, 48000, None) as synth:
            self.assertGreaterEqual(synth.lib.fluid_synth_program_select(synth.synth,0,synth.sfont_id,0,5),0)
            self.assertGreaterEqual(synth.lib.fluid_synth_cc(synth.synth,0,91,0),0)
            self.assertGreaterEqual(synth.lib.fluid_synth_noteon(synth.synth,0,67,84),0)
            first = synth.samples(28800)
            synth.noteoff(0,67)
            tail = synth.samples(19200)
        expected = np.concatenate((first,tail)).astype('float64')
        actual, diag = self.render(document())
        self.assertTrue(np.array_equal(actual, expected))
        self.assertNotIn('chorus',diag['parts'][0])

    def test_reverb_parallel_target_is_not_mistaken_for_first_chorus_target(self):
        doc = document(two_parts=True)
        doc['mix']['effects'] = [effect(), {'kind':'reverb', 'effect_id':'room',
            'target_part_ids':['untouched'], 'parameters':{}}]
        _, diag = self.render(doc)
        parts = {p['part_id']:p for p in diag['parts']}
        self.assertFalse(parts['target']['includes_shared_settings_reverb'])
        self.assertTrue(parts['untouched']['includes_shared_settings_reverb'])
        self.assertIn('chorus',parts['target'])
        self.assertNotIn('chorus',parts['untouched'])

    def test_capability_uses_actual_representative_probe(self):
        self.assertIsNotNone(importlib.util.find_spec('chorus_effect'), '缺少原生chorus模块')
        mod=importlib.import_module('chorus_effect')
        result={'capabilities':{'mixing':{},'evidence':[]}}
        self.assertIs(mod.add_chorus_capability(result,self.font),result)
        self.assertEqual(result['capabilities']['mixing']['chorus']['state'],'supported')
        self.assertEqual(result['chorus_probe']['status'],'通过')
        self.assertTrue(result['chorus_probe']['same_event_repeat_sample_exact'])
        self.assertGreater(result['chorus_probe']['wet_side_fraction'],result['chorus_probe']['dry_side_fraction']+.05)


if __name__ == '__main__':
    unittest.main()
