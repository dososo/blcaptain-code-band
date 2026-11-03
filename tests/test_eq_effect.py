"""单段等化器的真实频响、路由、失败封闭与旧音频兼容。"""
import copy
import hashlib
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'skills/code-band-bgm/scripts'))
from eq_effect import add_eq_capability, apply_eq, eq_parameters, probe_eq
from delay_effect import validate_effects
from fluid_engine import EngineError, render_events
from score_tools import ScoreError, compile_score, load, validate_frozen, validate_score
from soundfont_inventory import inventory


def effect():
    return {'effect_id': 'tame_upper', 'kind': 'eq', 'target_part_ids': ['piano'],
            'parameters': {'frequency_hz': 3500, 'q': .9, 'gain_db': -3.5}, 'required': True, 'notes': '保主奏单段修音'}


class EqTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.font = ROOT / 'skills/code-band-bgm/assets/soundfonts/MuseScore_General.sf3'
        cls.score = load(ROOT / 'tests/fixtures/音频闭环乐谱.json')
        cls.cap = load(ROOT / 'tests/fixtures/engine-capabilities-v1.json')
        cls.mapping = inventory(cls.font)
        cls.frozen = compile_score(cls.score, cls.cap, cls.mapping['source']['sha256'], cls.mapping)

    def test_real_tone_response_frames_stereo_and_repeat(self):
        result = probe_eq()
        self.assertAlmostEqual(result['center_gain_db'], -3.5, delta=.02)
        self.assertLess(abs(result['low_gain_db']), .03)
        self.assertTrue(result['float_repeat_equal'])

    def test_numeric_whitelist_and_nyquist(self):
        invalid = [{'frequency_hz': '3500;anything'}, {'q': True}, {'gain_db': float('nan')},
                   {'gain_db': 1}, {'q': 0}, {'frequency_hz': 24000}, {'text': 'filter'},
                   {'q': float('inf')}]
        for update in invalid:
            candidate = effect(); candidate['parameters'].update(update)
            with self.subTest(update=update), self.assertRaises(ValueError):
                eq_parameters(candidate, 48000)
        with self.assertRaises(ValueError):
            eq_parameters({'parameters': {}}, 48000)
        with self.assertRaises(ValueError):
            eq_parameters(effect(), 4000)

    def test_order_targets_unique_loop_and_effect_count(self):
        eq = effect()
        validate_effects([eq], {'piano'})
        for effects, ids, loop in [([eq] * 2, {'piano'}, False), ([eq], {'piano'}, True),
                                   ([eq], {'other'}, False),
                                   ([{'kind': 'delay', 'effect_id': 'd', 'parameters': {}}, eq], {'piano'}, False)]:
            with self.subTest(effects=effects, ids=ids, loop=loop), self.assertRaises(ValueError):
                validate_effects(effects, ids, loop)
        for targets in [[], ['piano', 'piano']]:
            candidate = effect(); candidate['target_part_ids'] = targets
            with self.assertRaises(ValueError):
                validate_effects([candidate], {'piano'})

    def test_missing_capability_and_frozen_bypass_rejected(self):
        score = copy.deepcopy(self.score); score['mix']['effects'] = [effect()]
        with self.assertRaises(ScoreError):
            compile_score(score, self.cap, self.mapping['source']['sha256'], self.mapping)
        cap = add_eq_capability({'capabilities': copy.deepcopy(self.cap)})['capabilities']
        frozen = compile_score(score, cap, self.mapping['source']['sha256'], self.mapping)
        self.assertIs(validate_frozen(frozen, score, cap, self.mapping['source']['sha256']), frozen)
        with self.assertRaises(ScoreError):
            validate_frozen(frozen, score, self.cap, self.mapping['source']['sha256'])
        no_probe = copy.deepcopy(cap)
        no_probe['mixing']['eq']['evidence_ids'] = self.cap['mixing']['reverb']['evidence_ids']
        with self.assertRaisesRegex(ScoreError, '探测证据'):
            compile_score(score, no_probe, self.mapping['source']['sha256'], self.mapping)
        with self.assertRaisesRegex(ScoreError, '探测证据'):
            validate_frozen(frozen, score, no_probe, self.mapping['source']['sha256'])

    def test_schema_rejects_non_numeric_or_extra_eq_fields(self):
        for update in [{'gain_db': True}, {'q': '0.9'}, {'filter': '任意文本'}]:
            score = copy.deepcopy(self.score); candidate = effect()
            candidate['parameters'].update(update)
            score['mix']['effects'] = [candidate]
            with self.subTest(update=update), self.assertRaises(ScoreError):
                validate_score(score)

    def test_public_validate_rejects_eq_order_and_nyquist(self):
        score = copy.deepcopy(self.score)
        score['mix']['effects'] = [effect(), {'effect_id': 'r', 'kind': 'reverb',
            'target_part_ids': ['piano'], 'parameters': {}, 'required': True, 'notes': '次序反证'}]
        with self.assertRaisesRegex(ScoreError, '混响'):
            validate_score(score)
        score['mix']['effects'] = [effect()]
        score['mix']['effects'][0]['parameters']['frequency_hz'] = 17000
        score['render']['sample_rate_hz'] = 32000
        with self.assertRaisesRegex(ScoreError, 'Nyquist'):
            validate_score(score)

    def test_processing_failure_and_corrupt_output_do_not_return_dry(self):
        signal = np.ones((4800, 2)) * .1
        with patch('eq_effect.subprocess.run', side_effect=FileNotFoundError('缺失')):
            with self.assertRaises(ValueError):
                apply_eq(signal, 48000, effect())
        from types import SimpleNamespace
        for output in [b'', np.full((4800, 2), np.nan).astype('<f8').tobytes()]:
            with patch('eq_effect.subprocess.run', return_value=SimpleNamespace(stdout=output, stderr=b'')):
                with self.assertRaises(ValueError):
                    apply_eq(signal, 48000, effect())

    def test_legacy_audio_exact_and_only_selected_part_changes(self):
        original, _ = render_events(self.frozen, self.font)
        self.assertEqual(hashlib.sha256(original.tobytes()).hexdigest(),
                         '31a8c882f87512b6f7aa87f558c6490c2aaed12642ddab193a1c44b3a12816d3')
        document = {'sample_rate_hz': 48000, 'total_frames': 24000,
            'parts': [{'part_id': 'piano', 'bank': 0, 'program': 48, 'gain_db': -3, 'pan': -1},
                      {'part_id': 'untouched', 'bank': 0, 'program': 0, 'gain_db': -3, 'pan': 1}],
            'events': [{'event_id': name, 'part_id': name, 'onset_frame': 0, 'gate_frames': 12032, 'midi': 74, 'velocity': 80}
                       for name in ['piano', 'untouched']],
            'mix': {'master_gain_db': -3, 'effects': []}, 'render': {'loop': {'enabled': False}}}
        before, _ = render_events(document, self.font)
        changed = copy.deepcopy(document); changed['mix']['effects'] = [effect()]
        after, details = render_events(changed, self.font)
        repeat, _ = render_events(changed, self.font)
        self.assertTrue(np.array_equal(before[:, 1], after[:, 1]))
        self.assertFalse(np.array_equal(before[:, 0], after[:, 0]))
        self.assertTrue(np.array_equal(after, repeat))
        self.assertEqual(changed['events'], document['events'])
        self.assertEqual(len(details['parts'][0]['eq_effects']), 1)
        self.assertNotIn('eq_effects', details['parts'][1])
        with patch('eq_effect.apply_eq', side_effect=ValueError('实际处理失败')):
            with self.assertRaises(EngineError) as caught:
                render_events(changed, self.font)
        self.assertEqual(caught.exception.code, 'eq_processing_failed')


if __name__ == '__main__':
    unittest.main()
