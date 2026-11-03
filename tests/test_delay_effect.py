"""有限延迟的时间、路由、错误与旧音频兼容性；不代替听感。"""
import copy
import hashlib
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'skills/code-band-bgm/scripts'))
from delay_effect import add_delay_capability, apply_delay, delay_parameters, probe_delay, validate_effects
from fluid_engine import EngineError, render_events
from score_tools import ScoreError, compile_score, load, validate_frozen
from soundfont_inventory import inventory


class DelayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.font = ROOT / 'skills/code-band-bgm/assets/soundfonts/MuseScore_General.sf3'
        cls.score = load(ROOT / 'tests/fixtures/音频闭环乐谱.json')
        cls.cap = load(ROOT / 'tests/fixtures/engine-capabilities-v1.json')
        cls.mapping = inventory(cls.font)
        cls.compiled = compile_score(cls.score, cls.cap, cls.mapping['source']['sha256'], cls.mapping)
        cls.effect = {'effect_id': 'test_echo', 'kind': 'delay', 'target_part_ids': ['piano'],
            'parameters': {'delay_ms': 250, 'decay': .5, 'repeats': 2}, 'required': True, 'notes': '隔离测试'}

    def test_actual_impulse_positions_and_no_added_duration(self):
        self.assertEqual(probe_delay()['maximum_error'], 0.)

    def test_user_text_nan_unknown_parameters_and_loop_are_rejected(self):
        for parameters in [{'delay_ms': '1;anything'}, {'decay': float('nan')}, {'feedback': .4}, {'repeats': True}, {'delay_ms': 2000, 'repeats': 4}]:
            with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                delay_parameters({'parameters': parameters})
        with self.assertRaises(ValueError):
            validate_effects([self.effect], {'piano'}, loop=True)
        with self.assertRaises(ValueError):
            validate_effects([self.effect, {'kind': 'reverb'}], {'piano'})
        with self.assertRaises(ValueError):
            validate_effects([self.effect] * 3, {'piano'})

    def test_missing_capability_does_not_silently_use_dry_audio(self):
        score = copy.deepcopy(self.score)
        score['mix']['effects'] = [self.effect]
        with self.assertRaises(ScoreError):
            compile_score(score, self.cap, self.mapping['source']['sha256'], self.mapping)

    def test_legacy_no_effect_audio_is_sample_exact(self):
        audio, _ = render_events(self.compiled, self.font)
        self.assertEqual(hashlib.sha256(audio.tobytes()).hexdigest(),
            '31a8c882f87512b6f7aa87f558c6490c2aaed12642ddab193a1c44b3a12816d3')

    def test_frozen_events_cannot_bypass_current_effect_capability(self):
        score = copy.deepcopy(self.score)
        score['mix']['effects'] = [self.effect]
        cap = add_delay_capability({'capabilities': copy.deepcopy(self.cap)})['capabilities']
        frozen = compile_score(score, cap, self.mapping['source']['sha256'], self.mapping)
        self.assertIs(validate_frozen(frozen, score, cap, self.mapping['source']['sha256']), frozen)
        with self.assertRaises(ScoreError):
            validate_frozen(frozen, score, self.cap, self.mapping['source']['sha256'])

    def test_selected_part_routing_and_fixed_event_repeat(self):
        single, _ = render_events(self.compiled, self.font)
        changed = copy.deepcopy(self.compiled)
        changed['mix']['effects'] = [self.effect]
        expected, _ = apply_delay(single, 48000, self.effect)
        first, diagnostics = render_events(changed, self.font)
        second, _ = render_events(changed, self.font)
        self.assertTrue(np.allclose(first, expected, rtol=0, atol=1e-12))
        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(len(diagnostics['parts'][0]['delay_effects']), 1)
        self.assertEqual(changed['events'], self.compiled['events'])
        changed['mix']['effects'][0]['target_part_ids'] = ['unknown']
        with self.assertRaises(EngineError):
            render_events(changed, self.font)


if __name__ == '__main__':
    unittest.main()
