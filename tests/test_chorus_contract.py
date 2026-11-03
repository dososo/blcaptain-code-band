"""原生合唱的谱、能力和冻结事件门禁；音乐听感另行验收。"""
import copy
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'skills/code-band-bgm/scripts'))
from delay_effect import validate_effects
from score_tools import (ScoreError, compile_score, load, validate_capabilities,
                         validate_frozen, validate_score)
from soundfont_inventory import inventory


def chorus():
    return {'effect_id': 'keyboard_width', 'kind': 'chorus', 'target_part_ids': ['piano'],
            'parameters': {}, 'required': True, 'notes': '真实原生效果候选，不认定原曲效果器'}


class ChorusContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.score = load(ROOT / 'tests/fixtures/音频闭环乐谱.json')
        cls.base = load(ROOT / 'tests/fixtures/engine-capabilities-v1.json')
        cls.mapping = inventory(ROOT / 'skills/code-band-bgm/assets/soundfonts/MuseScore_General.sf3')
        cls.sha = cls.mapping['source']['sha256']
        # 契约单测的明确能力夹具；不冒称由此完成doctor真实探测。
        cls.cap = copy.deepcopy(cls.base)
        cls.cap['mixing']['chorus'] = {'state': 'supported', 'evidence_ids': ['runtime_probe'],
                                       'limitations': ['契约夹具，真实探测在test_chorus_effect']}

    def test_score_compile_and_frozen_preserve_notes_and_require_current_capability(self):
        validate_capabilities(self.cap)
        wet = copy.deepcopy(self.score)
        wet['mix']['effects'] = [chorus()]
        validate_score(wet)
        compiled = compile_score(wet, self.cap, self.sha, self.mapping)
        dry = compile_score(self.score, self.base, self.sha, self.mapping)
        self.assertEqual(compiled['events'], dry['events'])
        self.assertEqual(compiled['mix']['effects'], [chorus()])
        self.assertIs(validate_frozen(compiled, wet, self.cap, self.sha), compiled)
        with self.assertRaises(ScoreError):
            compile_score(wet, self.base, self.sha, self.mapping)
        with self.assertRaises(ScoreError):
            validate_frozen(compiled, wet, self.base, self.sha)

    def test_validation_rejects_bad_parameters_before_resource_compilation(self):
        for parameters in [{'speed_hz': 0}, {'voices': True}, {'send': 128},
                           {'level': float('nan')}, {'waveform': 'triangle'}, {'unknown': 1}]:
            score = copy.deepcopy(self.score)
            effect = chorus()
            effect['parameters'] = parameters
            score['mix']['effects'] = [effect]
            with self.subTest(parameters=parameters), self.assertRaises(ScoreError):
                validate_score(score)

    def test_native_parallel_prefix_and_five_effect_limit(self):
        reverb = {'kind': 'reverb', 'target_part_ids': ['piano']}
        delay = {'effect_id': 'echo', 'kind': 'delay', 'target_part_ids': ['piano'],
                 'parameters': {'delay_ms': 100, 'decay': .1, 'repeats': 1}}
        duck = {'kind': 'duck', 'target_part_ids': ['piano'],
                'parameters': {'trigger_part_id': 'kick'}}
        ids = {'piano', 'kick'}
        for native in [[reverb, chorus()], [chorus(), reverb]]:
            validate_effects(native + [delay, delay, duck], ids)
        for effects in [[delay, chorus()], [delay, reverb], [chorus(), chorus()],
                        [reverb, chorus(), delay, delay, delay, duck]]:
            with self.subTest(effects=effects), self.assertRaises(ValueError):
                validate_effects(effects, ids)
        with self.assertRaisesRegex(ValueError, '循环'):
            validate_effects([chorus()], ids, loop=True)


if __name__ == '__main__':
    unittest.main()
