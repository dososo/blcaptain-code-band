"""显式触发包络的数值边界、门禁及真实双声部路由；不替代听感。"""
import copy
import hashlib
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'skills/code-band-bgm/scripts'))
from ducking_effect import add_duck_capability, apply_duck, duck_parameters
from delay_effect import validate_effects
from fluid_engine import EngineError, render_events
from score_tools import ScoreError, compile_score, load, validate_frozen
from soundfont_inventory import inventory


def effect():
    return {'effect_id': 'test_duck', 'kind': 'duck', 'target_part_ids': ['target'],
        'parameters': {'trigger_part_id': 'trigger', 'attack_ms': 10, 'hold_ms': 20,
                       'release_ms': 30, 'depth_db': -12},
        'required': True, 'notes': '只验证显式触发与目标路由'}


class DuckEnvelopeTests(unittest.TestCase):
    def test_legacy_single_output_and_complete_diagnostic_remain_exact(self):
        output, diagnostic = apply_duck(np.tile([-.2, .1], (1600, 1)), 8000, effect(),
            [{'part_id': 'trigger', 'onset_frame': 400}, {'part_id': 'trigger', 'onset_frame': 560}])
        self.assertEqual(hashlib.sha256(output.astype('<f8').tobytes()).hexdigest(),
                         'a91f5e25c3e55d383b5f661f3b3587c71f240770e8c78bf2fbf3f9b15917104d')
        self.assertEqual(diagnostic, {
            'backend': 'explicit_event_gain_envelope',
            'parameters': {'attack_ms': 10, 'hold_ms': 20, 'release_ms': 30, 'depth_db': -12, 'trigger_part_id': 'trigger'},
            'trigger_count': 2, 'attack_frames': 80, 'hold_frames': 160, 'release_frames': 240,
            'minimum_gain': 0.251188643150958,
            'envelope_sha256': '1a7686f1beeaf792990d96d0ac28c236dfb4172fec4073fb83ee54b32d8f9e5a',
            'overlap_rule': '取增益包络最小值，不相乘加深；线性幅度attack/hold/release',
            'tail_policy': '恢复包络在既定总帧数内处理，超界不延长文件；未验证循环',
            'scope': '自有制作手段；触发来自已编译音符，不检测原音频、不声称已识别原sidechain'})

    def test_plural_union_preserves_all_26_onsets_and_never_stacks_depth(self):
        wave = np.tile([-.2, .1], (1600, 1))
        frames = list(range(128, 960, 32))
        events = [{'part_id': 'body' if i in (3, 9, 15, 21) else 'trigger', 'onset_frame': frame}
                  for i, frame in enumerate(frames)]
        events.append({'part_id': 'body', 'onset_frame': frames[0]})
        plural = effect()
        plural['parameters'].pop('trigger_part_id')
        plural['parameters']['trigger_part_ids'] = ['trigger', 'body']
        try:
            output, diagnostic = apply_duck(wave, 8000, plural, events)
        except ValueError as exc:
            self.fail('多来源并集尚未实现：' + str(exc))
        merged = [{**event, 'part_id': 'trigger'} for event in events]
        expected, previous = apply_duck(wave, 8000, effect(), merged)
        self.assertTrue(np.array_equal(output, expected))
        self.assertEqual(diagnostic['trigger_count'], 26)
        self.assertEqual(diagnostic['envelope_sha256'], previous['envelope_sha256'])
        self.assertEqual(diagnostic['trigger_source_event_counts'], {'trigger': 22, 'body': 5})
        self.assertEqual(diagnostic['trigger_source_onset_counts'], {'trigger': 22, 'body': 5})
        self.assertGreaterEqual(float(output[:, 1].min()), .1 * 10 ** (-12 / 20))
        with self.assertRaisesRegex(ValueError, '真实触发'):
            apply_duck(wave, 8000, plural, [event for event in events if event['part_id'] == 'trigger'])

    def test_plural_ids_are_explicit_unique_existing_and_disjoint_from_targets(self):
        for ids in [[], ['trigger'] * 2, [''], [' '], ['trigger', 'missing'], ['trigger', 'target'],
                    'trigger', ['trigger', True], list('abcdefghi')]:
            candidate = effect()
            candidate['parameters'].pop('trigger_part_id')
            candidate['parameters']['trigger_part_ids'] = ids
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                duck_parameters(candidate, {'trigger', 'target'})
        candidate = effect()
        candidate['parameters']['trigger_part_ids'] = ['trigger']
        with self.assertRaises(ValueError):
            duck_parameters(candidate, {'trigger', 'target'})

    def test_actual_trigger_linear_gain_hold_recovery_and_determinism(self):
        wave = np.tile([-.2, .1], (1600, 1))
        events = [{'part_id': 'target', 'onset_frame': 0},
                  {'part_id': 'trigger', 'onset_frame': 400}]
        original = wave.copy()
        output, diagnostic = apply_duck(wave, 8000, effect(), events)
        repeated, repeated_diagnostic = apply_duck(wave, 8000, effect(), events)
        floor = 10 ** (-12 / 20)
        self.assertTrue(np.array_equal(wave, original))
        self.assertTrue(np.array_equal(output[:400], wave[:400]))
        np.testing.assert_allclose(output[440], wave[440] * ((1 + floor) / 2), rtol=0, atol=1e-15)
        np.testing.assert_allclose(output[480:640], wave[480:640] * floor, rtol=0, atol=1e-15)
        np.testing.assert_allclose(output[760], wave[760] * ((1 + floor) / 2), rtol=0, atol=1e-15)
        self.assertTrue(np.array_equal(output[880:], wave[880:]))
        self.assertEqual(output.shape, wave.shape)
        self.assertTrue(np.isfinite(output).all())
        self.assertTrue(np.array_equal(output, repeated))
        self.assertEqual(diagnostic['envelope_sha256'], repeated_diagnostic['envelope_sha256'])
        self.assertEqual(diagnostic['trigger_count'], 1)
        self.assertAlmostEqual(diagnostic['minimum_gain'], floor)

    def test_overlapping_triggers_take_minimum_and_never_multiply_depth(self):
        wave = np.ones((1600, 2))
        first = [{'part_id': 'trigger', 'onset_frame': 400}]
        second = [{'part_id': 'trigger', 'onset_frame': 560}]
        a, _ = apply_duck(wave, 8000, effect(), first)
        b, _ = apply_duck(wave, 8000, effect(), second)
        combined, diagnostic = apply_duck(wave, 8000, effect(), first + second + first)
        self.assertTrue(np.array_equal(combined, np.minimum(a, b)))
        self.assertGreaterEqual(float(combined.min()), 10 ** (-12 / 20))
        self.assertEqual(diagnostic['trigger_count'], 2)
        self.assertTrue(np.array_equal(combined[1040:], wave[1040:]))

    def test_tail_truncation_preserves_frame_count_and_zero_depth_is_identity(self):
        wave = np.tile([.1, -.2], (800, 1))
        events = [{'part_id': 'trigger', 'onset_frame': 768}]
        output, _ = apply_duck(wave, 8000, effect(), events)
        self.assertEqual(output.shape, wave.shape)
        self.assertTrue(np.array_equal(output[:768], wave[:768]))
        self.assertLess(abs(output[-1, 0]), abs(wave[-1, 0]))
        neutral = effect()
        neutral['parameters']['depth_db'] = 0
        unchanged, _ = apply_duck(wave, 8000, neutral, events)
        self.assertTrue(np.array_equal(unchanged, wave))

    def test_missing_actual_trigger_and_out_of_range_times_are_rejected(self):
        for events in [[], [{'part_id': 'target', 'onset_frame': 100}],
                       [{'part_id': 'trigger', 'onset_frame': -1}],
                       [{'part_id': 'trigger', 'onset_frame': 800}],
                       [{'part_id': 'trigger', 'onset_frame': 10.5}]]:
            with self.subTest(events=events), self.assertRaises(ValueError):
                apply_duck(np.ones((800, 2)), 8000, effect(), events)

    def test_invalid_parameters_targets_and_trigger_ids_are_rejected(self):
        invalid = [{'attack_ms': .49}, {'attack_ms': 50.1}, {'hold_ms': -1}, {'hold_ms': 251},
            {'release_ms': 0}, {'release_ms': 2001}, {'depth_db': -24.1}, {'depth_db': .1},
            {'attack_ms': True}, {'depth_db': float('nan')}, {'hold_ms': float('inf')},
            {'release_ms': '1;anything'}, {'feedback': .1}, {'trigger_part_id': ''},
            {'trigger_part_id': ['trigger']}, {'trigger_part_id': 'missing'}]
        for parameters in invalid:
            candidate = effect()
            candidate['parameters'].update(parameters)
            with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                duck_parameters(candidate, {'target', 'trigger'})
        for targets in [[], ['trigger'], ['missing'], 'target']:
            candidate = effect()
            candidate['target_part_ids'] = targets
            with self.subTest(targets=targets), self.assertRaises(ValueError):
                duck_parameters(candidate, {'target', 'trigger'})
        for parameters in [None, [], '任意文字']:
            candidate = effect()
            candidate['parameters'] = parameters
            with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                duck_parameters(candidate)

    def test_invalid_audio_or_rate_is_rejected(self):
        trigger = [{'part_id': 'trigger', 'onset_frame': 10}]
        for wave, rate in [(np.zeros(800), 8000), (np.zeros((800, 1)), 8000),
                           (np.full((800, 2), np.nan), 8000), (np.zeros((800, 2)), 7999),
                           (np.zeros((800, 2)), True)]:
            with self.subTest(shape=wave.shape, rate=rate), self.assertRaises(ValueError):
                apply_duck(wave, rate, effect(), trigger)

    def test_effect_must_be_single_last_item_and_loop_is_rejected(self):
        echo = {'effect_id': 'echo', 'kind': 'delay', 'target_part_ids': ['target'],
                'parameters': {'delay_ms': 100, 'decay': .2, 'repeats': 1}}
        validate_effects([echo, effect()], {'target', 'trigger'})
        for effects in [[effect(), echo], [effect(), effect()]]:
            with self.subTest(effects=effects), self.assertRaisesRegex(ValueError, '末项'):
                validate_effects(effects, {'target', 'trigger'})
        with self.assertRaisesRegex(ValueError, '循环'):
            validate_effects([effect()], {'target', 'trigger'}, loop=True)

    def test_capability_contains_real_envelope_probe_without_duplicate_evidence(self):
        result = {'capabilities': {'mixing': {}, 'evidence': []}}
        self.assertIs(add_duck_capability(result), result)
        add_duck_capability(result)
        self.assertEqual(result['duck_probe']['status'], '通过')
        self.assertEqual(result['capabilities']['mixing']['duck']['state'], 'supported')
        evidence = [e for e in result['capabilities']['evidence'] if e['evidence_id'] == 'explicit_duck_envelope']
        self.assertEqual(len(evidence), 1)

    def test_capability_probes_actual_split_union_and_deduplicates_new_evidence(self):
        result = add_duck_capability({'capabilities': {'mixing': {}, 'evidence': []}})
        self.assertIn('explicit_duck_multi_trigger', result['capabilities']['mixing']['duck']['evidence_ids'])
        probe = result['duck_multi_trigger_probe']
        self.assertEqual(probe['status'], '通过')
        self.assertEqual(probe['trigger_count'], 4)
        self.assertTrue(probe['split_equals_single_merged'])
        self.assertEqual(probe['split_output_sha256'], probe['single_merged_output_sha256'])
        add_duck_capability(result)
        self.assertEqual(sum(e['evidence_id'] == 'explicit_duck_multi_trigger'
                             for e in result['capabilities']['evidence']), 1)


class DuckIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.font = ROOT / 'skills/code-band-bgm/assets/soundfonts/MuseScore_General.sf3'
        cls.mapping = inventory(cls.font)
        cls.base_cap = load(ROOT / 'tests/fixtures/engine-capabilities-v1.json')
        cls.cap = add_duck_capability({'capabilities': copy.deepcopy(cls.base_cap)})['capabilities']
        cls.dry_score = load(ROOT / 'tests/fixtures/音频闭环乐谱.json')
        target = copy.deepcopy(cls.dry_score['parts'][0])
        target['part_id'], target['pan'] = 'target', -1
        target['notes'] = [copy.deepcopy(target['notes'][0])]
        target['notes'][0]['duration_ticks'] = 5760
        trigger = copy.deepcopy(target)
        trigger['part_id'], trigger['pan'] = 'trigger', 1
        trigger['notes'] = []
        for index, tick in enumerate([960, 1920]):
            note = copy.deepcopy(target['notes'][0])
            note['note_id'] = 'trigger_' + str(index)
            note['onset_tick'], note['duration_ticks'], note['pitch']['midi'] = tick, 240, 67
            trigger['notes'].append(note)
        cls.dry_score['parts'] = [target, trigger]
        cls.wet_score = copy.deepcopy(cls.dry_score)
        cls.wet_score['mix']['effects'] = [effect()]
        cls.asset = cls.mapping['source']['sha256']
        cls.dry = compile_score(cls.dry_score, cls.cap, cls.asset, cls.mapping)
        cls.wet = compile_score(cls.wet_score, cls.cap, cls.asset, cls.mapping)

    def test_real_soundfont_routes_only_target_and_keeps_trigger_sample_exact(self):
        # 两次真实SF渲染；硬左右声像让两声部在最终音频中可独立核对。
        dry, _ = render_events(self.dry, self.font)
        wet, diagnostic = render_events(self.wet, self.font)
        expected, expected_diagnostic = apply_duck(dry, 48000, effect(), self.wet['events'])
        self.assertEqual(dry.shape, (240000, 2))
        self.assertTrue(np.isfinite(wet).all())
        self.assertGreater(np.max(np.abs(dry[:, 0])), 1e-5)
        self.assertGreater(np.max(np.abs(dry[:, 1])), 1e-5)
        self.assertTrue(np.array_equal(wet[:, 1], dry[:, 1]))
        np.testing.assert_allclose(wet[:, 0], expected[:, 0], rtol=0, atol=1e-15)
        first_trigger = min(e['onset_frame'] for e in self.wet['events'] if e['part_id'] == 'trigger')
        self.assertTrue(np.array_equal(wet[:first_trigger], dry[:first_trigger]))
        self.assertGreater(np.max(np.abs(wet[:, 0] - dry[:, 0])), 1e-5)
        parts = {p['part_id']: p for p in diagnostic['parts']}
        self.assertNotIn('duck_effects', parts['trigger'])
        self.assertEqual(len(parts['target']['duck_effects']), 1)
        self.assertEqual(parts['target']['duck_effects'][0]['envelope_sha256'], expected_diagnostic['envelope_sha256'])
        self.assertEqual(self.wet['events'], self.dry['events'])

    def test_schema_semantics_and_current_capability_guard_compile_and_frozen(self):
        with self.assertRaises(ScoreError):
            compile_score(self.wet_score, self.base_cap, self.asset, self.mapping)
        self.assertIs(validate_frozen(self.wet, self.wet_score, self.cap, self.asset), self.wet)
        with self.assertRaises(ScoreError):
            validate_frozen(self.wet, self.wet_score, self.base_cap, self.asset)
        for parameters in [{'trigger_part_id': ['trigger']}, {'depth_db': '任意文本'}, {'unknown': 1}]:
            score = copy.deepcopy(self.wet_score)
            score['mix']['effects'][0]['parameters'].update(parameters)
            with self.subTest(parameters=parameters), self.assertRaises(ScoreError):
                compile_score(score, self.cap, self.asset, self.mapping)
        score = copy.deepcopy(self.wet_score)
        score['mix']['effects'][0]['parameters']['trigger_part_id'] = 'missing'
        with self.assertRaises(ScoreError):
            compile_score(score, self.cap, self.asset, self.mapping)
        changed = copy.deepcopy(self.wet)
        changed['mix']['effects'] = []
        with self.assertRaises(ScoreError):
            validate_frozen(changed, self.wet_score, self.cap, self.asset)

    def test_engine_rejects_bad_route_and_loop_before_synthesis(self):
        bad_target = copy.deepcopy(self.wet)
        bad_target['mix']['effects'][0]['target_part_ids'] = ['missing']
        bad_trigger = copy.deepcopy(self.wet)
        bad_trigger['mix']['effects'][0]['parameters']['trigger_part_id'] = 'missing'
        loop = copy.deepcopy(self.wet)
        loop['render']['loop']['enabled'] = True
        with patch('fluid_engine._Synth') as synth:
            for document in [bad_target, bad_trigger, loop]:
                with self.subTest(document=document['mix']), self.assertRaises(EngineError):
                    render_events(document, self.font)
            synth.assert_not_called()

    def test_plural_capability_gate_and_actual_three_part_target_equal_legacy_union(self):
        score = copy.deepcopy(self.dry_score)
        second = copy.deepcopy(score['parts'][1])
        second['part_id'] = 'body'
        for i, note in enumerate(second['notes']):
            note['note_id'] = 'body_' + str(i)
            note['onset_tick'] += 960
        score['parts'].append(second)
        score['mix']['effects'] = [effect()]
        score['mix']['effects'][0]['parameters'].pop('trigger_part_id')
        score['mix']['effects'][0]['parameters']['trigger_part_ids'] = ['trigger', 'body']
        legacy_cap = copy.deepcopy(self.cap)
        legacy_cap['mixing']['duck']['evidence_ids'] = ['explicit_duck_envelope']
        legacy_cap['evidence'] = [x for x in legacy_cap['evidence']
                                  if x['evidence_id'] != 'explicit_duck_multi_trigger']
        with self.assertRaisesRegex(ScoreError, '多触发'):
            compile_score(score, legacy_cap, self.asset, self.mapping)
        try:
            wet = compile_score(score, self.cap, self.asset, self.mapping)
        except ScoreError as exc:
            self.fail('新能力不能编译真实多触发：' + str(exc))
        with self.assertRaisesRegex(ScoreError, '多触发'):
            validate_frozen(wet, score, legacy_cap, self.asset)
        self.assertIs(validate_frozen(wet, score, self.cap, self.asset), wet)
        empty = copy.deepcopy(score)
        empty['parts'][-1]['notes'] = []
        with self.assertRaises(ScoreError):
            compile_score(empty, self.cap, self.asset, self.mapping)
        missing_source = copy.deepcopy(wet)
        missing_source['events'] = [x for x in missing_source['events'] if x['part_id'] != 'body']
        with self.assertRaisesRegex(ScoreError, '真实触发'):
            validate_frozen(missing_source, score, self.cap, self.asset)
        dry_score = copy.deepcopy(score)
        dry_score['mix']['effects'] = []
        dry = compile_score(dry_score, self.cap, self.asset, self.mapping)
        dry_audio, _ = render_events(dry, self.font)
        output, diagnostic = render_events(wet, self.font)
        merged = [{**event, 'part_id': 'trigger' if event['part_id'] in ('trigger', 'body') else event['part_id']}
                  for event in wet['events']]
        expected, expected_diagnostic = apply_duck(dry_audio, 48000, effect(), merged)
        self.assertTrue(np.array_equal(output[:, 1], dry_audio[:, 1]))
        np.testing.assert_allclose(output[:, 0], expected[:, 0], rtol=0, atol=1e-15)
        self.assertEqual(wet['mix']['master_gain_db'], dry['mix']['master_gain_db'])
        target = next(x for x in diagnostic['parts'] if x['part_id'] == 'target')
        detail = target['duck_effects'][0]
        self.assertEqual(detail['trigger_count'], 3)
        self.assertEqual(detail['trigger_source_event_counts'], {'trigger': 2, 'body': 2})
        self.assertEqual(detail['envelope_sha256'], expected_diagnostic['envelope_sha256'])


if __name__ == '__main__':
    unittest.main()
