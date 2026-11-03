"""固定的真实双轮换预设、随机流保护、拒绝边界及真实发声。"""
import copy
import json
import subprocess
import tempfile
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'skills/code-band-bgm/scripts'))
from fluid_engine import render_events
from sample_rotation import rotate_score, validate_rotation_parent, validate_rotation_output
from score_tools import ScoreError, compile_score, validate_frozen
from soundfont_inventory import inventory
from soundfont_packs import load_packs


class RotationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = json.loads((ROOT / 'tests/fixtures/音频闭环乐谱.json').read_text())
        cls.cap = json.loads((ROOT / 'tests/fixtures/engine-capabilities-expansion-20261031-v9.json').read_text())
        cls.main = inventory(ROOT / 'skills/code-band-bgm/assets/soundfonts/MuseScore_General.sf3')
        cls.sha = cls.main['source']['sha256']
        cls.packs = load_packs(ROOT / 'skills/code-band-bgm/assets/soundfont-packs.json')
        cls.resources = ['vcsl_cajon_rr1', 'vcsl_cajon_rr2']

    def score(self):
        score = copy.deepcopy(self.base)
        piano = score['parts'][0]
        piano['humanize'] = {'enabled': True, 'profile_id': 'fixed-other', 'rationale': '随机流隔离测试',
          'timing': {'max_offset_us': 3000, 'distribution': 'symmetric_triangular_integer',
                     'chord_policy': 'shared_offset', 'preserve_voice_order': True, 'lock_section_starts': True},
          'velocity': {'max_delta': 5, 'distribution': 'symmetric_triangular_integer', 'respect_notated_accents': True}}
        cajon = copy.deepcopy(self.base['parts'][0])
        cajon['part_id'] = 'cajon'
        cajon['resource_binding']['instrument_id'] = 'vcsl_cajon_rr1'
        for note in cajon['notes']:
            note['pitch'] = {'kind': 'unpitched', 'hit_key': 'hit_60'}
        score['parts'].append(cajon)
        return score

    def run_rotation(self, score, resources=None):
        return rotate_score(score, self.cap, self.sha, self.main, self.packs['inventories'],
                            'cajon', resources if resources is not None else self.resources)

    def test_seeded_rotation_preserves_score_notes_and_other_streams(self):
        score = self.score(); original = copy.deepcopy(score)
        before = compile_score(score, self.cap, self.sha, self.main, self.packs['inventories'])
        validate_rotation_parent(before, score, self.cap, self.sha, self.main, self.packs['inventories'])
        revised, events, report = self.run_rotation(score)
        self.assertTrue(validate_rotation_output(before, events, 'cajon', report['generated_part_ids']))
        self.assertEqual(score, original)
        self.assertEqual(self.run_rotation(score), (revised, events, report))
        self.assertEqual([x['slot'] for x in report['assignments']], [2, 1, 2])
        self.assertTrue(set(report['assignments'][0]['samples']).isdisjoint(report['assignments'][1]['samples']))
        self.assertEqual([x for x in before['events'] if x['part_id'] == 'piano'],
                         [x for x in events['events'] if x['part_id'] == 'piano'])
        source = {n['note_id']: n for n in original['parts'][1]['notes']}
        self.assertEqual(source, {n['note_id']: n for p in revised['parts'] if p['part_id'].startswith('cajon_rr') for n in p['notes']})
        available = list(self.packs['inventories'])
        inventories = {self.sha: self.main, **self.packs['inventories']}
        self.assertEqual(validate_frozen(events, revised, self.cap, self.sha, available, inventories), events)

    def test_mix_change_keeps_rotation_and_piano_random_stream(self):
        score = self.score(); a = self.run_rotation(score)
        score['mix']['master_gain_db'] -= 1
        b = self.run_rotation(score)
        self.assertEqual(a[2]['assignments'], b[2]['assignments'])
        self.assertEqual(a[1]['events'], b[1]['events'])

    def test_reject_unsafe_or_unverified_conditions(self):
        cases = []
        s = self.score(); s['parts'][1]['humanize'] = copy.deepcopy(s['parts'][0]['humanize']); cases.append(s)
        s = self.score(); s['render']['loop']['enabled'] = True; cases.append(s)
        s = self.score(); s['mix']['effects'].append({'effect_id': 'test', 'kind': 'reverb', 'target_part_ids': ['cajon'], 'parameters': {}, 'required': True, 'notes': 'unsafe'}); cases.append(s)
        s = self.score(); s['parts'][0]['part_id'] = 'cajon_rr1'; cases.append(s)
        s = self.score(); s['parts'][1]['notes'][0]['pitch']['hit_key'] = 'not_a_hit'; cases.append(s)
        for s in cases:
            with self.subTest(score=s), self.assertRaises(ScoreError): self.run_rotation(s)
        with self.assertRaises(ScoreError): self.run_rotation(self.score(), ['vcsl_cajon_rr1', 'vcsl_frame_drum_rr2'])
        with self.assertRaises(ScoreError): self.run_rotation(self.score(), ['vcsl_cajon_rr1', 'vcsl_cajon_rr1'])
        with self.assertRaises(ScoreError):
            rotate_score(self.score(), self.cap, self.sha, self.main, {}, 'cajon', self.resources)
        bad_cap = copy.deepcopy(self.cap)
        next(r for r in bad_cap['resources'] if r['instrument_id'] == 'vcsl_cajon_rr2')['engine_resource_id'] = 'sf:0:4:' + 'a5f223c7a587cf646fa4aef094bc132ee2d1eb1291bceb18527e78dcebe03052'
        with self.assertRaises(ScoreError):
            rotate_score(self.score(), bad_cap, self.sha, self.main, self.packs['inventories'], 'cajon', self.resources)

    def test_parent_rejects_external_controls_and_frozen_drift(self):
        score = self.score()
        before = compile_score(score, self.cap, self.sha, self.main, self.packs['inventories'])
        bad = copy.deepcopy(before)
        bad['control_events'] = [{'event_id': 'cc', 'part_id': 'piano', 'frame': 0, 'controller': 11, 'value': 100}]
        with self.assertRaises(ScoreError):
            validate_rotation_parent(bad, score, self.cap, self.sha, self.main, self.packs['inventories'])
        bad = copy.deepcopy(before)
        bad['score_sha256'] = '0' * 64
        with self.assertRaises(ScoreError):
            validate_rotation_parent(bad, score, self.cap, self.sha, self.main, self.packs['inventories'])
        bad = copy.deepcopy(before)
        other = next(e for e in bad['events'] if e['part_id'] == 'piano')
        other['velocity'] += 1
        with self.assertRaises(ScoreError):
            validate_rotation_parent(bad, score, self.cap, self.sha, self.main, self.packs['inventories'])
        _, events, report = self.run_rotation(score)
        next(e for e in events['events'] if e['part_id'] == 'piano')['velocity'] += 1
        with self.assertRaises(ScoreError):
            validate_rotation_output(before, events, 'cajon', report['generated_part_ids'])
        _, events, report = self.run_rotation(score)
        events['events'].append(copy.deepcopy(next(e for e in events['events'] if e['part_id'] == 'cajon_rr1')))
        with self.assertRaises(ScoreError):
            validate_rotation_output(before, events, 'cajon', report['generated_part_ids'])

    def test_cli_error_is_json_and_creates_no_output(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'input.json'
            score = self.score()
            source.write_text(json.dumps(score, ensure_ascii=False))
            frozen = root / 'events.json'
            frozen.write_text(json.dumps(compile_score(score, self.cap, self.sha, self.main, self.packs['inventories']), ensure_ascii=False))
            output = root / 'output'
            result = subprocess.run([sys.executable, '-B', str(ROOT / 'skills/code-band-bgm/scripts/sample_rotation.py'),
                '--score', str(source), '--events', str(frozen), '--part', 'cajon', '--resources', 'vcsl_cajon_rr1', 'vcsl_frame_drum_rr2',
                '--capabilities', str(ROOT / 'tests/fixtures/engine-capabilities-expansion-20261031-v9.json'),
                '--soundfont', str(ROOT / 'skills/code-band-bgm/assets/soundfonts/MuseScore_General.sf3'),
                '--soundfont-packs', str(ROOT / 'skills/code-band-bgm/assets/soundfont-packs.json'), '--out', str(output)],
                text=True, capture_output=True)
            self.assertEqual(result.returncode, 1)
            self.assertEqual(json.loads(result.stderr)['status'], '失败')
            self.assertFalse(output.exists())
            output.mkdir()
            sentinel = output / 'keep.txt'
            sentinel.write_text('不可覆盖')
            again = subprocess.run([sys.executable, '-B', str(ROOT / 'skills/code-band-bgm/scripts/sample_rotation.py'),
                '--score', str(source), '--events', str(frozen), '--part', 'cajon', '--resources', 'vcsl_cajon_rr1', 'vcsl_cajon_rr2',
                '--capabilities', str(ROOT / 'tests/fixtures/engine-capabilities-expansion-20261031-v9.json'),
                '--soundfont', str(ROOT / 'skills/code-band-bgm/assets/soundfonts/MuseScore_General.sf3'),
                '--soundfont-packs', str(ROOT / 'skills/code-band-bgm/assets/soundfont-packs.json'), '--out', str(output)],
                text=True, capture_output=True)
            self.assertEqual(again.returncode, 1)
            self.assertEqual(json.loads(again.stderr)['status'], '失败')
            self.assertEqual(sentinel.read_text(), '不可覆盖')

    def test_actual_fixed_preset_rotation_renders_and_replays(self):
        revised, events, report = self.run_rotation(self.score())
        font = ROOT / 'skills/code-band-bgm/assets/soundfonts/MuseScore_General.sf3'
        a, info = render_events(events, font, font_paths=self.packs['font_paths'])
        b, _ = render_events(events, font, font_paths=self.packs['font_paths'])
        self.assertEqual(a.shape, (events['total_frames'], 2))
        self.assertTrue(np.isfinite(a).all())
        self.assertGreater(float(np.max(np.abs(a))), 1e-5)
        self.assertTrue(np.array_equal(a, b))
        for identity in report['generated_part_ids']:
            part = next(x for x in info['parts'] if x['part_id'] == identity)
            self.assertGreater(part['rms_linear'], 0)


if __name__ == '__main__': unittest.main()
