"""额外音色登记必须绑定真实文件、许可和预设，不能按标签假接入。"""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'skills/code-band-bgm/scripts'))
from score_tools import ScoreError
from soundfont_packs import load_packs

REGISTRY = ROOT / 'skills/code-band-bgm/assets/soundfont-packs.json'


class SoundfontPacksTests(unittest.TestCase):
    def fixture(self, root):
        data = json.loads(REGISTRY.read_text())
        for pack in data['packs']:
            pack['path'] = str((REGISTRY.parent / pack['path']).resolve())
            pack['license_file'] = str((REGISTRY.parent / pack['license_file']).resolve())
            if pack.get('source_manifest'):
                pack['source_manifest'] = str((REGISTRY.parent / pack['source_manifest']).resolve())
        path = Path(root) / 'packs.json'
        return data, path

    def test_relative_registry_resolves_real_expansions_and_original_prototypes(self):
        packs = load_packs(REGISTRY)
        self.assertGreaterEqual(len(packs['font_paths']), 4)
        self.assertGreaterEqual(len(packs['resources']), 28)
        self.assertTrue({'erhu_long_cc0', 'guqin_open_strings_ccby4', 'vcsl_cajon_rr1',
                         'vcsl_upright_piano_rr1'} <= {r['instrument_id'] for r in packs['resources']})
        self.assertEqual(set(packs['license_paths']), set(packs['font_paths']))
        by_id = {r['instrument_id']: r for r in packs['resources']}
        self.assertEqual(by_id['erhu_long_cc0']['pitch_range'], {'min_midi': 62, 'max_midi': 81})
        self.assertEqual(by_id['fp_ukulele_percussion']['unpitched_hits'], {'body_tap': '47', 'chuck': '48'})
        self.assertEqual(packs['assets'][0]['license_id'], 'CC0-1.0')

    def test_corrupt_hash_never_reaches_instrument_registration(self):
        with tempfile.TemporaryDirectory() as root:
            data, path = self.fixture(root)
            data['packs'][0]['sha256'] = '0' * 64
            path.write_text(json.dumps(data))
            with self.assertRaises(ScoreError) as raised:
                load_packs(path)
            self.assertEqual(raised.exception.details['field'], 'soundfont_packs/sha256')

    def test_declared_keys_cannot_hide_missing_real_sample(self):
        with tempfile.TemporaryDirectory() as root:
            data, path = self.fixture(root)
            pack = next(p for p in data['packs'] if p['pack_id'] == 'pipaset_20261031_v1')
            pack['resources'][0]['supported_keys'].append(87)
            path.write_text(json.dumps(data))
            with self.assertRaises(ScoreError) as raised:
                load_packs(path)
            self.assertEqual(raised.exception.details['field'], 'soundfont_packs/supported_keys')

    def test_invalid_container_returns_structured_validation_error(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'invalid.json'
            bad_hits, _ = self.fixture(root)
            bad_hits['packs'][0]['resources'][2]['unpitched_hits'] = []
            for document in ([], {'schema_version': '1.0.0', 'packs': [[]]}, bad_hits):
                path.write_text(json.dumps(document))
                with self.assertRaises(ScoreError):
                    load_packs(path)

    def test_missing_license_and_nonexistent_preset_are_not_available(self):
        with tempfile.TemporaryDirectory() as root:
            original, path = self.fixture(root)
            for field in ['license_file', 'preset']:
                data = copy.deepcopy(original)
                if field == 'license_file':
                    data['packs'][0]['license_file'] = str(Path(root) / '不存在的许可.txt')
                else:
                    data['packs'][0]['resources'][0]['program'] = 127
                path.write_text(json.dumps(data))
                with self.assertRaises(ScoreError):
                    load_packs(path)

    def test_duplicate_instrument_and_unmapped_hit_are_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            original, path = self.fixture(root)
            for invalid in ['duplicate', 'hit']:
                data = copy.deepcopy(original)
                if invalid == 'duplicate':
                    data['packs'][0]['resources'].append(copy.deepcopy(data['packs'][0]['resources'][0]))
                else:
                    pack = next(p for p in data['packs'] if any('unpitched_hits' in r for r in p['resources']))
                    next(r for r in pack['resources'] if 'unpitched_hits' in r)['unpitched_hits']['fake'] = '127'
                path.write_text(json.dumps(data))
                with self.assertRaises(ScoreError):
                    load_packs(path)


if __name__ == '__main__':
    unittest.main()
