"""复用缓存必须有逐资源真实探测记录；空列表不能冒充已测130项。"""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'skills/code-band-bgm/scripts'))
import band
from score_tools import ScoreError, load


class CapabilityReuseTests(unittest.TestCase):
    def test_missing_duplicate_or_invalid_probe_is_rejected_before_extension(self):
        base = load(ROOT / 'tests/fixtures/engine-capabilities-v1.json')
        original = load(ROOT / 'tests/fixtures/engine-capabilities-v1.json.probe.json')
        cases = [[], original['probe_results'][:-1], original['probe_results'] + original['probe_results'][:1]]
        for key, value in [('peak', 0), ('peak', True), ('frames', 0), ('tested_midi', True)]:
            records = copy.deepcopy(original['probe_results'])
            records[0][key] = value
            cases.append(records)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'cap.json'
            path.write_text(json.dumps(base))
            for records in cases:
                previous = copy.deepcopy(original)
                previous['probe_results'] = records
                Path(str(path) + '.probe.json').write_text(json.dumps(previous))
                with patch('soundfont_packs.load_packs', side_effect=AssertionError('基础证据无效时不能继续探测')):
                    with self.assertRaises(ScoreError) as raised:
                        band.extend_capabilities(band.DEFAULT_SF, path, '隔离额外登记.json')
                self.assertEqual(raised.exception.details['field'], 'capabilities/evidence')

    def test_complete_original_records_can_be_explicitly_reused(self):
        with patch('soundfont_packs.load_packs', return_value={'resources': [], 'probe_results': [], 'assets': []}):
            result = band.extend_capabilities(band.DEFAULT_SF, ROOT / 'tests/fixtures/engine-capabilities-v1.json', '隔离额外登记.json')
        self.assertEqual(len(result['probe_results']), len(result['capabilities']['resources']))
        self.assertIn('reused_primary_probe', result)


if __name__ == '__main__':
    unittest.main()
