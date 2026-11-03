"""已注册打击键必须有逐键发声证据，不能只留鼓组的一个代表音。"""
import copy
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'skills/code-band-bgm/scripts'))
from percussion_tools import validate_cached_percussion
from score_tools import ScoreError, load


class PercussionCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = load(ROOT / 'tests/fixtures/engine-capabilities-final.json.probe.json')
        cls.cap = cls.report['capabilities']

    def test_actual_complete_report_has_all_registered_keys(self):
        validate_cached_percussion(self.cap, self.report)

    def test_missing_record_silence_or_wrong_frame_count_cannot_be_reused(self):
        for fault in ['missing_all', 'missing_key', 'silence', 'wrong_frames']:
            changed = copy.deepcopy(self.report)
            if fault == 'missing_all':
                changed.pop('percussion_probe_results')
            elif fault == 'missing_key':
                changed['percussion_probe_results'][0]['hits'].pop()
            elif fault == 'silence':
                changed['percussion_probe_results'][0]['hits'][0]['peak_linear'] = 0
            else:
                changed['percussion_probe_results'][0]['hits'][0]['frames'] = 1
            with self.assertRaises(ScoreError):
                validate_cached_percussion(self.cap, changed)


if __name__ == '__main__':
    unittest.main()
