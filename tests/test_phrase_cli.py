"""统一入口的乐句修改、零强度和失败保护。"""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / 'skills/code-band-bgm/scripts/band.py'
SCORE = ROOT / 'tests/fixtures/乐谱_结构示例.json'


class PhraseCLITests(unittest.TestCase):
    def test_zero_and_invalid_plan(self):
        score = json.loads(SCORE.read_text())
        plan = {'schema_version': '1.0.0', 'start_tick': 0, 'end_tick': 7680,
                'edits': [{'part_id': 'demo_keys', 'note_ids': ['demo_note_001'],
                           'offset_ticks': 12, 'velocity_delta': 2}]}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            planfile, out = path / 'plan.json', path / 'score.json'
            planfile.write_text(json.dumps(plan))
            args = [sys.executable, str(CLI), 'phrase', '--score', str(SCORE), '--plan', str(planfile),
                    '--strength', '0', '--out', str(out)]
            result = subprocess.run(args, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(json.loads(out.read_text()), score)
            change = json.loads(Path(str(out) + '.change.json').read_text())
            self.assertEqual(change['changes'], [])
            self.assertEqual(change['score_sha256'], change['parent_score_sha256'])
            original = out.read_bytes()
            self.assertNotEqual(subprocess.run(args, capture_output=True).returncode, 0)
            self.assertEqual(out.read_bytes(), original)
            plan['edits'][0]['note_ids'] = ['不存在']
            planfile.write_text(json.dumps(plan))
            args[-1] = str(path / 'invalid.json')
            self.assertNotEqual(subprocess.run(args, capture_output=True).returncode, 0)
            self.assertFalse((path / 'invalid.json').exists())


if __name__ == '__main__':
    unittest.main()
