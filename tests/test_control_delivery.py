"""真实控制事件随作品保存，并能通过统一入口冻结复演。"""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / 'skills/code-band-bgm/scripts/band.py'
SCORE = ROOT / 'skills/code-band-bgm/assets/示例/五秒钢琴.json'
CAP = ROOT / 'tests/fixtures/engine-capabilities-v24-eq.json'
FONT = ROOT / 'skills/code-band-bgm/assets/soundfonts/MuseScore_General.sf3'


class ControlDeliveryTests(unittest.TestCase):
    def test_controlled_delivery_replay_and_conflict(self):
        if not CAP.is_file() or not FONT.is_file():
            self.skipTest('缺真实能力记录或采样库')
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            control = root / 'control.json'
            control.write_text(json.dumps([{'event_id': 'cc_a', 'part_id': 'piano', 'frame': 0,
                                            'controller': 11, 'value': 64}]))
            first, replay, conflict = root / 'first', root / 'replay', root / 'conflict'
            base = [sys.executable, str(CLI), 'render', '--score', str(SCORE), '--capabilities', str(CAP)]
            def run(args):
                return subprocess.run(args, capture_output=True, text=True)
            result = run(base + ['--control-events', str(control), '--out', str(first)])
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(json.loads(result.stdout)['activity_review'],
                             json.loads((first / 'audio_check.json').read_text())['activity_review'])
            self.assertEqual(json.loads((first / 'manifest.json').read_text())['activity_review'],
                             json.loads(result.stdout)['activity_review'])
            events = json.loads((first / 'events.json').read_text())
            self.assertEqual(events['control_events'][0]['event_id'], 'cc_a')
            result = run(base + ['--events', str(first / 'events.json'), '--out', str(replay)])
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual((first / 'final.wav').read_bytes(), (replay / 'final.wav').read_bytes())
            result = run(base + ['--events', str(first / 'events.json'), '--control-events', str(control), '--out', str(conflict)])
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(conflict.exists())
            control.write_text('null')
            invalid = root / 'invalid'
            result = run(base + ['--control-events', str(control), '--out', str(invalid)])
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(invalid.exists())
            events['control_events'] = None
            frozen = root / 'frozen-null.json'
            frozen.write_text(json.dumps(events))
            result = run(base + ['--events', str(frozen), '--out', str(invalid)])
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(invalid.exists())
            gated = root / 'gated'
            result = run(base + ['--continuous-tail-max-seconds', '0', '--out', str(gated)])
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(gated.exists())
            reports = list(root.glob('.render-*/audio_check_failed.json'))
            self.assertEqual(len(reports), 1)
            self.assertFalse(json.loads(reports[0].read_text())['measurements']['continuous_tail_coverage']['accepted'])


if __name__ == '__main__':
    unittest.main()
