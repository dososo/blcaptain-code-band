"""多标签乐器候选统一入口；接口可用不代替乐器身份或风格核准。"""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
BAND = ROOT / 'skills/code-band-bgm/scripts/band.py'


class InstrumentCliTests(unittest.TestCase):
    def test_help_exposes_explicit_local_assets_and_window_plan(self):
        result = subprocess.run([sys.executable, '-B', str(BAND), 'analyze-instruments', '--help'],
                                capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
        for flag in ('--input', '--model-config', '--backend-python', '--window-plan', '--out'):
            self.assertIn(flag, result.stdout)

    def test_missing_assets_return_json_failure_without_false_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / '不能假成功'
            result = subprocess.run([sys.executable, '-B', str(BAND), 'analyze-instruments',
                '--input', '不存在的参考.wav', '--model-config', '不存在的模型配置.json',
                '--backend-python', sys.executable, '--window-plan', '不存在的分窗.json', '--out', str(out)],
                capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, 2)
            self.assertEqual(json.loads(result.stdout)['status'], '失败')
            self.assertFalse(out.exists())


if __name__ == '__main__':
    unittest.main()
