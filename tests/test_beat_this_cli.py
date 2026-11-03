"""统一入口的模型参数与失败发布边界；不依赖安装可选模型。"""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]


class BeatThisCliTests(unittest.TestCase):
    def test_required_model_and_unused_model_arguments_fail_without_creating_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / '不应发布.json'
            for extra in [
                ['--rhythm-backend', 'beat_this'],
                ['--rhythm-backend', 'basic', '--rhythm-model', '不应静默忽略.ckpt']]:
                result = subprocess.run([sys.executable, str(ROOT/'skills/code-band-bgm/scripts/band.py'),
                    'analyze', '--input', '无需读取的缺失文件.wav', '--out', str(output), *extra],
                    capture_output=True, text=True, timeout=20)
                report = json.loads(result.stdout)
                self.assertEqual(result.returncode, 2)
                self.assertEqual(report['code'], 'rhythm_model')
                self.assertFalse(output.exists())

    def test_wrong_checkpoint_hash_is_rejected_through_real_cli_before_optional_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source, model, output = directory/'诊断.wav', directory/'模型.ckpt', directory/'不应发布.json'
            time = np.arange(22050*5)/22050
            sf.write(source, .03*np.sin(2*np.pi*440*time), 22050, subtype='PCM_16')
            model.write_bytes(b'not-a-model')
            result = subprocess.run([sys.executable, str(ROOT/'skills/code-band-bgm/scripts/band.py'),
                'analyze', '--input', str(source), '--rhythm-backend', 'beat_this',
                '--rhythm-model', str(model), '--rhythm-model-sha256', '0'*64,
                '--out', str(output)], capture_output=True, text=True, timeout=20)
            report = json.loads(result.stdout)
            self.assertEqual(result.returncode, 2)
            self.assertIn('SHA256', report['details']['reason'])
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
