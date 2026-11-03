"""音符候选的语义、虚拟环境与失败发布回归；不加载模型。"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'skills/code-band-bgm/scripts'))
import note_analysis as notes
from audio_tools import AudioError


class NoteAnalysisTests(unittest.TestCase):
    def report(self):
        return {
            'status': 'analysis_candidates_only',
            'source': {'sha256': 'a' * 64, 'frames': 8000, 'sample_rate_hz': 16000, 'channels': 1},
            'notes': [{'program': 0, 'is_drum': False, 'pitch': 60, 'velocity': 1,
                       'onset': 0.0, 'offset': .5}],
            'runtime': {'checkpoint_sha256': 'b' * 64, 'strict_weights': [
                {'strict': True, 'model_key_count': 3, 'checkpoint_key_count': 3,
                 'missing_keys': [], 'unexpected_keys': []}]}}

    def validate(self, report):
        return notes.validate_candidates(report, 8000, 'a' * 64, 'b' * 64)

    def files(self):
        # 保留独占临时证据；不递归删除文件或目录。
        base = Path(tempfile.mkdtemp(prefix='code-band-notes-test-'))
        source, config = base / '输入.wav', base / '模型配置.json'
        sf.write(source, np.zeros(8000, dtype=np.float32), 16000, subtype='FLOAT')
        config.write_text(json.dumps({'checkpoint_sha256': 'b' * 64}), encoding='utf-8')
        executable = base / '基础解释器'
        executable.write_text('仅作路径测试，不执行', encoding='utf-8')
        link = base / 'venv/bin/python'
        link.parent.mkdir(parents=True)
        link.symlink_to(executable)
        return base, source, config, link, base / '候选交付'

    def test_valid_notes_remain_candidates_and_drum_sentinel_is_allowed(self):
        report = self.report()
        self.assertIs(self.validate(report), report)
        report['notes'][0].update(program=128, is_drum=True)
        self.assertEqual(self.validate(report)['status'], 'analysis_candidates_only')

    def test_source_hash_frames_and_format_must_match(self):
        for key, value in [('sha256', 'c' * 64), ('frames', 7999),
                           ('sample_rate_hz', 48000), ('channels', 2)]:
            with self.subTest(key=key):
                report = self.report()
                report['source'][key] = value
                with self.assertRaises(AudioError):
                    self.validate(report)

    def test_nan_infinity_and_out_of_window_times_are_rejected(self):
        for key, value in [('onset', float('nan')), ('offset', float('inf')),
                           ('onset', -.01), ('offset', .501), ('offset', 0),
                           ('onset', True)]:
            with self.subTest(key=key, value=value):
                report = self.report()
                report['notes'][0][key] = value
                with self.assertRaises(AudioError):
                    self.validate(report)

    def test_invalid_pitch_velocity_and_non_drum_program128_are_rejected(self):
        for key, value in [('pitch', 128), ('pitch', True), ('velocity', 0),
                           ('program', 129), ('program', 128), ('is_drum', 1)]:
            with self.subTest(key=key, value=value):
                report = self.report()
                report['notes'][0][key] = value
                with self.assertRaises(AudioError):
                    self.validate(report)

    def test_strict_missing_extra_mismatched_and_zero_weights_are_rejected(self):
        for key, value in [('strict', False), ('missing_keys', ['encoder.weight']),
                           ('unexpected_keys', ['unknown.weight']),
                           ('checkpoint_key_count', 2), ('model_key_count', 0),
                           ('model_key_count', True)]:
            with self.subTest(key=key):
                report = self.report()
                report['runtime']['strict_weights'][0][key] = value
                with self.assertRaises(AudioError):
                    self.validate(report)

    def test_wrong_model_hash_and_incomplete_backend_status_are_rejected(self):
        report = self.report()
        report['runtime']['checkpoint_sha256'] = 'c' * 64
        with self.assertRaises(AudioError):
            self.validate(report)
        for status in ['通过', 'failed', None]:
            with self.subTest(status=status):
                report = self.report()
                report['status'] = status
                with self.assertRaises(AudioError):
                    self.validate(report)

    def test_venv_python_symlink_is_preserved_and_backend_failure_never_publishes(self):
        base, source, config, link, output = self.files()
        captured = []
        def fail_backend(argv, **kwargs):
            captured.append(argv)
            return SimpleNamespace(returncode=2, stdout='', stderr='受控后端失败')
        backend = SimpleNamespace(run=fail_backend, TimeoutExpired=subprocess.TimeoutExpired)
        with patch.object(notes, 'subprocess', backend):
            with self.assertRaises(AudioError) as raised:
                notes.analyze_notes(source, config, link, output, seconds=.5)
        self.assertEqual(raised.exception.code, 'note_backend_failed')
        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0][0], os.path.abspath(link))
        self.assertNotEqual(captured[0][0], str(link.resolve()))
        self.assertFalse(output.exists())
        failures = list(base.glob('.notes-*/failure.json'))
        self.assertEqual(len(failures), 1)
        self.assertEqual(json.loads(failures[0].read_text())['status'], '失败')
        self.assertIn('受控后端失败', (failures[0].parent / 'backend.log').read_text())

    def test_existing_output_is_rejected_before_backend_or_decoding(self):
        base, source, config, link, output = self.files()
        output.mkdir()
        marker = output / '原文件.txt'
        marker.write_text('保留', encoding='utf-8')
        with patch.object(notes, '_probe') as probe:
            with self.assertRaises(AudioError) as raised:
                notes.analyze_notes(source, config, link, output, seconds=.5)
        self.assertEqual(raised.exception.code, 'output_exists')
        probe.assert_not_called()
        self.assertEqual(marker.read_text(), '保留')
        self.assertEqual(list(base.glob('.notes-*')), [])

    def test_backend_timeout_and_invalid_windows_do_not_publish(self):
        base, source, config, link, output = self.files()
        def timed_out(argv, **kwargs):
            raise subprocess.TimeoutExpired(argv, 210)
        backend = SimpleNamespace(run=timed_out, TimeoutExpired=subprocess.TimeoutExpired)
        with patch.object(notes, 'subprocess', backend):
            with self.assertRaises(AudioError) as raised:
                notes.analyze_notes(source, config, link, output, seconds=.5)
        self.assertEqual(raised.exception.code, 'note_backend_timeout')
        self.assertFalse(output.exists())
        for args in [dict(seconds=float('nan')), dict(start=-1), dict(seconds=31),
                     dict(seconds=.49), dict(prefix_seconds=.1)]:
            with self.subTest(args=args), self.assertRaises(AudioError) as invalid:
                notes.analyze_notes(source, config, link, base/'不应创建', **args)
            self.assertEqual(invalid.exception.code, 'note_window')
        self.assertFalse((base/'不应创建').exists())

    def test_import_does_not_load_torch_or_model_backend(self):
        code = ('import sys; sys.path.insert(0, ' + repr(str(ROOT/'skills/code-band-bgm/scripts')) +
                '); import note_analysis; assert "torch" not in sys.modules; '
                'assert "yourmt3_local" not in sys.modules; print("无可选模型导入")')
        result = subprocess.run([sys.executable, '-B', '-c', code],
                                capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('无可选模型导入', result.stdout)


if __name__ == '__main__':
    unittest.main()
