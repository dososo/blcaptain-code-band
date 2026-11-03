"""可选CPU拍点入口的失败边界；夹具不是产品音乐或模型准确率测试。"""
from contextlib import nullcontext
import hashlib
from importlib.metadata import PackageNotFoundError
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'skills/code-band-bgm/scripts'))
import beat_this_analysis


class BeatThisTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='code-band-beat-this-')
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / '本地模型.ckpt'
        self.payload = b'checkpoint-test-data'
        self.path.write_bytes(self.payload)
        self.sha = hashlib.sha256(self.payload).hexdigest()
        self.audio = np.full(22050 * 5, .02, dtype=np.float32)

    def fake_dependencies(self, beats=(.5, 1., 1.5, 2.), downbeats=(.5,), factory=None, device='cpu'):
        torch = types.ModuleType('torch')
        torch.float32 = 'float32'
        torch.get_num_threads = Mock(return_value=2)
        torch.set_num_threads = Mock()
        torch.inference_mode = nullcontext
        tracker = Mock(return_value=(np.asarray(beats), np.asarray(downbeats)))
        tracker.model.parameters.return_value = iter([types.SimpleNamespace(
            device=types.SimpleNamespace(type=device), dtype='float32')])
        inference = types.ModuleType('beat_this.inference')
        inference.Audio2Beats = Mock(side_effect=factory) if factory else Mock(return_value=tracker)
        return torch, inference, tracker

    def call_with_fake(self, **options):
        torch, inference, tracker = self.fake_dependencies(**options)
        with patch.dict(sys.modules, {'torch': torch, 'beat_this.inference': inference}), \
                patch.object(beat_this_analysis, 'version', side_effect=lambda n: {'beat-this': '1.1.0', 'torch': '2.9.1'}[n]):
            result = beat_this_analysis.analyze_beats(self.audio, 22050, self.path, self.sha)
        return result, torch, inference, tracker

    def test_same_verified_open_handle_survives_path_removal_and_cpu_settings_restore(self):
        torch, inference, tracker = self.fake_dependencies()
        handles = []
        def construct(checkpoint_path, **kwargs):
            self.assertTrue(hasattr(checkpoint_path, 'read'))
            self.assertEqual(checkpoint_path.tell(), 0)
            self.path.unlink()
            self.assertEqual(checkpoint_path.read(), self.payload)
            self.assertEqual(kwargs, {'device': 'cpu', 'float16': False, 'dbn': False})
            handles.append(checkpoint_path)
            return tracker
        inference.Audio2Beats.side_effect = construct
        with patch.dict(sys.modules, {'torch': torch, 'beat_this.inference': inference}), \
                patch.object(beat_this_analysis, 'version', side_effect=lambda n: {'beat-this': '1.1.0', 'torch': '2.9.1'}[n]):
            report = beat_this_analysis.analyze_beats(self.audio, 22050, self.path, self.sha)
        self.assertTrue(handles[0].closed)
        self.assertEqual(torch.set_num_threads.call_args_list, [unittest.mock.call(4), unittest.mock.call(2)])
        self.assertEqual(report['status_scope'], 'rhythm_measurement_candidates')
        self.assertEqual(report['backend']['model_sha256'], self.sha)
        self.assertEqual(report['beats_seconds'], [.5, 1., 1.5, 2.])
        self.assertEqual(report['observation']['end_seconds'], 5.)
        self.assertNotIn('tempo_bpm', report)
        json.dumps(report, allow_nan=False)

    def test_missing_directory_and_wrong_sha_fail_before_optional_import(self):
        with patch.object(beat_this_analysis, 'version', side_effect=AssertionError('不应加载依赖')):
            for path, checksum in [(self.path.with_name('缺失.ckpt'), self.sha),
                                    (self.path.parent, self.sha), (self.path, '0' * 64),
                                    (self.path, '非法摘要'), ('https://example.com/model.ckpt', self.sha)]:
                with self.subTest(path=path, checksum=checksum), self.assertRaises(ValueError):
                    beat_this_analysis.analyze_beats(self.audio, 22050, path, checksum)

    def test_nonfinite_stereo_wrong_rate_silence_and_duration_are_rejected(self):
        long_view = np.lib.stride_tricks.as_strided(np.array([.1], dtype=np.float32),
                    shape=(22050 * 600 + 1,), strides=(0,))
        cases = [(np.full(22050 * 5, np.nan), 22050), (np.full(22050 * 5, np.inf), 22050),
                 (np.zeros((22050 * 5, 2)), 22050), (self.audio, 44100), (self.audio, True),
                 (np.zeros(22050 * 5), 22050), (self.audio[:100], 22050), (long_view, 22050),
                 (np.array(['并非波形']), 22050), (np.full(22050 * 5, 1 + 2j), 22050)]
        for audio, rate in cases:
            with self.subTest(shape=audio.shape, rate=rate), self.assertRaises(ValueError):
                beat_this_analysis.analyze_beats(audio, rate, self.path, self.sha)

    def test_missing_or_unverified_dependencies_are_explicit(self):
        with patch.object(beat_this_analysis, 'version', side_effect=PackageNotFoundError('beat-this')):
            with self.assertRaisesRegex(ValueError, '依赖'):
                beat_this_analysis.analyze_beats(self.audio, 22050, self.path, self.sha)
        for package_version, torch_version in [('1.0.0', '2.9.1'), ('1.1.0', '1.13.1')]:
            with patch.object(beat_this_analysis, 'version', side_effect=lambda n: {'beat-this': package_version, 'torch': torch_version}[n]):
                with self.assertRaises(ValueError):
                    beat_this_analysis.analyze_beats(self.audio, 22050, self.path, self.sha)

    def test_empty_unordered_nonfinite_and_out_of_range_outputs_do_not_pass(self):
        for beats, downbeats in [((), ()), ((1., .5), (.5,)), ((.5, .5), (.5,)),
                                 (np.array([2, 1], dtype=np.uint8), (1.,)),
                                 ((.5, np.nan), (.5,)), ((-.1, .5), (.5,)), ((.5, 5.1), (.5,)),
                                 ((.5, 1.), (1., .5)), ((.5, 1.), (.75,)),
                                 ((.5, 1.), (np.inf,))]:
            with self.subTest(beats=beats, downbeats=downbeats), self.assertRaises(ValueError):
                self.call_with_fake(beats=beats, downbeats=downbeats)

    def test_beats_without_downbeats_keep_unknown_instead_of_inventing_meter(self):
        report, *_ = self.call_with_fake(downbeats=())
        self.assertEqual(report['downbeats_seconds'], [])
        self.assertEqual(report['downbeat_status'], '未知：模型未检出强拍')

    def test_backend_gpu_or_model_failure_never_silently_falls_back(self):
        with self.assertRaises(ValueError):
            self.call_with_fake(device='mps')
        with self.assertRaises(ValueError):
            self.call_with_fake(factory=Mock(side_effect=RuntimeError('本地载入失败')))


if __name__ == '__main__':
    unittest.main()
