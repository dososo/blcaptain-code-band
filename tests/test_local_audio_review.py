"""本地描述适配器的离线输入与解码边界，不冒充音乐听感验收。"""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock

import numpy as np
import soundfile as sf

SCRIPT = Path(__file__).resolve().parents[1] / "skills/code-band-bgm/scripts/local_audio_review.py"
sys.path.insert(0, str(SCRIPT.parent))
spec = importlib.util.spec_from_file_location("local_audio_review", SCRIPT)
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)


class LocalAudioReviewTests(unittest.TestCase):
    def test_long_reference_is_sampled_at_reportable_positions(self):
        self.assertEqual(review.segment_starts(26.215), [0.0])
        self.assertEqual(review.segment_starts(180), [0.0, 75.0, 150.0])

    def test_local_decode_preserves_actual_window_and_treats_filename_as_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "参考 $(不应执行).wav"
            signal = .1 * np.sin(np.arange(48000) * 440 * 2 * np.pi / 48000)
            sf.write(path, np.column_stack([signal, signal]), 48000)
            wave = review.decode_segment(path, .25)
            self.assertEqual(len(wave), 12000)
            self.assertTrue(np.isfinite(wave).all())
            self.assertGreater(np.max(np.abs(wave)), .09)

    def test_uninstalled_model_or_remote_audio_fails_before_model_import(self):
        with self.assertRaisesRegex(ValueError, "本地音频"):
            review.describe_audio("https://example.com/song.wav", "missing-model")
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "reference.wav"
            source.write_bytes(b"not audio")
            with self.assertRaisesRegex(ValueError, "不自动下载或上传"):
                review.describe_audio(source, Path(directory) / "missing-model")

    def test_corrupt_audio_has_no_semantic_result(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "损坏.wav"
            source.write_bytes(b"not audio")
            with self.assertRaisesRegex(ValueError, "无法在预算内解码"):
                review.decode_segment(source, 0)

    def test_playlist_cannot_indirectly_read_another_file(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "不是直接音频.m3u8"
            source.write_text('#EXTM3U\n#EXTINF:1,\nfile:///不应读取的文件.wav\n')
            with self.assertRaisesRegex(ValueError, "无法在预算内解码"):
                review.decode_segment(source, 0)

    def test_malformed_model_config_does_not_import_optional_backend(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "reference.wav"
            source.write_bytes(b"not decoded for this test")
            model = Path(directory) / "model"
            model.mkdir()
            (model / "config.json").write_text("[]")
            with self.assertRaisesRegex(ValueError, "当前可选适配器"):
                review.describe_audio(source, model)

    def test_invalid_feature_shape_is_rejected_before_encoder(self):
        model = Mock()
        model._processor.feature_extractor.return_value = {"input_features": np.zeros((1, 128, 3001))}
        review.use_official_features(model, Mock())
        with self.assertRaisesRegex(ValueError, "特征维度"):
            model._extract_features(np.zeros(16000))

    def test_valid_feature_lengths_match_official_conv_and_pool(self):
        self.assertEqual(review.audio_feature_lengths(500), (250, 125))
        self.assertEqual(review.audio_feature_lengths(1499), (750, 375))
        self.assertEqual(review.audio_feature_lengths(3000), (1500, 750))
        for invalid in (0, 2, 3001, True, 500.):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                review.audio_feature_lengths(invalid)


if __name__ == "__main__":
    unittest.main()
