"""可选上游固定实现的自动掩码补丁与恢复。"""
import importlib.util
import os
from pathlib import Path
import sys
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'skills/code-band-bgm/scripts'))
import optional_ace_mask_fix as fix

try:
    import torch
except ImportError:
    torch = None
SOURCE = os.environ.get('ACE_MASK_SOURCE')


@unittest.skipUnless(torch is not None and SOURCE and Path(SOURCE).is_file(), '未提供本地ACE源码与Torch，不代表修复通过')
class AceMaskFixTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location('fixed_version_masks', SOURCE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.host = module.ConditioningMaskMixin()
        self.host.device, self.host.sample_rate = 'cpu', 48000

    def invoke(self, modes, repaint):
        return self.host._build_chunk_masks_and_src_latents(
            batch_size=2, max_latent_length=8, instructions=['x'] * 2, audio_code_hints=[None] * 2,
            target_wavs=torch.ones((2, 2, 15360)), target_latents=torch.zeros((2, 8, 64)),
            repainting_start=[.04, .08] if repaint else None,
            repainting_end=[.12, .20] if repaint else None,
            silence_latent_tiled=torch.zeros((8, 64)), chunk_mask_modes=modes, task_type='text2music')

    def test_auto_explicit_repaint_and_restore(self):
        before = self.invoke(['auto', 'explicit'], True)
        self.assertEqual(before[0].dtype, torch.bool)
        with fix.corrected_auto_mask(self.host):
            after = self.invoke(['auto', 'explicit'], True)
            self.assertEqual(after[0].tolist(), [[2.] * 8, [0., 0., 1., 1., 1., 0., 0., 0.]])
            self.assertEqual(after[0].dtype, torch.float32)
            for a, b in zip(before[1:], after[1:]):
                self.assertTrue(torch.equal(a, b) if isinstance(a, torch.Tensor) else a == b)
        self.assertNotIn(fix.METHOD, self.host.__dict__)
        self.assertTrue(torch.equal(self.invoke(['auto', 'explicit'], True)[0], before[0]))

    def test_explicit_unchanged_and_exception_restores(self):
        before = self.invoke(['explicit', 'explicit'], True)
        with self.assertRaisesRegex(RuntimeError, '中断'):
            with fix.corrected_auto_mask(self.host):
                after = self.invoke(['explicit', 'explicit'], True)
                self.assertEqual(after[0].dtype, torch.bool)
                self.assertTrue(torch.equal(before[0], after[0]))
                raise RuntimeError('中断')
        self.assertNotIn(fix.METHOD, self.host.__dict__)

    def test_unknown_source_rejected(self):
        with mock.patch.object(fix, 'SUPPORTED_SOURCE_SHA256', '0' * 64):
            with self.assertRaisesRegex(ValueError, '版本不匹配'):
                with fix.corrected_auto_mask(self.host):
                    self.fail('不得套用未知源码')
        self.assertNotIn(fix.METHOD, self.host.__dict__)

    def test_nested_patch_is_rejected_and_outer_scope_restores(self):
        with fix.corrected_auto_mask(self.host):
            with self.assertRaisesRegex(ValueError, '嵌套'):
                with fix.corrected_auto_mask(self.host):
                    self.fail('不得重复包装')
            self.assertEqual(self.invoke(['auto', 'explicit'], False)[0][0].tolist(), [2.] * 8)
        self.assertNotIn(fix.METHOD, self.host.__dict__)
