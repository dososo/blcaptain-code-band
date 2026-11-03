"""资产登记只封存显式文件，失败不写假配置；不加载模型。"""
from pathlib import Path
import hashlib
import json
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'skills/code-band-bgm/scripts'))
from register_note_backend import register
from yourmt3_local import Session


class RegisterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source, self.deps = self.root/'source', self.root/'deps'
        self.source.mkdir(); self.deps.mkdir()
        (self.source/'module.py').write_text('value = 1\n')
        (self.deps/'package.py').write_text('value = 2\n')
        (self.deps/'__pycache__').mkdir()
        (self.deps/'__pycache__/ignored.pyc').write_bytes(b'cache')
        self.model = self.root/'model'; self.model.write_bytes(b'known model fixture')
        self.sha = hashlib.sha256(self.model.read_bytes()).hexdigest()
        self.output = self.root/'config.json'

    def tearDown(self):
        self.temp.cleanup()

    def run_register(self, sha=None):
        return register(self.source, self.deps, self.model, sha or self.sha, 'a'*40, self.output)

    def test_relative_paths_and_only_actual_static_file_hashes(self):
        value = self.run_register()
        self.assertEqual(value['source_root'], 'source')
        self.assertEqual(set(value['target_dependency_sha256']), {'package.py'})
        self.assertEqual(json.loads(self.output.read_text()), value)

    def test_wrong_model_hash_does_not_publish(self):
        with self.assertRaises(ValueError): self.run_register('b'*64)
        self.assertFalse(self.output.exists())

    def test_existing_config_is_untouched(self):
        self.output.write_text('用户原文件')
        with self.assertRaises(ValueError): self.run_register()
        self.assertEqual(self.output.read_text(), '用户原文件')

    def test_file_symlink_is_rejected(self):
        (self.deps/'linked.py').symlink_to(self.source/'module.py')
        with self.assertRaises(ValueError): self.run_register()
        self.assertFalse(self.output.exists())

    def test_source_directory_symlink_is_rejected(self):
        (self.source/'unlocked_package').symlink_to(self.deps, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, '符号链接'): self.run_register()
        self.assertFalse(self.output.exists())

    def test_dependency_directory_symlink_is_rejected(self):
        (self.deps/'unlocked_package').symlink_to(self.source, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, '符号链接'): self.run_register()
        self.assertFalse(self.output.exists())

    def test_worker_rejects_directory_links_added_after_registration(self):
        self.run_register()
        for root, target in ((self.source, self.deps), (self.deps, self.source)):
            link = root/'unlocked_package'
            link.symlink_to(target, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, '符号链接'): Session(self.output)
            link.unlink()
