"""项目入口生命周期与冲突保护；不触碰真实宿主配置。"""
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "skills/code-band-bgm/scripts"))
from install_skill import operate


class InstallTests(unittest.TestCase):
    def test_two_hosts_one_source_and_reversible_removal(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp).resolve(); source = ROOT / "skills/code-band-bgm"
            r = operate("install", "both", p, source)
            self.assertEqual(len(r["changed_entries"]), 2)
            for e in r["entries"]: self.assertEqual(Path(e["entry"]).resolve(), source)
            self.assertEqual(operate("install", "both", p, source)["changed_entries"], [])
            self.assertEqual(len(operate("remove", "both", p, source)["changed_entries"]), 2)
            self.assertTrue((source / "SKILL.md").is_file())

    def test_conflict_prevents_all_mutations(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp).resolve(); source = ROOT / "skills/code-band-bgm"
            conflict = p / ".claude/skills/code-band-bgm"
            conflict.parent.mkdir(parents=True); conflict.write_text("用户已有技能")
            with self.assertRaises(ValueError): operate("install", "both", p, source)
            self.assertFalse((p / ".agents/skills/code-band-bgm").exists())
            self.assertEqual(conflict.read_text(), "用户已有技能")

    def test_parent_link_cannot_install_outside_project(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp).resolve(); project = base / "project"; other = base / "other"
            project.mkdir(); other.mkdir(); (project / ".agents").symlink_to(other, target_is_directory=True)
            with self.assertRaises(ValueError): operate("install", "codex", project, ROOT / "skills/code-band-bgm")
            self.assertFalse((other / "skills/code-band-bgm").exists())

    def test_removal_does_not_delete_other_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp).resolve(); other = p / "other"; other.mkdir()
            entry = p / ".agents/skills/code-band-bgm"; entry.parent.mkdir(parents=True); entry.symlink_to(other, target_is_directory=True)
            with self.assertRaises(ValueError): operate("remove", "codex", p, ROOT / "skills/code-band-bgm")
            self.assertTrue(entry.is_symlink())
            self.assertTrue(other.is_dir())
