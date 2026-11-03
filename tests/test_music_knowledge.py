"""知识查询需保留适用边界与来源，不能执行资料或伪造验收。"""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "skills/code-band-bgm/scripts"))
from music_knowledge import load_library, query_knowledge


class MusicKnowledgeTests(unittest.TestCase):
    def test_index_is_small_and_does_not_mark_capabilities_approved(self):
        index = query_knowledge()
        data, sha = load_library()
        self.assertEqual(index["library_sha256"], sha)
        self.assertEqual(index["status_scope"], "knowledge_selection")
        self.assertNotIn("sources", index)
        self.assertNotIn("action", index["rules"][0])
        self.assertEqual(index["rule_count"], len(data["rules"]))

    def test_union_deduplicates_and_keeps_exact_source_and_caveats(self):
        data, _ = load_library()
        rule = data["rules"][0]
        topic = rule["topics"][0]
        got = query_knowledge([topic, topic], [rule["id"], rule["id"]])
        expected = [r for r in data["rules"] if topic in r["topics"] or r["id"] == rule["id"]]
        self.assertEqual(got["rules"], expected)
        self.assertEqual({r["id"] for r in got["sources"]},
                         {s for r in expected for s in r["source_ids"]})
        self.assertTrue(all(r["counterexample"] and r["limits"] for r in got["rules"]))

    def test_unknown_selectors_fail_instead_of_guessing(self):
        for topics, ids in [(["不存在"], []), ([], ["不是规则"]), (["../密钥"], [])]:
            with self.assertRaises(ValueError):
                query_knowledge(topics, ids)

    def test_bad_library_refuses_dangling_sources_topics_or_duplicate_ids(self):
        data, _ = load_library()
        changes = [lambda d: d["rules"][0].update(source_ids=["不存在"]),
                   lambda d: d["rules"][0].update(topics=["不存在"]),
                   lambda d: d["sources"].append(d["sources"][0]),
                   lambda d: d["rules"][0].update(counterexample=""),
                   lambda d: d.update(schema_version="2.0.0")]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "库.json"
            for mutate in changes:
                bad = copy.deepcopy(data)
                mutate(bad)
                path.write_text(json.dumps(bad, ensure_ascii=False))
                with self.assertRaises(ValueError):
                    load_library(path)

    def test_standalone_pack_has_no_workspace_dependency(self):
        import shutil
        with tempfile.TemporaryDirectory() as folder:
            skill = Path(folder) / "独立技能"
            (skill / "scripts").mkdir(parents=True)
            (skill / "assets/knowledge").mkdir(parents=True)
            for relative in ("scripts/music_knowledge.py", "assets/knowledge/音乐方法库.json"):
                shutil.copyfile(ROOT / "skills/code-band-bgm" / relative, skill / relative)
            result = subprocess.run([sys.executable, "-c",
                "from music_knowledge import query_knowledge; import json; print(json.dumps(query_knowledge()))"],
                cwd=skill / "scripts", capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)["library_sha256"], query_knowledge()["library_sha256"])

    def test_cli_from_other_cwd_preserves_files_and_literal_paths(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / "研究 ; $(不要执行).json"
            args = [sys.executable, str(ROOT / "skills/code-band-bgm/scripts/band.py"),
                    "knowledge", "--topic", "音色重建", "--out", str(out)]
            first = subprocess.run(args, cwd=folder, capture_output=True, text=True)
            self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
            before = out.read_bytes()
            again = subprocess.run(args, cwd=folder, capture_output=True, text=True)
            self.assertEqual(again.returncode, 2)
            self.assertEqual(before, out.read_bytes())
            self.assertEqual(json.loads(before)["status_scope"], "knowledge_selection")
            args[args.index("--topic")+1] = "不存在"
            args[-1] = str(Path(folder) / "不应产生.json")
            failed = subprocess.run(args, cwd=folder, capture_output=True, text=True)
            self.assertEqual(failed.returncode, 2)
            self.assertFalse(Path(args[-1]).exists())


if __name__ == "__main__":
    unittest.main()
