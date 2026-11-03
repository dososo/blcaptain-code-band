"""交付门禁隔离回归；测试信号仅留在临时目录，不计为产品音乐。"""

import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "skills/code-band-bgm" / "scripts"))
import band
from score_tools import ScoreError, compile_score, digest, file_hash, load
from soundfont_inventory import inventory


class DeliveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.score = load(ROOT / "tests/fixtures/音频闭环乐谱.json")
        cls.cap = load(ROOT / "tests/fixtures/engine-capabilities-v1.json")
        cls.font = ROOT / "skills/code-band-bgm/assets/soundfonts/MuseScore_General.sf3"
        cls.mapping = inventory(cls.font)
        cls.sha = cls.mapping["source"]["sha256"]

    def fixture(self, temporary, score, cap=None):
        temporary = Path(temporary)
        score_path = temporary / "score.json"
        cap_path = temporary / "capabilities.json"
        score_path.write_text(json.dumps(score, ensure_ascii=False), encoding="utf-8")
        cap_path.write_text(json.dumps(cap or self.cap, ensure_ascii=False), encoding="utf-8")
        frames = score["render"]["target_file_frames"]
        signal = 0.05 * np.sin(np.arange(frames) * (2 * np.pi * 440 / 48000))
        audio = np.column_stack([signal, signal])
        args = SimpleNamespace(score=str(score_path), capabilities=str(cap_path),
                               soundfont=str(self.font), out=str(temporary / "work-v1"),
                               events=None, change=None, reference_analysis=None)
        return args, audio

    def test_reference_id_mismatch_never_publishes_work(self):
        score = copy.deepcopy(self.score)
        score["intent"]["reference_policy"].update(
            mode="abstract_features_only", source_ids=["expected_reference"])
        with tempfile.TemporaryDirectory() as temporary:
            args, audio = self.fixture(temporary, score)
            analysis = Path(temporary) / "reference.json"
            analysis.write_text(json.dumps({"status": "通过", "source": {
                "source_id": "different_reference", "file_sha256": "0" * 64}}), encoding="utf-8")
            args.reference_analysis = str(analysis)
            with patch.object(band, "render_events", side_effect=AssertionError("参考检查必须早于渲染")):
                with self.assertRaises(ScoreError) as raised:
                    band.render_work(args)
            self.assertEqual(raised.exception.details["field"], "reference/source_ids")
            self.assertFalse(Path(args.out).exists())
            self.assertFalse((Path(args.out) / "final.wav").exists())

    def reference_fixture(self, temporary, with_understanding=True):
        score = copy.deepcopy(self.score)
        score["intent"]["reference_policy"].update(mode="abstract_features_only", source_ids=["reference_test"])
        args, audio = self.fixture(temporary, score)
        temporary = Path(temporary)
        source = temporary / "reference.wav"
        band.sf.write(str(source), audio, 48000, subtype="PCM_24")
        evidence = temporary / "用户音乐描述.md"
        evidence.write_text("隔离测试的用户音乐说明：克制的短句、轻拍脉冲与稀疏和声；不是实际试听记录。", encoding="utf-8")
        analysis = {"status": "通过", "measurement_status": "通过", "analysis_scope": "objective_measurements_only",
                    "source": {"source_id": "reference_test", "file_sha256": file_hash(source),
                               "target_binding": {"relationship": "user_supplied_audio", "target_id": "隔离测试用户给出的文件", "input_file_sha256": file_hash(source)}},
                    "reference_understanding": {"status": "待语义证据"}}
        if with_understanding:
            analysis["reference_understanding"] = {
                "status": "证据已记录", "method": "user_description", "source_file_sha256": file_hash(source),
                "evidence_file": evidence.name, "evidence_sha256": file_hash(evidence),
                "dimensions": {key: {"observation": "来自测试用户描述，不是算法听辨", "implementation": "以当前声部编写留白与短句",
                                     "part_ids": [] if key == "structure_and_mood" else [score["parts"][0]["part_id"]]}
                               for key in ("timbre_and_space", "rhythm", "texture", "structure_and_mood")}}
        analysis_path = temporary / "reference.json"
        analysis_path.write_text(json.dumps(analysis, ensure_ascii=False), encoding="utf-8")
        args.reference_analysis = str(analysis_path)
        return args, audio, score, analysis

    def published_history(self, temporary, score, audio, analysis):
        parent = Path(temporary) / "已交付历史"
        parent.mkdir()
        frozen = compile_score(score, self.cap, self.sha, self.mapping)
        band.write_json(parent / "score.json", score)
        band.write_json(parent / "events.json", frozen)
        band.write_json(parent / "reference_analysis.json", analysis)
        band.sf.write(str(parent / "final.wav"), audio, 48000, subtype="PCM_24")
        band.write_json(parent / "manifest.json", {"technical_check": "通过", "score_sha256": digest(score),
                        "score_file_sha256": file_hash(parent / "score.json"), "events_sha256": digest(frozen),
                        "wav_file_sha256": file_hash(parent / "final.wav"),
                        "style_review": {"status": "需修改", "feedback": "用户已否决风格"}})
        return parent, frozen

    def test_first_reference_measurements_or_approval_boolean_cannot_start_render(self):
        for approved in (False, True):
            with self.subTest(approved=approved), tempfile.TemporaryDirectory() as temporary:
                args, _, _, analysis = self.reference_fixture(temporary, False)
                analysis["style_approved"] = approved
                Path(args.reference_analysis).write_text(json.dumps(analysis), encoding="utf-8")
                with patch.object(band, "render_events", side_effect=AssertionError("缺证据不能渲染")), \
                     patch.object(band, "compile_score", side_effect=AssertionError("缺证据不必编译")):
                    with self.assertRaises(ScoreError) as raised:
                        band.render_work(args)
                self.assertEqual(raised.exception.details["field"], "reference/understanding")
                self.assertFalse(Path(args.out).exists())
                self.assertFalse(list(Path(temporary).glob(".render-*")))

    def test_reference_evidence_binding_and_part_ids_are_checked_before_render(self):
        for defect, expected in (("source", "reference/source_file_sha256"), ("artifact", "reference/evidence_sha256"),
                                 ("part", "reference/dimensions/rhythm"), ("missing", "reference/evidence_file")):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as temporary:
                args, _, _, analysis = self.reference_fixture(temporary)
                review = analysis["reference_understanding"]
                if defect == "source":
                    review["source_file_sha256"] = "0" * 64
                elif defect == "artifact":
                    review["evidence_sha256"] = "0" * 64
                elif defect == "part":
                    review["dimensions"]["rhythm"]["part_ids"] = ["not_a_part"]
                else:
                    review["evidence_file"] = "不存在.md"
                Path(args.reference_analysis).write_text(json.dumps(analysis), encoding="utf-8")
                with patch.object(band, "render_events", side_effect=AssertionError("证据失效不能渲染")):
                    with self.assertRaises(ScoreError) as raised:
                        band.render_work(args)
                self.assertEqual(raised.exception.details["field"], expected)
                self.assertFalse(Path(args.out).exists())

    def test_valid_reference_keeps_evidence_and_never_claims_style_acceptance(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, audio, _, analysis = self.reference_fixture(temporary)
            with patch.object(band, "render_events", return_value=(audio, {"isolated_test": True})):
                result = band.render_work(args)
            manifest = load(Path(args.out) / "manifest.json")
            self.assertEqual(result["status_scope"], "audio_render_and_technical_checks")
            self.assertEqual(manifest["reference_understanding"]["status"], "证据已记录")
            self.assertEqual(manifest["style_review"]["status"], "待人工验收")
            copied = Path(args.out) / manifest["reference_understanding"]["evidence_file"]
            self.assertEqual(file_hash(copied), analysis["reference_understanding"]["evidence_sha256"])
            self.assertIn(copied.name, manifest["files"])
            self.assertIn("未试听", manifest["human_review"])

    def test_new_compiled_events_do_not_bypass_reference_understanding(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, _, score, _ = self.reference_fixture(temporary, False)
            events = Path(temporary) / "events.json"
            band.write_json(events, compile_score(score, self.cap, self.sha, self.mapping))
            args.events = str(events)
            with patch.object(band, "render_events", side_effect=AssertionError("新编译事件不能绕过门禁")):
                with self.assertRaises(ScoreError) as raised:
                    band.render_work(args)
            self.assertEqual(raised.exception.details["field"], "reference/understanding")

    def test_unverified_associated_music_never_starts_render(self):
        for relation in (None, "associated_audio_unverified"):
            with self.subTest(relation=relation), tempfile.TemporaryDirectory() as temporary:
                args, _, _, analysis = self.reference_fixture(temporary)
                analysis["source"]["target_binding"]["relationship"] = relation
                Path(args.reference_analysis).write_text(json.dumps(analysis), encoding="utf-8")
                with patch.object(band, "render_events", side_effect=AssertionError("目标未核实不能渲染")):
                    with self.assertRaises(ScoreError) as raised:
                        band.render_work(args)
                self.assertEqual(raised.exception.details["field"], "reference/target_binding")
                self.assertFalse(Path(args.out).exists())

    def test_semantic_similarity_cannot_verify_associated_recording(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, _, score, analysis = self.reference_fixture(temporary)
            evidence = Path(temporary) / "身份证据.json"
            evidence.write_text(json.dumps({"status": "matched", "method": "audio_embedding_similarity", "cosine": .99,
                "source_file_sha256": analysis["source"]["file_sha256"], "target_id": "目标视频"}), encoding="utf-8")
            analysis["source"]["target_binding"].update(relationship="associated_audio_verified", target_id="目标视频",
                evidence_file=evidence.name, evidence_sha256=file_hash(evidence))
            Path(args.reference_analysis).write_text(json.dumps(analysis), encoding="utf-8")
            with self.assertRaises(ScoreError) as raised:
                band.reference_context(args, score)
            self.assertEqual(raised.exception.details["field"], "reference/target_match")

    def test_actual_target_track_evidence_is_copied_into_delivery(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, audio, _, analysis = self.reference_fixture(temporary)
            evidence = Path(temporary) / "取得证据.json"
            evidence.write_text(json.dumps({"status": "acquired", "target_id": "目标视频", "source_file_sha256": analysis["source"]["file_sha256"],
                "method": "isolated_test", "record": {"说明": "隔离测试取得记录，不是实际网页取证"}}), encoding="utf-8")
            analysis["source"]["target_binding"].update(relationship="target_media_audio", target_id="目标视频",
                evidence_file=evidence.name, evidence_sha256=file_hash(evidence))
            Path(args.reference_analysis).write_text(json.dumps(analysis), encoding="utf-8")
            with patch.object(band, "render_events", return_value=(audio, {"isolated_test": True})):
                band.render_work(args)
            manifest = load(Path(args.out) / "manifest.json")
            target = manifest["reference_source"]["target_binding"]
            self.assertEqual(file_hash(Path(args.out) / target["evidence_file"]), file_hash(evidence))
            self.assertIn(target["evidence_file"], manifest["files"])

    def test_history_retains_target_evidence_path_without_resetting_rejection(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, audio, score, analysis = self.reference_fixture(temporary)
            evidence = Path(temporary) / '目标证据.json'
            evidence.write_text('{"说明":"隔离历史取得记录，不是实际网页取证"}', encoding='utf-8')
            analysis['source']['target_binding'].update(relationship='target_media_audio',
                evidence_file=evidence.name, evidence_sha256=file_hash(evidence))
            Path(args.reference_analysis).write_text(json.dumps(analysis), encoding='utf-8')
            parent, _ = self.published_history(temporary, score, audio, analysis)
            args.events = str(parent / 'events.json')
            context = band.reference_context(args, score)
            self.assertEqual(context['target_evidence_path'], evidence.resolve())
            self.assertEqual(context['style_review']['status'], '需修改')
            evidence.write_text('已改变，不得发布过期绑定', encoding='utf-8')
            with self.assertRaises(ScoreError) as raised:
                band.reference_context(args, score)
            self.assertEqual(raised.exception.details['field'], 'reference/target_evidence')

    def test_target_evidence_content_must_bind_same_media_and_audio(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, _, score, analysis = self.reference_fixture(temporary)
            evidence = Path(temporary) / "取得证据.json"
            analysis["source"]["target_binding"].update(relationship="target_media_audio", target_id="目标视频", evidence_file=evidence.name)
            for target, sha in [("另一个视频", analysis["source"]["file_sha256"]), ("目标视频", "0" * 64)]:
                evidence.write_text(json.dumps({"status": "acquired", "target_id": target, "source_file_sha256": sha, "method": "test", "record": {"说明": "测试"}}))
                analysis["source"]["target_binding"]["evidence_sha256"] = file_hash(evidence)
                Path(args.reference_analysis).write_text(json.dumps(analysis))
                with self.assertRaises(ScoreError) as raised:
                    band.reference_context(args, score)
                self.assertEqual(raised.exception.details["field"], "reference/target_evidence")

    def test_associated_match_needs_real_time_spans_and_target_audio_digest(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, _, score, analysis = self.reference_fixture(temporary)
            evidence = Path(temporary) / "匹配.json"
            analysis["source"]["target_binding"].update(relationship="associated_audio_verified", target_id="目标视频", evidence_file=evidence.name)
            match = {"status": "matched", "target_id": "目标视频", "source_file_sha256": analysis["source"]["file_sha256"], "method": "waveform_alignment"}
            for invalid in [match, dict(match, target_file_sha256="1" * 64, alignment=[]), dict(match, target_file_sha256="1" * 64, alignment=[{"source_start_seconds": 0, "target_start_seconds": 2, "duration_seconds": 0}])]:
                evidence.write_text(json.dumps(invalid))
                analysis["source"]["target_binding"]["evidence_sha256"] = file_hash(evidence)
                Path(args.reference_analysis).write_text(json.dumps(analysis))
                with self.assertRaises(ScoreError) as raised:
                    band.reference_context(args, score)
                self.assertEqual(raised.exception.details["field"], "reference/target_match")

    def test_supported_review_sources_do_not_depend_on_one_model_vendor(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, _, score, analysis = self.reference_fixture(temporary)
            for method in ("host_audio_perception", "local_audio_model", "external_audio_model", "user_description"):
                analysis["reference_understanding"]["method"] = method
                if method == "local_audio_model":
                    report = Path(temporary) / "本地原始模型报告.json"
                    report.write_text(json.dumps({"status": "通过", "analysis_scope": "local_audio_language_model_estimates",
                        "source": {"file_sha256": analysis["source"]["file_sha256"]},
                        "segments": [{"description": "隔离测试模型输出，非实际音乐听辨"}]}), encoding="utf-8")
                    analysis["reference_understanding"].update(evidence_file=report.name, evidence_sha256=file_hash(report))
                Path(args.reference_analysis).write_text(json.dumps(analysis), encoding="utf-8")
                context = band.reference_context(args, score)
                self.assertEqual(context["understanding"]["method"], method)
                self.assertEqual(context["style_review"]["status"], "待人工验收")

    def test_local_model_report_cannot_bind_another_recording_or_manual_wrapper(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, _, score, analysis = self.reference_fixture(temporary)
            analysis["reference_understanding"]["method"] = "local_audio_model"
            evidence = Path(temporary) / "模型原始报告.json"
            for document, field in [
                ({"status": "通过", "analysis_scope": "local_audio_language_model_estimates", "source": {"file_sha256": "0" * 64}, "segments": [{"description": "另一录音的描述"}]}, "reference/local_model_source"),
                ({"说明": "手写包装不能替代原始本地模型报告"}, "reference/local_model_report")]:
                evidence.write_text(json.dumps(document), encoding="utf-8")
                analysis["reference_understanding"].update(evidence_file=evidence.name, evidence_sha256=file_hash(evidence))
                Path(args.reference_analysis).write_text(json.dumps(analysis), encoding="utf-8")
                with self.assertRaises(ScoreError) as raised:
                    band.reference_context(args, score)
                self.assertEqual(raised.exception.details["field"], field)

    def test_evidence_changed_during_render_never_publishes_stale_binding(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, audio, _, analysis = self.reference_fixture(temporary)
            evidence = Path(temporary) / analysis["reference_understanding"]["evidence_file"]
            def render(*unused):
                evidence.write_text("渲染期间已改变", encoding="utf-8")
                return audio, {"isolated_test": True}
            with patch.object(band, "render_events", side_effect=render):
                with self.assertRaises(ScoreError) as raised:
                    band.render_work(args)
            self.assertEqual(raised.exception.details["field"], "reference/evidence_sha256")
            self.assertFalse(Path(args.out).exists())

    def test_published_reference_replay_keeps_historical_rejection_without_recomposition(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, audio, score, analysis = self.reference_fixture(temporary, False)
            parent, frozen = self.published_history(temporary, score, audio, analysis)
            original_manifest_hash = file_hash(parent / "manifest.json")
            args.events, args.reference_analysis = str(parent / "events.json"), None
            with patch.object(band, "compile_score", side_effect=AssertionError("历史重演不能重作")), \
                 patch.object(band, "render_events", return_value=(audio, {"isolated_test": True})):
                result = band.render_work(args)
            self.assertEqual(result["reference_understanding"]["status"], "历史未核准")
            self.assertEqual(result["style_review"]["status"], "需修改")
            self.assertEqual(load(Path(args.out) / "events.json"), frozen)
            self.assertEqual(file_hash(parent / "manifest.json"), original_manifest_hash)

    def test_changed_published_wav_cannot_claim_historical_exception(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, audio, score, analysis = self.reference_fixture(temporary, False)
            parent, _ = self.published_history(temporary, score, audio, analysis)
            with (parent / "final.wav").open("ab") as stream:
                stream.write(b"changed")
            args.events = str(parent / "events.json")
            with patch.object(band, "render_events", side_effect=AssertionError("历史证据失效不能重演")):
                with self.assertRaises(ScoreError) as raised:
                    band.render_work(args)
            self.assertEqual(raised.exception.details["field"], "reference/history")

    def test_optional_describe_is_lazy_and_has_structured_dependency_failure(self):
        module = ModuleType("local_audio_review")
        module.describe_audio = lambda source, model: {"status": "通过", "input": source, "model": model}
        # describe 在加载模型前先过信号/相位门禁（门禁本身见 test_reference_gate）；这里只测懒加载，门禁打桩为通过
        gate = ({"rms_dbfs": -20.0}, {"mono_models_allowed": True, "signal": "通过", "phase": "通过"})
        with tempfile.TemporaryDirectory() as temporary:
            out = str(Path(temporary) / "描述.json")
            argv = ["band.py", "describe", "--input", "测试.m4a", "--model", "本地模型", "--out", out]
            with patch.dict(sys.modules, {"local_audio_review": module}), patch("reference_gate.require_mono_models", return_value=gate), \
                 patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()):
                self.assertEqual(band.main(), 0)
            self.assertEqual(load(out)["model"], "本地模型")
            self.assertEqual(load(out)["signal_gate"]["signal"], "通过")
        with patch.dict(sys.modules, {"local_audio_review": None}), \
             patch.object(sys, "argv", ["band.py", "describe", "--input", "测试.m4a", "--model", "本地模型"]), redirect_stdout(io.StringIO()) as output:
            self.assertEqual(band.main(), 2)
        self.assertEqual(json.loads(output.getvalue())["code"], "dependency_missing")
        with patch.dict(sys.modules, {"local_audio_review": None}), \
             patch.object(sys, "argv", ["band.py", "validate", "--score", str(ROOT / "tests/fixtures/音频闭环乐谱.json")]), redirect_stdout(io.StringIO()):
            self.assertEqual(band.main(), 0)

    def test_frozen_replay_keeps_events_and_never_calls_compiler(self):
        frozen = compile_score(self.score, self.cap, self.sha, self.mapping)
        cap = copy.deepcopy(self.cap)
        cap["observed_at"] = "2026-10-31T00:00:00+00:00"
        with tempfile.TemporaryDirectory() as temporary:
            args, audio = self.fixture(temporary, self.score, cap)
            events_path = Path(temporary) / "events.json"
            events_path.write_text(json.dumps(frozen, ensure_ascii=False), encoding="utf-8")
            args.events = str(events_path)

            def render(document, soundfont):
                self.assertEqual(document, frozen)
                return audio, {"isolated_delivery_test": True}

            with patch.object(band, "compile_score", side_effect=AssertionError("冻结重演不能重新编译")), \
                 patch.object(band, "render_events", side_effect=render):
                result = band.render_work(args)
            self.assertEqual(result["status"], "通过")
            self.assertEqual(load(Path(args.out) / "events.json"), frozen)


if __name__ == "__main__":
    unittest.main()
