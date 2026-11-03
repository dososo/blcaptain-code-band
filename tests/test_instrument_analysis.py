"""多标签候选契约；模拟后端不计为真实识别或模型能力验收。"""
import copy
import importlib
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
from audio_tools import AudioError, _hash_file


class InstrumentAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.mod = importlib.import_module('instrument_analysis')
        self.registry = importlib.import_module('register_instrument_backend')
        self.base = Path(tempfile.mkdtemp(prefix='band-instrument-test-'))
        self.source = self.base / '输入 $(禁止执行).wav'
        self.wave = (.03 * np.sin(np.arange(16000) * .03)).astype(np.float32)
        sf.write(self.source, self.wave, 32000, subtype='FLOAT')
        self.plan = self.base / '窗口.json'
        self.plan_data = {'source_sha256': _hash_file(self.source),
                          'windows': [{'window_id': 'a', 'start_seconds': 0, 'duration_seconds': .5}]}
        self.plan.write_text(json.dumps(self.plan_data))
        self.config = self.base / '后端.json'
        self.cfg = {'backend_id': 'panns_cnn10_cpu', 'checkpoint_sha256': self.registry.MODEL_SHA256,
                    'worker_sha256': _hash_file(ROOT / 'skills/code-band-bgm/scripts/panns_instruments.py'),
                    'source_sha256': self.registry.SOURCE_SHA256, 'dependency_sha256': {'test.py': 'f' * 64},
                    'versions': self.registry.VERSIONS}
        self.config.write_text(json.dumps(self.cfg))
        self.out = self.base / '候选'
        target = self.base / '基础Python'; target.write_text('仅契约测试')
        self.python = self.base / 'venv/bin/python'; self.python.parent.mkdir(parents=True); self.python.symlink_to(target)
        self.labels = [{'index': i, 'mid': '/test/' + str(i), 'label': '测试标签' + str(i)} for i in range(527)]

    def report(self, manifest):
        return {'analysis_scope': 'local_multi_instrument_candidates', 'physical_identity_confirmed': False,
                'composition_ready': False, 'score_semantics': self.mod.SCORE_SEMANTICS,
                'limitations': self.mod.LIMITATIONS, 'manual_listening': '未进行',
                'config_sha256': _hash_file(self.config),
                'source_sha256': manifest['source_sha256'], 'labels_sha256': self.registry.SOURCE_SHA256['class_labels_indices.csv'],
                'labels': copy.deepcopy(self.labels), 'runtime': {'checkpoint_sha256': self.cfg['checkpoint_sha256'],
                'worker_sha256': self.cfg['worker_sha256'], 'source_sha256': self.cfg['source_sha256'],
                'dependency_sha256': self.cfg['dependency_sha256'], 'versions': self.cfg['versions'],
                'device': 'cpu', 'dtype': 'float32', 'threads': 2, 'batch_size': 1, 'network_blocked': True,
                'weights_only': True, 'strict_state': True, 'model_key_count': 60, 'missing_keys': [],
                'unexpected_keys': [], 'peak_rss_bytes': 1000000, 'elapsed_seconds': 1.0,
                'parameters': self.registry.PARAMETERS},
                'windows': [{'window_id': w['window_id'], 'input_sha256': w['sha256'], 'frames': w['frames'],
                             'sample_rate_hz': 32000, 'channels': 1, 'scores': [0.001] * 527} for w in manifest['windows']]}

    def run_analysis(self, mutate=None, failure=None):
        def backend(argv, **kwargs):
            if failure == 'timeout':
                raise subprocess.TimeoutExpired(argv, 330, output='模拟超时日志')
            if failure:
                return SimpleNamespace(returncode=2, stdout='模拟失败', stderr='受控错误')
            manifest = json.loads(Path(argv[argv.index('--inputs') + 1]).read_text())
            folder = Path(argv[argv.index('--out') + 1]); folder.mkdir()
            report = self.report(manifest)
            if mutate:
                mutate(report, manifest)
            (folder / 'report.json').write_text(json.dumps(report))
            return SimpleNamespace(returncode=0, stdout='模拟契约，不是实际模型', stderr='')
        fake = SimpleNamespace(run=backend, TimeoutExpired=subprocess.TimeoutExpired)
        with patch.object(self.mod, 'validate_config', return_value=(self.cfg, {})), \
             patch.object(self.mod, 'read_labels', return_value=self.labels), \
             patch.object(self.mod, 'subprocess', fake):
            return self.mod.analyze_instruments(self.source, self.config, self.python, self.plan, self.out)

    def test_valid_candidates_keep_all_scores_without_claiming_identity(self):
        result = self.run_analysis()
        self.assertEqual(result['analysis_scope'], 'local_multi_instrument_candidates')
        self.assertFalse(result['composition_ready']); self.assertFalse(result['physical_identity_confirmed'])
        report = json.loads((self.out / 'analysis.json').read_text())
        self.assertEqual(len(report['windows'][0]['scores']), 527)
        self.assertEqual(report['windows'][0]['input_file'], 'analysis_inputs/000.wav')
        info = sf.info(self.out / report['windows'][0]['input_file'])
        self.assertEqual((info.samplerate, info.channels, info.frames, info.subtype), (32000, 1, 16000, 'FLOAT'))
        self.assertEqual(report['raw_report_sha256'], _hash_file(self.out / 'backend/report.json'))
        argv = report['commands']['backend']
        self.assertEqual(argv[0], os.path.abspath(self.python)); self.assertNotEqual(argv[0], str(self.python.resolve()))
        self.assertIn('pycache_prefix=', ' '.join(argv))

    def test_plan_additional_fields_cannot_override_validated_measurements(self):
        self.plan_data['windows'][0].update(scores=[1.0], frames=1, input_sha256='b' * 64)
        self.plan.write_text(json.dumps(self.plan_data))
        self.run_analysis()
        row = json.loads((self.out / 'analysis.json').read_text())['windows'][0]
        self.assertEqual(len(row['scores']), 527)
        self.assertEqual(row['frames'], 16000)
        self.assertNotEqual(row['input_sha256'], 'b' * 64)

    def test_wrong_plans_never_start_backend_or_publish(self):
        defects = [{'source_sha256': 'b' * 64}, {'windows': []},
                   {'windows': self.plan_data['windows'] * 2},
                   {'windows': [{'window_id': 'a', 'start_seconds': -1, 'duration_seconds': 1}]},
                   {'windows': [{'window_id': 'a', 'start_seconds': 0, 'duration_seconds': True}]},
                   {'windows': [{'window_id': 'a', 'start_seconds': 0, 'duration_seconds': 31}]},
                   {'windows': [{'window_id': 'a', 'start_seconds': 1, 'duration_seconds': .5}]},
                   {'windows': [{'window_id': str(i), 'start_seconds': 0, 'duration_seconds': 30} for i in range(5)]},
                   {'windows': [{'window_id': str(i), 'start_seconds': 0, 'duration_seconds': .5} for i in range(9)]}]
        for defect in defects:
            with self.subTest(defect=defect):
                data = {**self.plan_data, **defect}; self.plan.write_text(json.dumps(data))
                with patch.object(self.mod, 'validate_config', return_value=(self.cfg, {})), \
                     patch.object(self.mod, '_probe', return_value={'container_duration_seconds': .5}), \
                     patch.object(self.mod.subprocess, 'run', side_effect=AssertionError('无效计划不得执行')):
                    with self.assertRaises(AudioError):
                        self.mod.analyze_instruments(self.source, self.config, self.python, self.plan, self.out)
                self.assertFalse(self.out.exists())

    def test_bad_binding_labels_scores_or_frames_do_not_publish(self):
        mutations = [lambda r, m: r.update(source_sha256='b' * 64),
                     lambda r, m: r.update(labels_sha256='b' * 64),
                     lambda r, m: r['labels'][0].update(label='假标签'),
                     lambda r, m: r['windows'][0].update(window_id='不存在'),
                     lambda r, m: r['windows'][0].update(input_sha256='b' * 64),
                     lambda r, m: r['windows'][0].update(frames=15999),
                     lambda r, m: r['windows'][0].update(scores=[0.1] * 526),
                     lambda r, m: r['windows'][0]['scores'].__setitem__(0, float('nan')),
                     lambda r, m: r['windows'][0]['scores'].__setitem__(0, 1.1),
                     lambda r, m: r['runtime'].update(device='mps'),
                     lambda r, m: r['runtime'].update(worker_sha256='b' * 64),
                     lambda r, m: r.update(physical_identity_confirmed=True)]
        for index, mutation in enumerate(mutations):
            with self.subTest(index=index):
                self.out = self.base / ('坏输出' + str(index))
                self.labels = copy.deepcopy(self.labels)
                with self.assertRaises(AudioError): self.run_analysis(mutate=mutation)
                self.assertFalse(self.out.exists())

    def test_source_plan_or_config_changed_during_backend_blocks_publication(self):
        for file in (self.source, self.plan, self.config):
            with self.subTest(file=file):
                original = file.read_bytes()
                def changed(report, manifest): file.write_bytes(original + b' ')
                with self.assertRaises(AudioError): self.run_analysis(mutate=changed)
                self.assertFalse(self.out.exists()); file.write_bytes(original)

    def test_backend_failure_and_timeout_keep_logs_without_published_directory(self):
        for failure in ('failed', 'timeout'):
            with self.subTest(failure=failure):
                with self.assertRaises(AudioError): self.run_analysis(failure=failure)
                self.assertFalse(self.out.exists())
        self.assertTrue(list(self.base.glob('.instruments-*/failure.json')))

    def test_decode_error_log_rejected_even_with_exact_frames(self):
        decoded = SimpleNamespace(stdout=self.wave.astype('<f4').tobytes(), stderr=b'Error decoding stream')
        with patch.object(self.mod, 'validate_config', return_value=(self.cfg, {})), \
             patch.object(self.mod, '_probe', return_value={'container_duration_seconds': .5}), \
             patch.object(self.mod, '_run', return_value=decoded):
            with self.assertRaises(AudioError):
                self.mod.analyze_instruments(self.source, self.config, self.python, self.plan, self.out)
        self.assertFalse(self.out.exists())

    def test_existing_output_is_preserved(self):
        self.out.mkdir(); marker = self.out / '保留.txt'; marker.write_text('保留')
        with self.assertRaises(AudioError): self.run_analysis()
        self.assertEqual(marker.read_text(), '保留')

    def test_registered_dependency_changes_or_extra_files_are_rejected(self):
        self.cfg.update(checkpoint='model.pth', source_root='source', dependency_root='dependencies')
        self.config.write_text(json.dumps(self.cfg))
        for actual in ({'test.py': '0' * 64}, {**self.cfg['dependency_sha256'], '新增.py': 'b' * 64}):
            with self.subTest(actual=actual), patch.object(self.registry, 'check_assets', return_value=actual):
                with self.assertRaises(ValueError): self.registry.validate_config(self.config)
        folder = self.base / '依赖'; (folder / 'torchlibrosa').mkdir(parents=True)
        (folder / 'torchlibrosa/stft.py').write_text('仅依赖目录边界测试')
        (folder / '链接.py').symlink_to(self.source)
        with self.assertRaises(ValueError): self.registry.dependency_hashes(folder)

    def test_report_change_after_validation_never_publishes(self):
        original = self.mod.validate_instrument_report
        def validated(*args):
            result = original(*args)
            reports = list(self.base.glob('.instruments-*/backend/report.json'))
            self.assertEqual(len(reports), 1)
            reports[0].write_text('{}')
            return result
        with patch.object(self.mod, 'validate_instrument_report', side_effect=validated):
            with self.assertRaises(AudioError): self.run_analysis()
        self.assertFalse(self.out.exists())

    def test_registration_rejects_wrong_model_before_probing_interpreter(self):
        with patch.object(self.registry.subprocess, 'run', side_effect=AssertionError('错误权重不得探测环境')):
            with self.assertRaises(ValueError):
                self.registry.register_backend(self.source, self.base, self.base, self.python, self.base / '注册.json')
        self.assertFalse((self.base / '注册.json').exists())

    def test_imports_do_not_load_optional_models_or_torch(self):
        code = ('import sys;sys.path.insert(0,' + repr(str(ROOT / 'skills/code-band-bgm/scripts')) + ');'
                'import instrument_analysis,register_instrument_backend,panns_instruments;'
                'assert not any(n in sys.modules for n in ["torch","torchlibrosa","models","mlx","demucs"])')
        p = subprocess.run([sys.executable, '-B', '-c', code], capture_output=True, text=True, timeout=20)
        self.assertEqual(p.returncode, 0, p.stderr)


if __name__ == '__main__': unittest.main()
