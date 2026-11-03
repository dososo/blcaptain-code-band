"""角色证据只验证来源、文件与限制，不把结构通过当音乐理解正确。"""
import copy
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'skills/code-band-bgm/scripts'))


class RoleEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.module = importlib.import_module('role_evidence')
        self.directory = Path(tempfile.mkdtemp(prefix='band-role-evidence-'))
        self.measurement = self.directory / '测量.json'
        self.measurement.write_text('{"测量": "仅为测试证据"}', encoding='utf-8')
        self.source_hash = 'a' * 64
        self.report = {
            'analysis_scope': 'local_role_measurement_evidence',
            'source': {'file_sha256': self.source_hash},
            'observations': {role: {
                'observation': '存在局部活动候选，不能确认物理乐器。',
                'limitations': ['测量不等于人工试听或风格验收。'],
                'evidence_ids': ['measure_1'],
            } for role in ('drums', 'bass', 'harmonic_rhythm', 'foreground')},
            'evidence': [{'evidence_id': 'measure_1', 'path': self.measurement.name,
                          'sha256': hashlib.sha256(self.measurement.read_bytes()).hexdigest(),
                          'kind': 'measurement'}],
            'windows': [{'start_seconds': 60, 'duration_seconds': 8}],
            'composition_ready': False,
            'manual_listening': '未进行',
        }
        self.package = self.directory / '角色证据.json'

    def validate(self, report=None, source_hash=None):
        self.package.write_text(json.dumps(self.report if report is None else report,
                                          ensure_ascii=False), encoding='utf-8')
        return self.module.validate_role_evidence(self.package, self.source_hash if source_hash is None else source_hash)

    def test_valid_report_is_returned_without_promoting_acceptance(self):
        original = copy.deepcopy(self.report)
        self.assertEqual(self.validate(), original)
        self.assertEqual(self.report, original)
        self.report['manual_listening'] = '待人工验收'
        self.assertEqual(self.validate()['manual_listening'], '待人工验收')

    def test_wrong_source_or_scope_is_rejected(self):
        for section, key, value in [('source', 'file_sha256', 'b' * 64),
                                    ('source', 'file_sha256', '非摘要'),
                                    ('root', 'analysis_scope', '自动理解通过')]:
            with self.subTest(key=key, value=value):
                report = copy.deepcopy(self.report)
                (report if section == 'root' else report[section])[key] = value
                with self.assertRaises(ValueError):
                    self.validate(report)
        with self.assertRaises(ValueError):
            self.validate(source_hash='无效')

    def test_each_role_needs_observation_limitations_and_existing_evidence(self):
        for role in self.report['observations']:
            for key, value in [('observation', ''), ('limitations', []), ('limitations', ['']),
                               ('evidence_ids', []), ('evidence_ids', ['不存在']), ('evidence_ids', [1])]:
                with self.subTest(role=role, key=key, value=value):
                    report = copy.deepcopy(self.report)
                    report['observations'][role][key] = value
                    with self.assertRaises(ValueError):
                        self.validate(report)
            report = copy.deepcopy(self.report)
            del report['observations'][role]
            with self.assertRaises(ValueError):
                self.validate(report)

    def test_evidence_requires_unique_ids_and_known_kind(self):
        for evidence in [[], [None], [{**self.report['evidence'][0], 'kind': '模型已识别'}],
                         [self.report['evidence'][0], self.report['evidence'][0]]]:
            with self.subTest(evidence=evidence):
                report = copy.deepcopy(self.report)
                report['evidence'] = evidence
                with self.assertRaises(ValueError):
                    self.validate(report)
        for kind in ('source_separation', 'controlled_render', 'official_comparison'):
            self.report['evidence'][0]['kind'] = kind
            self.validate()

    def test_missing_empty_changed_and_absolute_evidence_are_rejected(self):
        for mode in ('missing', 'empty', 'changed', 'absolute', 'directory'):
            with self.subTest(mode=mode):
                report = copy.deepcopy(self.report)
                item = report['evidence'][0]
                if mode == 'missing':
                    item['path'] = '缺少.json'
                elif mode == 'empty':
                    path = self.directory / '空.json'
                    path.write_bytes(b'')
                    item.update(path=path.name, sha256=hashlib.sha256(b'').hexdigest())
                elif mode == 'changed':
                    item['sha256'] = '0' * 64
                elif mode == 'absolute':
                    item['path'] = str(self.measurement)
                else:
                    item['path'] = '.'
                with self.assertRaises(ValueError):
                    self.validate(report)

    def test_relative_paths_are_resolved_from_package_not_working_directory(self):
        nested = self.directory / 'package'
        nested.mkdir()
        self.package = nested / '角色证据.json'
        self.report['evidence'][0]['path'] = '../' + self.measurement.name
        self.validate()

    def test_copy_preserves_input_and_keeps_published_references_readable(self):
        self.validate()
        original = self.package.read_bytes()
        out = self.directory / '作品'
        copied = self.module.copy_role_evidence(self.package, self.source_hash, out)
        self.assertEqual(copied, out / 'reference_evidence.json')
        published = self.module.validate_role_evidence(copied, self.source_hash)
        self.assertEqual(self.package.read_bytes(), original)
        self.assertEqual(self.measurement.read_text(encoding='utf-8'), '{"测量": "仅为测试证据"}')
        item = published['evidence'][0]
        self.assertTrue(item['path'].startswith('reference_role_sources/'))
        self.assertEqual((out / item['path']).read_bytes(), self.measurement.read_bytes())
        self.assertFalse(published['composition_ready'])

    def test_copy_never_overwrites_existing_evidence_or_sources(self):
        self.validate()
        for existing in ('reference_evidence.json', 'reference_role_sources'):
            with self.subTest(existing=existing):
                out = self.directory / existing.replace('.', '_')
                out.mkdir()
                marker = out / existing
                if existing.endswith('.json'):
                    marker.write_text('保留', encoding='utf-8')
                else:
                    marker.mkdir()
                    (marker / '保留.txt').write_text('保留', encoding='utf-8')
                with self.assertRaises(ValueError):
                    self.module.copy_role_evidence(self.package, self.source_hash, out)
                self.assertEqual((marker if marker.is_file() else marker / '保留.txt').read_text(encoding='utf-8'), '保留')

    def test_windows_must_be_finite_real_nonempty_intervals(self):
        for windows in [[], [None], [{'start_seconds': -1, 'duration_seconds': 1}],
                        [{'start_seconds': True, 'duration_seconds': 1}],
                        [{'start_seconds': 0, 'duration_seconds': False}],
                        [{'start_seconds': 0, 'duration_seconds': 0}],
                        [{'start_seconds': float('inf'), 'duration_seconds': 1}],
                        [{'start_seconds': 0, 'duration_seconds': float('nan')}]]:
            with self.subTest(windows=windows):
                report = copy.deepcopy(self.report)
                report['windows'] = windows
                with self.assertRaises(ValueError):
                    self.validate(report)

    def test_evidence_does_not_claim_automatic_composition_or_listening_acceptance(self):
        for key, value in [('composition_ready', True), ('composition_ready', 0),
                           ('manual_listening', '已试听通过'), ('manual_listening', None)]:
            with self.subTest(key=key, value=value):
                report = copy.deepcopy(self.report)
                report[key] = value
                with self.assertRaises(ValueError):
                    self.validate(report)

    def test_bad_json_and_missing_package_raise_chinese_value_errors(self):
        for value in ('[]', '{无效', '{"x": NaN}'):
            self.package.write_text(value, encoding='utf-8')
            with self.assertRaises(ValueError) as error:
                self.module.validate_role_evidence(self.package, self.source_hash)
            self.assertRegex(str(error.exception), '[\u4e00-\u9fff]')
        with self.assertRaises(ValueError):
            self.module.validate_role_evidence(self.directory / '缺少.json', self.source_hash)

    def test_import_does_not_load_optional_audio_models(self):
        code = ('import sys;sys.path.insert(0,' + repr(str(ROOT / 'skills/code-band-bgm/scripts')) + ');'
                'import role_evidence;assert not any(n in sys.modules for n in '
                '["torch","mlx","demucs","transformers","numpy"])')
        result = subprocess.run([sys.executable, '-B', '-c', code], capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_band_accepts_role_evidence_without_claiming_style_acceptance(self):
        import band
        from types import SimpleNamespace
        self.validate()
        score = json.loads((ROOT / 'tests/fixtures/音频闭环乐谱.json').read_text(encoding='utf-8'))
        score['intent']['reference_policy'].update(mode='abstract_features_only', source_ids=['role_fixture'])
        analysis = {
            'status': '通过', 'measurement_status': '通过',
            'source': {'source_id': 'role_fixture', 'file_sha256': self.source_hash,
                       'target_binding': {'relationship': 'user_supplied_audio', 'target_id': '本地测试',
                                          'input_file_sha256': self.source_hash}},
            'reference_understanding': {
                'status': '证据已记录', 'method': 'local_role_analysis',
                'source_file_sha256': self.source_hash, 'evidence_file': self.package.name,
                'evidence_sha256': hashlib.sha256(self.package.read_bytes()).hexdigest(),
                'dimensions': {dimension: {
                    'observation': '隔离契约测试的局部角色观察，不是实际试听或风格核准。',
                    'implementation': '只验证声部映射，不在测试中创作或渲染。', 'part_ids': ['piano'],
                } for dimension in ('timbre_and_space', 'rhythm', 'texture', 'structure_and_mood')},
            },
        }
        reference = self.directory / '参考分析.json'
        reference.write_text(json.dumps(analysis, ensure_ascii=False), encoding='utf-8')
        result = band.reference_context(SimpleNamespace(events=None, reference_analysis=str(reference)), score)
        self.assertEqual(result['understanding']['method'], 'local_role_analysis')
        self.assertEqual(result['style_review']['status'], '待人工验收')
        self.assertEqual(result['evidence_path'], self.package.resolve())
        self.assertFalse(self.module.validate_role_evidence(self.package, self.source_hash)['composition_ready'])

    def test_band_publishing_rebases_files_updates_hash_and_survives_stage_move(self):
        import band
        self.validate()
        original = self.package.read_bytes()
        expected = hashlib.sha256(original).hexdigest()
        reference = {'evidence_path': self.package.resolve(),
                     'analysis': {'source': {'file_sha256': self.source_hash}}}
        holder = {'method': 'local_role_analysis', 'evidence_file': self.package.name, 'evidence_sha256': expected}
        stage = self.directory / 'stage'
        files = band.publish_role_evidence(reference, stage, holder)
        final = self.directory / 'published'
        stage.rename(final)
        published = final / holder['evidence_file']
        self.assertEqual(holder['input_evidence_sha256'], expected)
        self.assertNotEqual(holder['evidence_sha256'], expected)
        self.assertEqual(holder['evidence_sha256'], hashlib.sha256(published.read_bytes()).hexdigest())
        self.assertEqual(self.package.read_bytes(), original)
        self.assertTrue(all((final / name).is_file() for name in files))
        report = self.module.validate_role_evidence(published, self.source_hash)
        self.assertFalse(report['composition_ready'])
        self.assertEqual(report['manual_listening'], '未进行')

    def test_band_publishing_rejects_changed_package_or_evidence(self):
        import band
        from score_tools import ScoreError
        for changed in ('package', 'measurement'):
            with self.subTest(changed=changed):
                self.validate()
                expected = hashlib.sha256(self.package.read_bytes()).hexdigest()
                reference = {'evidence_path': self.package.resolve(),
                             'analysis': {'source': {'file_sha256': self.source_hash}}}
                holder = {'method': 'local_role_analysis', 'evidence_sha256': expected}
                if changed == 'package':
                    self.package.write_text('{}', encoding='utf-8')
                else:
                    self.measurement.write_text('已变化', encoding='utf-8')
                stage = self.directory / ('stage_' + changed)
                with self.assertRaises(ScoreError):
                    band.publish_role_evidence(reference, stage, holder)
                self.assertFalse((stage / 'reference_evidence.json').exists())

    def test_band_publishing_rejects_copy_content_drift(self):
        import band
        from score_tools import ScoreError
        from unittest.mock import patch
        self.validate()
        reference = {'evidence_path': self.package.resolve(),
                     'analysis': {'source': {'file_sha256': self.source_hash}}}
        holder = {'method': 'local_role_analysis',
                  'evidence_sha256': hashlib.sha256(self.package.read_bytes()).hexdigest()}
        stage = self.directory / 'stage'
        def changed_copy(reader, writer):
            writer.write(b'changed during copy')
        with patch.object(self.module.shutil, 'copyfileobj', side_effect=changed_copy):
            with self.assertRaises(ScoreError):
                band.publish_role_evidence(reference, stage, holder)
        self.assertFalse((stage / 'reference_evidence.json').exists())


if __name__ == '__main__':
    unittest.main()
