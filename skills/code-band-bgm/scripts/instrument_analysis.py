"""参考固定窗口→隔离CPU多标签候选；不自动识别核准或绑定乐器。"""
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import soundfile as sf

from audio_tools import AudioError, INPUT_FLAGS, _hash_file, _local, _probe, _run
from register_instrument_backend import PARAMETERS, SCORE_SEMANTICS, LIMITATIONS, read_labels, validate_config


def _json(path):
    def reject(_): raise ValueError('JSON不能含非有限数')
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'), parse_constant=reject)
    except (OSError, ValueError) as exc:
        raise AudioError('instrument_json', '候选计划、配置或报告不是有效本地JSON') from exc


def _write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def validate_instrument_report(report, manifest, cfg, labels):
    """验证实际候选绑定和完整分数；不以候选数目或排序证明物理来源。"""
    def need(value, message):
        if not value: raise AudioError('instrument_backend_report', message)
    need(isinstance(report, dict) and report.get('analysis_scope') == 'local_multi_instrument_candidates', '后端没有提供完整多标签候选报告')
    need(report.get('physical_identity_confirmed') is False and report.get('composition_ready') is False, '后端不能将候选提升为原乐器或创作核准')
    need(report.get('score_semantics') == SCORE_SEMANTICS, '标签分数必须明确为未校准候选')
    need(report.get('source_sha256') == manifest['source_sha256'], '候选报告与实际参考源摘要不符')
    need(report.get('config_sha256') == manifest['config_sha256'], '候选报告与实际配置摘要不符')
    need(report.get('labels_sha256') == cfg['source_sha256']['class_labels_indices.csv'] and report.get('labels') == labels, '实际527标签或标签表摘要不匹配')
    need(report.get('limitations') == LIMITATIONS and report.get('manual_listening') == '未进行', '报告缺少已知反证或夸大人工听感状态')
    runtime = report.get('runtime')
    need(isinstance(runtime, dict), '缺少真实运行身份')
    expected = {key: cfg[key] for key in ('checkpoint_sha256', 'worker_sha256', 'source_sha256', 'dependency_sha256', 'versions')}
    expected.update(device='cpu', dtype='float32', threads=2, batch_size=1, model_key_count=60, missing_keys=[], unexpected_keys=[])
    expected['parameters'] = PARAMETERS
    need(all(runtime.get(k) == v for k, v in expected.items()), '模型、源码、依赖、CPU或严格键身份不匹配')
    need(all(runtime.get(k) is True for k in ('network_blocked', 'weights_only', 'strict_state')), '后端必须禁网、仅权重加载并严格匹配')
    need(all(type(runtime.get(k)) is int for k in ('threads', 'batch_size', 'model_key_count')), '运行计数字段必须为整数')
    memory, elapsed = runtime.get('peak_rss_bytes'), runtime.get('elapsed_seconds')
    need(type(memory) is int and 0 < memory <= 3 * 1024**3, '缺少有效的3GiB内存门禁证据')
    need(type(elapsed) in (int, float) and math.isfinite(elapsed) and 0 <= elapsed <= 300, '运行耗时不符合300秒门禁')
    windows = report.get('windows')
    need(isinstance(windows, list) and len(windows) == len(manifest['windows']), '候选窗口数量不匹配')
    for row, item in zip(windows, manifest['windows']):
        need(isinstance(row, dict), '候选窗口必须是对象')
        expected = {'window_id': item['window_id'], 'input_sha256': item['sha256'], 'frames': item['frames'], 'sample_rate_hz': 32000, 'channels': 1}
        need(all(row.get(k) == v for k, v in expected.items()) and all(type(row.get(k)) is int for k in ('frames', 'sample_rate_hz', 'channels')), '候选窗口标识、输入摘要或精确帧数不符')
        scores = row.get('scores')
        need(isinstance(scores, list) and len(scores) == 527 and all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1 for v in scores), '必须保留527个有限且介于零一的原始标签分数')
    return report


def analyze_instruments(input, model_config, backend_python, window_plan, out):
    """显式本地计划和后端；保留分析副本，不将参考副本作为新作。"""
    source, config, plan_path, output = _local(input), Path(model_config).resolve(), Path(window_plan).resolve(), Path(out).resolve()
    if output.exists():
        raise AudioError('output_exists', '候选输出目录已存在，禁止覆盖')
    if not config.is_file() or not plan_path.is_file() or not Path(backend_python).is_file():
        raise AudioError('instrument_backend_missing', '须显式提供本地配置、计划及独立Python')
    hashes = {source: _hash_file(source), config: _hash_file(config), plan_path: _hash_file(plan_path)}
    plan = _json(plan_path)
    if not isinstance(plan, dict) or plan.get('source_sha256') != hashes[source]:
        raise AudioError('instrument_plan', '窗口计划须绑定本次真实参考源SHA256')
    windows = plan.get('windows')
    if not isinstance(windows, list) or not 1 <= len(windows) <= 8:
        raise AudioError('instrument_plan', '计划必须包含一至八个窗口')
    ids, total = set(), 0
    for item in windows:
        if not isinstance(item, dict) or not isinstance(item.get('window_id'), str) or not item['window_id'].strip() or item['window_id'] in ids:
            raise AudioError('instrument_plan', '窗口标识必须非空且唯一')
        ids.add(item['window_id'])
        start, duration = item.get('start_seconds'), item.get('duration_seconds')
        try:
            valid = (type(start) in (int, float) and type(duration) in (int, float) and start >= 0
                     and .5 <= duration <= 30 and math.isfinite(start + duration))
        except OverflowError:
            valid = False
        if not valid:
            raise AudioError('instrument_plan', '窗口须为有限非负起点及0.5至30秒时长')
        total += duration
    if total > 120:
        raise AudioError('instrument_plan', '窗口总长不能超过120秒')
    meta = _probe(source)
    if any(w['start_seconds'] + w['duration_seconds'] > meta['container_duration_seconds'] + 1 / 32000 for w in windows):
        raise AudioError('instrument_plan', '窗口超过真实参考时长，不补零冒充输入')
    try:
        cfg, assets = validate_config(config)
        labels = read_labels(assets.get('source_root'))
    except (ValueError, OSError, KeyError, TypeError) as exc:
        raise AudioError('instrument_config', '后端配置或本地资产未通过核验：' + str(exc)) from exc
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.instruments-', dir=output.parent))
    try:
        (stage / 'analysis_inputs').mkdir()
        (stage / 'window_plan.json').write_bytes(plan_path.read_bytes())
        manifest = {'source_sha256': hashes[source], 'config_sha256': hashes[config], 'windows': []}
        commands = []
        for index, window in enumerate(windows):
            frames = round(window['duration_seconds'] * 32000)
            argv = ['ffmpeg', '-v', 'error', '-nostdin', *INPUT_FLAGS, '-i', str(source), '-ss', str(window['start_seconds']),
                    '-t', str(window['duration_seconds']), '-map', '0:a:0', '-vn', '-ar', '32000', '-ac', '1', '-f', 'f32le', 'pipe:1']
            decoded = _run(argv)
            wave = np.frombuffer(decoded.stdout, dtype='<f4')
            if decoded.stderr.strip() or len(wave) != frames or not np.isfinite(wave).all():
                raise AudioError('instrument_decode', '未完整解码精确32k有限窗口，或解码器报告错误')
            name = 'analysis_inputs/' + str(index).zfill(3) + '.wav'
            sf.write(stage / name, wave, 32000, subtype='FLOAT')
            manifest['windows'].append({'window_id': window['window_id'], 'input_file': name,
                                        'sha256': _hash_file(stage / name), 'frames': frames})
            commands.append(argv)
        _write(stage / 'prepared_inputs.json', manifest)
        manifest_hash = _hash_file(stage / 'prepared_inputs.json')
        worker = Path(__file__).with_name('panns_instruments.py')
        cache = stage / 'private_pycache'; cache.mkdir()
        argv = [os.path.abspath(backend_python), '-B', '-X', 'pycache_prefix=' + str(cache), str(worker),
                '--config', str(config), '--inputs', str(stage / 'prepared_inputs.json'), '--out', str(stage / 'backend')]
        try:
            process = subprocess.run(argv, capture_output=True, text=True, timeout=330, check=False)
        except subprocess.TimeoutExpired as exc:
            log = exc.stdout or ''
            (stage / 'backend.log').write_text(log.decode(errors='replace') if isinstance(log, bytes) else log, encoding='utf-8')
            raise AudioError('instrument_backend_timeout', '候选后端超过330秒门禁') from exc
        (stage / 'backend.log').write_text(process.stdout + process.stderr, encoding='utf-8')
        if process.returncode:
            raise AudioError('instrument_backend_failed', '候选后端失败，未发布假目录', {'exit_code': process.returncode, 'log_directory': str(stage)})
        if any(_hash_file(path) != sha for path, sha in hashes.items()):
            raise AudioError('instrument_source_changed', '执行期间参考源、窗口计划或配置改变')
        report_path = stage / 'backend/report.json'
        if report_path.is_symlink() or not report_path.is_file():
            raise AudioError('instrument_backend_report', '后端没有提供普通本地候选报告')
        report_hash = _hash_file(report_path)
        report = validate_instrument_report(_json(report_path), manifest, cfg, labels)
        for item in manifest['windows']:
            file = stage / item['input_file']
            info = sf.info(file)
            if file.is_symlink() or _hash_file(file) != item['sha256'] or (info.samplerate, info.channels, info.frames, info.subtype) != (32000, 1, item['frames'], 'FLOAT'):
                raise AudioError('instrument_input_changed', '分析副本规格或摘要改变')
        result = {'status': '通过', 'analysis_scope': 'local_multi_instrument_candidates', 'physical_identity_confirmed': False,
                  'composition_ready': False, 'manual_listening': '未进行', 'score_semantics': SCORE_SEMANTICS,
                  'source': {'file_name': source.name, 'file_sha256': hashes[source], 'format': meta,
                             'usage': '仅本地参考分析副本，不能作为新作或原创采样'},
                  'window_plan_sha256': hashes[plan_path], 'backend_config_sha256': hashes[config],
                  'labels': labels, 'labels_sha256': report['labels_sha256'], 'runtime': report['runtime'],
                  'windows': [{**row, 'start_seconds': window['start_seconds'], 'duration_seconds': window['duration_seconds'], 'input_file': item['input_file']}
                              for row, window, item in zip(report['windows'], windows, manifest['windows'])],
                  'raw_report': 'backend/report.json', 'raw_report_sha256': report_hash, 'limitations': LIMITATIONS,
                  'commands': {'decode': commands, 'backend': argv}}
        _write(stage / 'analysis.json', result)
        validate_config(config)
        if (any(_hash_file(path) != sha for path, sha in hashes.items()) or _hash_file(stage / 'window_plan.json') != hashes[plan_path]
                or _hash_file(stage / 'prepared_inputs.json') != manifest_hash or _hash_file(report_path) != report_hash
                or any(_hash_file(stage / w['input_file']) != w['sha256'] for w in manifest['windows'])):
            raise AudioError('instrument_source_changed', '发布前源、配置、计划、分析副本或报告摘要改变')
        if output.exists():
            raise AudioError('output_exists', '执行期间输出路径已被占用，拒绝覆盖')
        os.rename(stage, output)
        return {'status': '通过', 'analysis_scope': result['analysis_scope'], 'physical_identity_confirmed': False,
                'composition_ready': False, 'out': str(output), 'analysis': str(output / 'analysis.json')}
    except BaseException as exc:
        _write(stage / 'failure.json', {'status': '失败', 'message': str(exc), 'scope': '未发布为候选分析交付'})
        raise
