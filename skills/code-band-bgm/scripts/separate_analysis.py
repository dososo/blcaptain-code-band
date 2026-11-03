"""本地参考短窗→隔离CPU分离→真实估计轨与证据；不核准物理乐器身份。"""
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import soundfile as sf

from audio_tools import AudioError, INPUT_FLAGS, _hash_file, _local, _probe, _run

SOURCES = ('drums', 'bass', 'other', 'vocals', 'guitar', 'piano')


def validate_separation(report, directory, frames, input_hash, model_hash, prepared):
    """核实际文件、绑定与重构，格式通过仍只是分离估计。"""
    def reject(message):
        raise AudioError('separation_backend_report', message)
    if not isinstance(report, dict) or report.get('status') != 'separation_estimates_only':
        reject('后端没有完成估计报告')
    if report.get('ready_for_composition') is not False or report.get('labels_are_physical_identities') is not False:
        reject('后端不得把轨名或执行通过提升为配器核准')
    source, runtime = report.get('source'), report.get('runtime')
    if not isinstance(source, dict) or any(source.get(k) != v for k, v in
            [('sha256', input_hash), ('frames', frames), ('sample_rate_hz', 44100), ('channels', 2)]):
        reject('实际输入摘要/帧数/规格不匹配')
    if not isinstance(runtime, dict) or any(runtime.get(k) != v for k, v in
            [('checkpoint_sha256', model_hash), ('device', 'cpu'), ('dtype', 'float32'), ('torch_threads', 2),
             ('network_blocked', True), ('strict_state', True), ('missing_keys', []), ('unexpected_keys', [])]):
        reject('后端模型、CPU或严格载入证据不匹配')
    if type(runtime['torch_threads']) is not int or runtime['network_blocked'] is not True or runtime['strict_state'] is not True:
        reject('CPU线程及断网/严格载入标记类型无效')
    if any(type(runtime.get(key)) is not int or runtime[key] != 525 for key in ('model_key_count', 'checkpoint_key_count')):
        reject('固定6源模型须严格匹配525个模型/权重键')
    peak = runtime.get('peak_rss_bytes')
    if type(peak) is not int or not 0 < peak <= 3 * 1024**3:
        reject('缺少有效的3GiB进程内存门禁证据')
    outputs = report.get('outputs')
    if not isinstance(outputs, dict) or set(outputs) != set((*SOURCES, 'reconstruction', 'residual')):
        reject('必须提供六估计轨及重构/残差')
    arrays = {}
    for name, item in outputs.items():
        if not isinstance(item, dict) or item.get('file') != name + '.wav':
            reject('输出必须是后端目录内的固定普通WAV文件')
        path = Path(directory) / item['file']
        if path.is_symlink() or not path.is_file() or _hash_file(path) != item.get('sha256'):
            reject('输出文件缺失、符号链接或SHA256不匹配')
        info = sf.info(path)
        if info.frames != frames or info.samplerate != 44100 or info.channels != 2 or info.subtype != 'FLOAT':
            reject('估计轨必须是精确帧数的44.1k双声道FLOAT32')
        value, _ = sf.read(path, dtype='float32', always_2d=True)
        if not np.isfinite(value).all():
            reject('估计轨包含NaN/Inf')
        arrays[name] = value
    original, _ = sf.read(prepared, dtype='float32', always_2d=True)
    reconstructed = sum(arrays[name].astype(np.float64) for name in SOURCES)
    if not np.allclose(reconstructed, arrays['reconstruction'], atol=1e-6, rtol=1e-6) or not np.allclose(
            original.astype(np.float64) - arrays['reconstruction'], arrays['residual'], atol=1e-6, rtol=1e-6):
        reject('实际重构/残差文件与六轨、输入不一致')
    return report


def separate_reference(input, model_config, backend_python, out, start=0, seconds=8):
    """显式本地模型/独立Python；0.5–30秒，分析副本不当作品交付。"""
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in (start, seconds)) or start < 0 or not .5 <= seconds <= 30:
        raise AudioError('separation_window', '窗口须为非负起点及0.5–30秒')
    path = _local(input)
    config, output = Path(model_config).resolve(), Path(out).resolve()
    if output.exists():
        raise AudioError('output_exists', '分离输出已存在，禁止覆盖')
    if not config.is_file() or not Path(backend_python).is_file():
        raise AudioError('separation_backend_missing', '须显式提供本地配置和独立Python')
    config_data = json.loads(config.read_text())
    if not isinstance(config_data, dict) or not isinstance(config_data.get('checkpoint_sha256'), str):
        raise AudioError('separation_config', '模型配置必须包含固定checkpoint_sha256')
    source_hash, config_hash = _hash_file(path), _hash_file(config)
    meta = _probe(path)
    if start + seconds > meta['container_duration_seconds'] + 1 / 44100:
        raise AudioError('separation_window', '窗口超过实际音频长度，不补零冒充输入')
    frames = round(seconds * 44100)
    decode = ['ffmpeg', '-v', 'error', '-nostdin', *INPUT_FLAGS, '-i', str(path), '-ss', str(start),
              '-t', str(seconds), '-map', '0:a:0', '-vn', '-ar', '44100', '-ac', '2', '-f', 'f32le', 'pipe:1']
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.separate-', dir=output.parent))
    try:
        decoded = _run(decode)
        if decoded.stderr.strip():
            text = decoded.stderr.decode(errors='replace') if isinstance(decoded.stderr, bytes) else decoded.stderr
            (stage / 'decode.log').write_text(text, encoding='utf-8')
            raise AudioError('separation_decode', '解码器报告错误，即使帧数完整也拒绝分析')
        wave = np.frombuffer(decoded.stdout, dtype='<f4')
        if len(wave) != frames * 2 or not np.isfinite(wave).all():
            raise AudioError('separation_decode', '未完整解码精确44.1k双声道有限分析窗')
        prepared = stage / 'analysis_input.wav'
        sf.write(prepared, wave.reshape(-1, 2), 44100, subtype='FLOAT')
        prepared_hash = _hash_file(prepared)
        worker = Path(__file__).with_name('demucs_local.py')
        cache = stage / 'private-pycache'
        cache.mkdir()
        argv = [os.path.abspath(backend_python), '-X', 'pycache_prefix=' + str(cache), '-B', str(worker), '--config', str(config),
                '--input', str(prepared), '--out', str(stage / 'backend')]
        try:
            process = subprocess.run(argv, capture_output=True, text=True, timeout=210, check=False)
        except subprocess.TimeoutExpired as exc:
            text = exc.stdout or ''
            (stage / 'backend.log').write_text(text.decode(errors='replace') if isinstance(text, bytes) else text)
            raise AudioError('separation_backend_timeout', '分离后端超过210秒总门禁') from exc
        (stage / 'backend.log').write_text(process.stdout + process.stderr, encoding='utf-8')
        if process.returncode:
            raise AudioError('separation_backend_failed', '分离后端失败，未发布假目录', {'exit_code': process.returncode, 'log_directory': str(stage)})
        if _hash_file(path) != source_hash or _hash_file(prepared) != prepared_hash or _hash_file(config) != config_hash:
            raise AudioError('separation_source_changed', '分析期间源、配置或准备输入发生变化，拒绝发布')
        raw_path = stage / 'backend/report.json'
        raw_bytes = raw_path.read_bytes()
        raw_hash = hashlib.sha256(raw_bytes).hexdigest()
        report = json.loads(raw_bytes, parse_constant=lambda _: (_ for _ in ()).throw(ValueError('非有限JSON')))
        validate_separation(report, stage / 'backend', frames, prepared_hash, config_data['checkpoint_sha256'], prepared)
        result = {'status': '通过', 'status_scope': 'source_separation_estimates', 'ready_for_composition': False,
                  'source': {'file_name': path.name, 'file_sha256': source_hash, 'format': meta},
                  'window': {'start_seconds': start, 'seconds': seconds, 'frames': frames, 'sample_rate_hz': 44100,
                             'channels': 2, 'sample_format': 'float32', 'prepared_file': 'analysis_input.wav', 'prepared_sha256': prepared_hash},
                  'backend_config_sha256': config_hash, 'worker_source_sha256': _hash_file(worker),
                  'runtime': report['runtime'], 'outputs': {k: {**v, 'file': 'backend/' + v['file']} for k, v in report['outputs'].items()},
                  'raw_report': 'backend/report.json', 'raw_report_sha256': raw_hash,
                  'commands': {'decode': decode, 'backend': argv},
                  'limits': ['估计轨名不是物理乐器身份；FM也可进入piano。', '源能量比例/残差不是准确率。',
                             '只作本地分析，不自动再分发参考片段或送入原创主题。'], 'musical_validation': '未核准原配器、和弦或风格'}
        result['commands_sha256'] = hashlib.sha256(json.dumps(result['commands'], ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        (stage / 'analysis.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
        if _hash_file(path) != source_hash or _hash_file(prepared) != prepared_hash or _hash_file(config) != config_hash:
            raise AudioError('separation_source_changed', '发布前源、配置或准备输入发生变化，拒绝发布')
        if raw_path.is_symlink() or not raw_path.is_file() or _hash_file(raw_path) != raw_hash:
            raise AudioError('separation_outputs_changed', '发布前原始报告发生变化，拒绝发布')
        for item in report['outputs'].values():
            file = stage / 'backend' / item['file']
            if file.is_symlink() or not file.is_file() or _hash_file(file) != item['sha256']:
                raise AudioError('separation_outputs_changed', '发布前估计轨/重构/残差发生变化，拒绝发布')
        if output.exists():
            raise AudioError('output_exists', '分析期间输出路径已被占用，拒绝覆盖')
        os.rename(stage, output)
        return {'status': '通过', 'status_scope': result['status_scope'], 'ready_for_composition': False,
                'out': str(output), 'analysis': str(output / 'analysis.json')}
    except BaseException as exc:
        (stage / 'failure.json').write_text(json.dumps({'status': '失败', 'message': str(exc), 'scope': '未发布为分析交付'}, ensure_ascii=False) + '\n')
        raise
