"""隔离运行本地多乐器转录；只交候选与原始证据，不作配器签收。"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import soundfile as sf

from audio_tools import AudioError, INPUT_FLAGS, _hash_file, _local, _probe, _run


def validate_candidates(report, expected_frames, input_hash, model_hash=None):
    """验证模型返回的真实输入和时界，不把格式有效提升成音乐正确。"""
    def reject(message):
        raise AudioError('note_backend_report', message)
    if not isinstance(report, dict) or report.get('status') != 'analysis_candidates_only':
        reject('本地后端没有返回已完成的候选报告')
    source = report.get('source', {})
    if source.get('sha256') != input_hash or source.get('frames') != expected_frames or source.get('sample_rate_hz') != 16000 or source.get('channels') != 1:
        reject('后端实际输入摘要或规格不匹配')
    notes = report.get('notes')
    if not isinstance(notes, list) or len(notes) > 50000:
        reject('候选音符列表缺失或超限')
    for note in notes:
        if not isinstance(note, dict) or type(note.get('is_drum')) is not bool:
            reject('候选打击属性无效')
        if any(type(note.get(key)) is not int or not lo <= note[key] <= hi for key, lo, hi in [('pitch', 0, 127), ('program', 0, 128), ('velocity', 1, 127)]):
            reject('候选音高/程序/力度无效')
        if note['program'] == 128 and not note['is_drum']:
            reject('程序128仅可作打击哨兵，不能是有调乐器')
        if any(type(note.get(key)) not in (int, float) or not math.isfinite(note[key]) for key in ('onset', 'offset')):
            reject('候选起止不是有限数')
        if not 0 <= note['onset'] < note['offset'] <= expected_frames / 16000:
            reject('候选越过实际输入窗口；不静默截断')
    runtime = report.get('runtime', {})
    loads = runtime.get('strict_weights', [])
    if len(loads) != 1 or loads[0].get('strict') is not True or loads[0].get('missing_keys') != [] or loads[0].get('unexpected_keys') != [] or type(loads[0].get('model_key_count')) is not int or loads[0]['model_key_count'] < 1 or loads[0]['model_key_count'] != loads[0].get('checkpoint_key_count'):
        reject('后端缺少严格权重匹配证据')
    if model_hash is not None and runtime.get('checkpoint_sha256') != model_hash:
        reject('后端模型摘要与显式配置不一致')
    return report


def analyze_notes(input_path, config_path, backend_python, out, start=0, seconds=8, prefix_seconds=.25):
    """本地文件短窗→显式隔离后端→候选/原始token；无下载或上传。"""
    if any(type(value) not in (int, float) or not math.isfinite(value) for value in (start, seconds, prefix_seconds)) or start < 0 or not .5 <= seconds <= 30 or prefix_seconds not in (0, .25):
        raise AudioError('note_window', '窗口须为非负起点及0.5–30秒，前置上下文仅支持0或0.25秒')
    path = _local(input_path)
    config = Path(config_path).resolve()
    output = Path(out).resolve()
    if output.exists():
        raise AudioError('output_exists', '分析输出已存在，禁止覆盖')
    if not config.is_file() or not Path(backend_python).is_file():
        raise AudioError('note_backend_missing', '须显式提供本地模型配置和隔离Python解释器')
    meta = _probe(path)
    if start + seconds > meta['container_duration_seconds'] + 1 / 16000:
        raise AudioError('note_window', '指定窗口超过音频长度，不补零冒充实际输入')
    frames = round(seconds * 16000)
    decode = ['ffmpeg', '-v', 'error', '-nostdin', *INPUT_FLAGS, '-i', str(path), '-ss', str(start), '-t', str(seconds), '-map', '0:a:0', '-vn', '-ar', '16000', '-ac', '1', '-f', 'f32le', 'pipe:1']
    wave = np.frombuffer(_run(decode).stdout, dtype='<f4')
    if len(wave) != frames or not np.isfinite(wave).all():
        raise AudioError('note_decode', '短窗未完整解码或包含非有限数')
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.notes-', dir=output.parent))
    try:
        prepared = stage / 'analysis_input.wav'
        sf.write(prepared, wave, 16000, subtype='FLOAT')
        worker = Path(__file__).with_name('yourmt3_local.py')
        argv = [os.path.abspath(backend_python), '-B', str(worker), '--config', str(config), '--input', str(prepared), '--out', str(stage / 'backend'), '--prefix-seconds', str(prefix_seconds)]
        try:
            process = subprocess.run(argv, capture_output=True, text=True, timeout=210, check=False)
        except subprocess.TimeoutExpired as exc:
            raise AudioError('note_backend_timeout', '本地转录超过总计210秒门禁') from exc
        (stage / 'backend.log').write_text(process.stdout + process.stderr, encoding='utf-8')
        if process.returncode:
            raise AudioError('note_backend_failed', '本地转录后端失败，未交付假候选', {'exit_code': process.returncode, 'log_directory': str(stage)})
        raw_path = stage / 'backend/0_analysis_input/analysis.json'
        report = json.loads(raw_path.read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
        config_data = json.loads(config.read_text())
        validate_candidates(report, frames, _hash_file(prepared), config_data['checkpoint_sha256'])
        result = {'status': '通过', 'status_scope': 'multitrack_note_candidates', 'source': {'file_name': path.name, 'file_sha256': _hash_file(path), 'format': meta},
                  'window': {'start_seconds': start, 'seconds': seconds, 'frames': frames, 'sample_rate_hz': 16000, 'channels': 1, 'prepared_file': 'analysis_input.wav', 'prepared_sha256': _hash_file(prepared)},
                  'backend_config_sha256': _hash_file(config), 'worker_source_sha256': _hash_file(worker), 'runtime': report['runtime'], 'padding': report['padding'],
                  'notes': report['notes'], 'excluded_candidates': report.get('excluded_candidates', []), 'left_censored_candidates': report.get('left_censored_candidates', []),
                  'decode_errors': report.get('decode_errors', {}), 'raw_directory': 'backend/0_analysis_input', 'commands': {'decode': decode, 'backend': argv},
                  'limits': report['limits'] + ['本命令通过只表示执行及数据链有效；控制失败、边缘敏感性与家族误判须另外审查', '没有生成MIDI、乐谱或新作品，不将参考候选直接当原创主题'],
                  'musical_validation': '未核准原配器、和弦或风格'}
        (stage / 'analysis.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        if output.exists():
            raise AudioError('output_exists', '分析期间输出路径已被占用，拒绝覆盖')
        os.rename(stage, output)
        return {'status': '通过', 'status_scope': result['status_scope'], 'out': str(output), 'analysis': str(output / 'analysis.json'), 'candidate_count': len(result['notes']), 'musical_validation': result['musical_validation']}
    except BaseException as exc:
        (stage / 'failure.json').write_text(json.dumps({'status': '失败', 'message': str(exc), 'scope': '未发布为分析交付'}, ensure_ascii=False) + '\n')
        raise
