"""可选本地CLAP辅助对照；测量、模型校准、风格验收分别报告。"""
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess

MODEL_SHA256 = '1cd3c601bc4afe0fa87be3de4c13dd2cfadd249fac1e29acf74a9b296c3219bb'
CONFIG_SHA256 = '9efb9557bc804f2ca6e394486af2e45dfed0b18554909735a99c6220b84e4288'
PROCESSOR_SHA256 = '9739f58296aa6f9ac18008fd0150fb2649bc554985fbde86d0a4041c882ac753'
RATE, WINDOW_SECONDS, HOP_SECONDS, SEED = 48000, 10, 5, 20261031


def file_hash(path):
    checksum = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            checksum.update(block)
    return checksum.hexdigest()


def _probe(path):
    from audio_tools import INPUT_FLAGS, LIMITS
    if not isinstance(path, (str, Path)) or not str(path).strip():
        raise ValueError('音频路径必须是非空本地路径')
    path = Path(path).resolve()
    if not path.is_file() or path.suffix.lower() in {'.m3u', '.m3u8', '.pls', '.xspf', '.asx', '.cue'}:
        raise ValueError('输入必须是本地音频文件，不接受URL或播放列表')
    if path.stat().st_size > LIMITS['max_bytes']:
        raise ValueError('单份输入超过本地音频分析100MB上限')
    with path.open('rb') as stream:
        header = stream.read(4096).lstrip().lower()
    if header.startswith((b'#extm3u', b'[playlist]', b'<?xml', b'<asx', b'ffconcat')):
        raise ValueError('不接受播放列表或间接资源清单')
    command = ['ffprobe', '-v', 'error', *INPUT_FLAGS, '-select_streams', 'a:0',
               '-show_entries', 'stream=channels:format=duration,format_name', '-of', 'json', str(path)]
    try:
        data = json.loads(subprocess.run(command, check=True, capture_output=True, timeout=30).stdout)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise ValueError('音频探测失败或超时，未生成对照结果') from error
    if not data.get('streams') or not isinstance(data.get('format'), dict):
        raise ValueError('输入没有可探测音频流')
    channels = data['streams'][0]['channels']
    duration = float(data['format']['duration'])
    if channels not in (1, 2) or not math.isfinite(duration) or duration < WINDOW_SECONDS:
        raise ValueError('只验证单/双声道、至少10秒、时长可知的音频')
    if any(name in data['format']['format_name'] for name in ('hls', 'dash', 'concat')):
        raise ValueError('不接受间接媒体或播放列表')
    return {'path': str(path), 'channels': channels, 'declared_seconds': duration,
            'file_sha256': file_hash(path)}


def _decode(metadata, duration_limit, tempo=None):
    from audio_tools import INPUT_FLAGS
    import numpy as np
    command = ['ffmpeg', '-v', 'error', '-nostdin', *INPUT_FLAGS, '-i', metadata['path'],
               '-map', '0:a:0', '-t', str(duration_limit)]
    if tempo is not None:
        command += ['-af', f'atempo={tempo}']
    command += ['-ar', str(RATE), '-ac', str(metadata['channels']), '-f', 'f32le', '-']
    try:
        output = subprocess.run(command, check=True, capture_output=True, timeout=60).stdout
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise ValueError('音频解码失败或超时，未生成对照结果') from error
    audio = np.frombuffer(output, dtype='<f4').copy()
    if metadata['channels'] == 2:
        audio = audio.reshape(-1, 2).mean(axis=1)
    if not np.isfinite(audio).all() or len(audio) < RATE * WINDOW_SECONDS:
        raise ValueError('音频解码不足10秒或包含非有限值')
    return audio, command


def _distribution(values):
    import numpy as np
    return dict(zip(['min', 'q25', 'median', 'q75', 'max'],
                    map(float, np.quantile(values, [0, .25, .5, .75, 1]))),
                mean=float(np.mean(values)), count=int(np.size(values)))


def compare_style(inputs_path, model_path, out_path, batch_size=4, rms_target=0.1,
                  max_total_audio_seconds=600, max_windows=128):
    """先校准再比较；失败返回诊断，绝不返回风格排名或同风格判决。"""
    if any(not isinstance(path, (str, Path)) or not str(path).strip() for path in (inputs_path, model_path, out_path)):
        raise ValueError('配置、模型与输出路径必须是非空本地路径')
    model_path, out_path = Path(model_path).resolve(), Path(out_path).resolve()
    if out_path.exists():
        raise ValueError('输出目录已存在，不覆盖已有结果')
    if batch_size not in range(1, 5) or not 0 < max_total_audio_seconds <= 600 or not 1 <= max_windows <= 128:
        raise ValueError('批量上限4、音频总预算上限600秒、短窗预算上限128')
    if rms_target is not None and not 0 < rms_target < 1:
        raise ValueError('RMS目标必须介于0和1之间，或明确为None保留原幅度')
    for name, expected in [('pytorch_model.bin', MODEL_SHA256), ('config.json', CONFIG_SHA256),
                            ('preprocessor_config.json', PROCESSOR_SHA256)]:
        path = model_path / name
        if not path.is_file() or file_hash(path) != expected:
            raise ValueError('只支持已校准的本地laion/clap-htsat-unfused固定版本，模型/配置摘要不符：' + name)
    config = json.loads(Path(inputs_path).read_text(encoding='utf-8'))
    if not isinstance(config, dict) or not isinstance(config.get('sources'), list) or len(config['sources']) < 2:
        raise ValueError('输入JSON必须包含至少两份sources')
    if len(config['sources']) > max_windows or len(config['sources']) * WINDOW_SECONDS > max_total_audio_seconds:
        raise ValueError('即使每份仅10秒也会超预算，拒绝探测过多输入')
    if any(not isinstance(source, dict) or not isinstance(source.get('source_id'), str) or
           not source['source_id'].strip() or not isinstance(source.get('path'), str) or not source['path'].strip()
           for source in config['sources']):
        raise ValueError('每份来源必须是包含非空source_id和path文本的JSON对象')
    ids = [source['source_id'] for source in config['sources']]
    if not all(isinstance(item, str) and item for item in ids) or len(ids) != len(set(ids)):
        raise ValueError('来源ID必须是非空且唯一的文本')
    reference_id = config.get('reference_source_id')
    if not isinstance(reference_id, str) or not reference_id.strip():
        raise ValueError('必须提供非空reference_source_id文本')
    if reference_id not in ids:
        raise ValueError('参考来源ID不存在')
    known_text = config.get('speech_control_text')
    if config.get('speech_control_owned') is not True or not isinstance(known_text, str) or not known_text.strip():
        raise ValueError('必须明确提供自有已知语音对照及其原文；不能自动把参考音乐当语音对照')
    if not isinstance(config.get('speech_control_path'), str) or not config['speech_control_path'].strip():
        raise ValueError('必须提供非空speech_control_path本地路径文本')
    metadata, declared_seconds = [], 0
    for source in config['sources']:
        item = dict(_probe(source['path']), source_id=source['source_id'])
        declared_seconds += item['declared_seconds']
        if declared_seconds > max_total_audio_seconds:
            raise ValueError('音频总时长超过预算，不静默截短')
        metadata.append(item)
    speech_metadata = _probe(config['speech_control_path'])
    if speech_metadata['declared_seconds'] > 60 or speech_metadata['file_sha256'] in {item['file_sha256'] for item in metadata}:
        raise ValueError('语音对照须为独立自有录音，时长10至60秒，不能复用参考或候选文件')

    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN'] = '1'
    import numpy as np
    import torch
    import torchaudio
    import transformers
    from transformers import ClapFeatureExtractor, ClapModel
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    def scale_window(clip):
        rms = float(np.sqrt(np.mean(clip.astype(np.float64) ** 2)))
        if rms <= 1e-8:
            raise ValueError('参考或候选包含近静音窗口，不能对其作风格排名')
        gain = 1.0 if rms_target is None else min(rms_target / rms, .99 / max(float(abs(clip).max()), 1e-12))
        return clip * gain, rms, gain

    report = {'status': '失败', 'status_scope': 'audio_embedding_measurements_only', 'style_acceptance': '待人工核准',
              'model_id': 'laion/clap-htsat-unfused', 'model_path': str(model_path),
              'model_files': {'pytorch_model.bin': MODEL_SHA256, 'config.json': CONFIG_SHA256,
                              'preprocessor_config.json': PROCESSOR_SHA256},
              'versions': {'torch': torch.__version__, 'torchaudio': torchaudio.__version__,
                           'transformers': transformers.__version__, 'numpy': np.__version__},
              'sample_rate_hz': RATE, 'window_seconds': WINDOW_SECONDS, 'hop_seconds': HOP_SECONDS,
              'seed': SEED, 'device': 'cpu', 'rms_target': rms_target,
              'budget': {'max_total_audio_seconds': max_total_audio_seconds, 'max_windows': max_windows},
              'speech_control': {'file_sha256': speech_metadata['file_sha256'], 'owned': True,
                                  'known_text': config['speech_control_text']},
              'sources': [], 'windows': [], 'controls': [], 'pairs': [],
              'limitations_zh': ['余弦不是同风格概率，也不是录音身份、原创或版权证明。',
                                 '重叠短窗不是独立样本；CLAP对精确速度/律动不敏感。',
                                 '模型只接收波形，不接收标题或风格标签；合法本地分析授权由宿主按实际任务记录。']}
    waveforms, first_reference, reference_metadata = [], None, None
    decoded_seconds = 0
    for item in metadata:
        audio, command = _decode(item, max_total_audio_seconds + 1)
        decoded_seconds += len(audio) / RATE
        if decoded_seconds > max_total_audio_seconds:
            raise ValueError('真实解码总时长超过预算，不静默截短')
        end = len(audio) - RATE * WINDOW_SECONDS
        starts = list(range(0, end + 1, RATE * HOP_SECONDS))
        if end - starts[-1] > RATE // 10:
            starts.append(end)
        if len(waveforms) + len(starts) > max_windows:
            raise ValueError('窗口数超过预算，不静默抽掉部分输入')
        report['sources'].append(dict(item, frames=len(audio), duration_seconds=len(audio) / RATE,
                                       decode_command=command, window_count=len(starts)))
        for start in starts:
            clip, before_rms, gain = scale_window(audio[start:start + RATE * WINDOW_SECONDS])
            waveforms.append(clip)
            report['windows'].append({'source_id': item['source_id'], 'start_seconds': start / RATE,
                                      'input_rms_before_gain': before_rms, 'linear_gain': gain,
                                      'rms': float(np.sqrt(np.mean(clip.astype(np.float64) ** 2))),
                                      'waveform_sha256': hashlib.sha256(clip.tobytes()).hexdigest()})
        if item['source_id'] == reference_id:
            first_reference = scale_window(audio[:RATE * WINDOW_SECONDS])[0]
            reference_metadata = item

    processor = ClapFeatureExtractor.from_pretrained(model_path, local_files_only=True, token=False)
    model, loading_info = ClapModel.from_pretrained(model_path, local_files_only=True, token=False,
                                                   use_safetensors=False, output_loading_info=True)
    if any(loading_info.values()):
        raise ValueError('模型权重加载存在缺失、多余或不符，拒绝随机初始化/不明差异')
    model = model.eval().to('cpu')
    report.update(loading_info=loading_info, feature_extractor=processor.to_dict())

    def run(waves):
        chunks = []
        for start in range(0, len(waves), batch_size):
            batch = processor(waves[start:start + batch_size], sampling_rate=RATE, return_tensors='pt',
                              truncation='rand_trunc', padding='repeatpad')
            if 'feature_probe' not in report:
                feature = batch['input_features']
                report['feature_probe'] = {'shape': list(feature.shape), 'finite': bool(torch.isfinite(feature).all())}
            with torch.inference_mode():
                value = torch.nn.functional.normalize(model.get_audio_features(**batch), dim=-1).cpu().numpy()
            if not np.isfinite(value).all():
                raise ValueError('嵌入含非有限值')
            chunks.append(value)
        return np.concatenate(chunks)

    speech, _ = _decode(speech_metadata, 11)
    faster, tempo_command = _decode(reference_metadata, 12.6, tempo=1.25)
    noise = np.random.default_rng(SEED).standard_normal(len(first_reference)).astype(np.float32)
    noise *= .1 / np.sqrt(np.mean(noise ** 2))
    lowpass = torchaudio.functional.lowpass_biquad(torch.from_numpy(first_reference), RATE, 1500).numpy()
    controls = [first_reference.copy(), first_reference * .25, lowpass,
                scale_window(faster[:RATE * WINDOW_SECONDS])[0], scale_window(speech[:RATE * WINDOW_SECONDS])[0],
                noise, np.zeros_like(first_reference)]
    control_embeddings = run(controls)
    repeated = run([first_reference])[0]
    batch = processor([first_reference], sampling_rate=RATE, return_tensors='pt', truncation='rand_trunc', padding='repeatpad')
    with torch.inference_mode():
        pooled = model.audio_model(**batch).pooler_output
        projected = torch.nn.functional.normalize(model.audio_projection(pooled), dim=-1).cpu().numpy()[0]
    scores = control_embeddings @ repeated
    for label, wave, score in zip(['同波形重复', '音量乘0.25', '1500Hz低通', 'atempo1.25变速及相应截取',
                                  '明确人声阴性', '白噪声阴性', '数字静音阴性'], controls, scores):
        report['controls'].append({'label': label, 'cosine_to_reference_first_window': float(score),
                                    'waveform_sha256': hashlib.sha256(wave.tobytes()).hexdigest()})
    duplicate_difference = float(abs(control_embeddings[0] - repeated).max())
    projection_difference = float(abs(projected - repeated).max())
    gates = {'duplicate_consistent': duplicate_difference <= 1e-6,
             'projection_consistent': projection_difference <= 1e-6,
             'gain_positive': float(scores[1]) > .8, 'known_negatives_separated': bool(np.all(scores[4:] < .5))}
    report['calibration'] = {'status': '通过' if all(gates.values()) else '失败', 'checks': gates,
                              'duplicate_max_abs_difference': duplicate_difference,
                              'projection_max_abs_difference': projection_difference,
                              'limits_scope_zh': '固定检查点的保守粗声音校准：增益阳性大于0.8，已知阴性小于0.5；不是同风格阈值。'}
    report['tempo_control_command'] = tempo_command
    out_path.mkdir(parents=True, exist_ok=False)
    np.save(out_path / '控制嵌入.npy', control_embeddings)
    if not all(gates.values()):
        report.update(code='style_calibration_failed', message='负对照或技术校准不合格，不返回风格排名')
        (out_path / '测量结果.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
        return report

    embeddings = run(waveforms)
    matrix = embeddings @ embeddings.T
    np.save(out_path / '窗口嵌入.npy', embeddings)
    np.save(out_path / '窗口余弦矩阵.npy', matrix)
    indices = {source_id: [i for i, window in enumerate(report['windows']) if window['source_id'] == source_id] for source_id in ids}
    for left_i, left in enumerate(ids):
        for right in ids[left_i:]:
            a, b = embeddings[indices[left]], embeddings[indices[right]]
            values = a @ b.T
            ca, cb = a.mean(axis=0), b.mean(axis=0)
            report['pairs'].append({'left': left, 'right': right, 'all_window_pairs': _distribution(values),
                                    'left_nearest_right': _distribution(values.max(axis=1)),
                                    'right_nearest_left': _distribution(values.max(axis=0)),
                                    'centroid_cosine': float(ca @ cb / (np.linalg.norm(ca) * np.linalg.norm(cb)))})
    report.update(status='通过', embedding_file_sha256=file_hash(out_path / '窗口嵌入.npy'),
                  checks={'all_embeddings_finite': True, 'embedding_dimension': int(embeddings.shape[1]),
                          'duplicate_max_abs_difference': duplicate_difference})
    (out_path / '测量结果.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    return report
