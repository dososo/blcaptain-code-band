"""可选librosa节拍分析；返回原始位置和候选，不宣称精确律动已识别。"""
from importlib.metadata import version

import numpy as np


def analyze_rhythm(audio, sample_rate):
    import librosa
    if version('librosa') != '0.11.0':
        raise ValueError('当前成熟节拍后端只验证librosa 0.11.0，请使用可选依赖文件')
    audio = np.asarray(audio)
    if audio.ndim != 1 or not np.isfinite(audio).all() or type(sample_rate) is not int or sample_rate != 22050:
        raise ValueError('节拍后端要求有限22.05kHz单声道波形')
    if len(audio) < sample_rate * 5 or len(audio) > sample_rate * 600:
        raise ValueError('节拍分析仅验证5至600秒音频')
    window = min(sample_rate * 30, len(audio))
    starts = sorted({0, (len(audio) - window) // 2, len(audio) - window})
    results = []
    for start in starts:
        wave = audio[start:start + window].astype(np.float32)
        if float(np.max(np.abs(wave))) < 1e-7:
            results.append({'start_seconds': start / sample_rate, 'end_seconds': (start + window) / sample_rate,
                'tempo_bpm': None, 'beat_times_seconds': [], 'confidence': '未知：窗口静音'})
            continue
        percussion = librosa.effects.percussive(wave)
        onset = librosa.onset.onset_strength(y=percussion, sr=sample_rate, hop_length=256)
        tempo, frames = librosa.beat.beat_track(onset_envelope=onset, sr=sample_rate, hop_length=256,
                                               start_bpm=120, trim=False)
        tempo = float(np.asarray(tempo).reshape(-1)[0])
        times = librosa.frames_to_time(frames, sr=sample_rate, hop_length=256) + start / sample_rate
        if not np.isfinite(times).all() or not np.isfinite(tempo):
            raise ValueError('成熟节拍后端返回非有限结果，不提供候选')
        results.append({'start_seconds': start / sample_rate, 'end_seconds': (start + window) / sample_rate,
            'tempo_bpm': tempo if tempo > 0 else None, 'beat_times_seconds': times.tolist(),
            'confidence': '未校准；可能半拍或倍拍，位置不是人工逐拍标注'})
    return {'status': '通过', 'status_scope': 'rhythm_measurement_candidates',
        'backend': {'library': 'librosa', 'version': version('librosa'), 'method': 'HPSS打击分离、谱通量起音、动态规划beat_track',
                    'sample_rate_hz': sample_rate, 'hop_frames': 256, 'initial_tempo_prior_bpm': 120},
        'windows': results, 'limits': ['最多三个30秒窗，不推断未覆盖段落。',
            '不估精确拍号、摇摆/切分标签、原曲和弦或主题。',
            '节拍候选须和源声轨、其他测量及真实听感交叉核对。']}
