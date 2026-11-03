"""固定同规格窗口的声学差异；不把相似度或单个描述符当风格验收。"""
import hashlib
import math

import numpy as np
import scipy
from scipy import signal

from audio_tools import AudioError, INPUT_FLAGS, _hash_file, _local, _probe, _run, _tools

VERSION = "0.1.1"
RATE = 48000
BANDS = [(20, 60), (60, 150), (150, 500), (500, 2000), (2000, 5000), (5000, 12000), (12000, 24000)]


def acoustic_profile(audio):
    wave = np.asarray(audio, dtype=np.float64)
    if wave.ndim != 2 or wave.shape[1] != 2 or len(wave) < RATE or not np.isfinite(wave).all():
        raise AudioError('audit_input', '声学对照要求至少一秒有限48k双声道波形')
    power = float(np.mean(wave ** 2))
    if power <= 1e-12:
        raise AudioError('audit_silent', '窗口近静音，不能据此比较音乐质感')
    frequencies, spectrum = signal.welch(wave, fs=RATE, nperseg=4096, noverlap=2048, axis=0)
    spectrum = spectrum.mean(axis=1)
    music_mask = frequencies >= 20
    total = float(spectrum[music_mask].sum())
    if total <= 1e-20:
        raise AudioError("audit_spectrum", "窗口缺少20Hz以上有效能量，不能比较音乐频谱")
    ratios = [float(spectrum[(frequencies >= lo) & ((frequencies <= hi) if hi == RATE / 2 else (frequencies < hi))].sum() / total) for lo, hi in BANDS]
    block = RATE // 50
    envelope = np.sqrt(np.mean(wave[:len(wave) // block * block].reshape(-1, block, 2) ** 2, axis=(1, 2)))
    relative = envelope / math.sqrt(power)
    modulation_hz, modulation = signal.welch(relative - relative.mean(), fs=50, nperseg=min(512, len(relative)))
    valid = (modulation_hz >= .5) & (modulation_hz <= 12)
    positions = np.flatnonzero(valid)
    strongest = positions[np.argsort(modulation[valid])[-3:][::-1]] if np.any(modulation[valid] > 1e-14) else []
    mid = (wave[:, 0] + wave[:, 1]) / 2
    side = (wave[:, 0] - wave[:, 1]) / 2
    _, ms_spectrum = signal.welch(np.column_stack([mid, side]), fs=RATE,
                                  nperseg=4096, noverlap=2048, axis=0)
    band_side = []
    for lo, hi in BANDS:
        mask = (frequencies >= lo) & ((frequencies <= hi) if hi == RATE / 2 else (frequencies < hi))
        mid_power, side_power = ms_spectrum[mask].sum(axis=0)
        band_power = float(mid_power + side_power)
        band_side.append(float(side_power / band_power) if band_power > total * 1e-12 else None)
    return {
        'rms_dbfs': 10 * math.log10(power),
        'crest_db': 20 * math.log10(float(np.max(np.abs(wave))) / math.sqrt(power)),
        'band_energy_fractions': ratios,
        'spectral_centroid_hz': float(np.sum(frequencies[music_mask] * spectrum[music_mask]) / total),
        'relative_rms_percentiles_10_50_90': np.percentile(relative, [10, 50, 90]).tolist(),
        'envelope_modulation_candidates_hz': [float(modulation_hz[i]) for i in strongest],
        'side_energy_fraction': float(np.mean(side ** 2) / power),
        'mono_energy_fraction': float(np.mean(mid ** 2) / power),
        'band_side_energy_fractions': band_side,
    }


def _window(path, start, seconds):
    path = _local(path)
    meta = _probe(path)
    if meta['channels'] not in (1, 2):
        raise AudioError('audit_channels', '本声学对照仅验证单/双声道来源')
    if start + seconds > meta['container_duration_seconds'] + 1 / RATE:
        raise AudioError('audit_window', '指定窗口超过真实音频时长，不静默补零或截短')
    result = _run(['ffmpeg', '-v', 'error', '-nostdin', *INPUT_FLAGS, '-i', str(path),
        '-ss', str(start), '-t', str(seconds), '-map', '0:a:0', '-vn', '-ar', str(RATE), '-ac', '2',
        '-f', 'f32le', '-acodec', 'pcm_f32le', 'pipe:1'])
    audio = np.frombuffer(result.stdout, dtype='<f4')
    if result.stderr.strip() or audio.size != round(seconds * RATE) * 2:
        raise AudioError('audit_decode', '窗口未精确完整解码，拒绝部分数据')
    audio = audio.reshape(-1, 2)
    profile = acoustic_profile(audio)
    return audio, {'file_name': path.name, 'file_sha256': _hash_file(path), 'original_format': meta,
        'window_start_seconds': start, 'window_frames': len(audio), 'window_seconds': seconds,
        'decoded_f32le_sha256': hashlib.sha256(audio.tobytes()).hexdigest(), 'profile': profile}


def audit_reference(reference, candidate, reference_start=0, candidate_start=0, seconds=20):
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in (reference_start, candidate_start, seconds)):
        raise AudioError('audit_window', '窗口参数须为有限数字')
    if min(reference_start, candidate_start) < 0 or not 1 <= seconds <= 30:
        raise AudioError('audit_window', '起点不能为负，每次对照窗口须1至30秒')
    wave, ref = _window(reference, reference_start, seconds)
    _, cand = _window(candidate, candidate_start, seconds)
    rp, cp = ref['profile'], cand['profile']
    quiet = acoustic_profile(wave * .25)
    invariant = max(abs(a-b) for a,b in zip(rp['band_energy_fractions'], quiet['band_energy_fractions']))
    if invariant > 1e-8:
        raise AudioError('audit_control', '同波形音量对照不稳定，不发布差异解释')
    deltas = {'band_energy_delta_db': [10 * math.log10(max(b, 1e-15) / max(a, 1e-15))
                  for a,b in zip(rp['band_energy_fractions'], cp['band_energy_fractions'])],
        'spectral_centroid_ratio': cp['spectral_centroid_hz'] / rp['spectral_centroid_hz'],
        'crest_delta_db': cp['crest_db'] - rp['crest_db'],
        'side_energy_fraction_delta': cp['side_energy_fraction'] - rp['side_energy_fraction'],
        'band_side_energy_fraction_delta': [b-a if a is not None and b is not None else None
            for a,b in zip(rp['band_side_energy_fractions'], cp['band_side_energy_fractions'])]}
    return {'status':'通过', 'status_scope':'fixed_window_acoustic_measurements_only',
        'tools':{**_tools(), 'scipy':scipy.__version__, 'reference_audit':VERSION},
        'method':{'spectrum':'左右功率谱平均，再归一20Hz至Nyquist的能量', 'welch_nperseg':4096, 'welch_noverlap':2048,
                  'welch_window':'hann', 'welch_detrend':'constant', 'envelope_window_frames':960,
                  'modulation_range_hz':[.5,12], 'highest_band_includes_nyquist':True,
                  'band_space':'同Welch频带内S/(M+S)功率，M=(L+R)/2，S=(L-R)/2；该频带功率不足总功率1e-12时为null'},
        'reference':ref, 'candidate':cand, 'sample_rate_hz':RATE, 'channels':2,
        'frequency_bands_hz':BANDS, 'candidate_minus_reference':deltas,
        'controls':{'same_wave_gain_025_band_fraction_max_error':invariant},
        'style_acceptance':'未判定', 'limits':[
            '频带、明亮度、包络调制与声场是描述符，不是流派、乐器或好听概率。',
            '调制峰不是已识别BPM、军鼓或拍号；相同速度仍可有不同律动。',
            '两个窗口可以来自不同音乐段落；本结果不能证明整曲或录音身份。',
            '频带比例与峰均比不随等量增益改变；RMS原值单列，不能用提高音量掩盖质感差异。']}
