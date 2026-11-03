"""复用FFmpeg因果双精度peaking EQ；只允许单段数值衰减。"""
import hashlib
import math
import subprocess

import numpy as np


def eq_parameters(effect, sample_rate=None):
    values = effect.get('parameters')
    limits = {'frequency_hz': (50, 18000), 'q': (.2, 8), 'gain_db': (-12, 0)}
    if not isinstance(values, dict) or set(values) != set(limits):
        raise ValueError('eq须显式且仅指定frequency_hz、q、gain_db三个数值')
    for key, (low, high) in limits.items():
        value = values[key]
        if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError('eq参数越界或不是有限数：' + key)
    if sample_rate is not None:
        if type(sample_rate) is not int or not 8000 <= sample_rate <= 96000:
            raise ValueError('eq采样率必须为8000至96000Hz整数')
        if values['frequency_hz'] >= sample_rate / 2:
            raise ValueError('eq中心频率须低于Nyquist频率')
    return dict(values)


def apply_eq(audio, sample_rate, effect):
    values = eq_parameters(effect, sample_rate)
    wave = np.asarray(audio, dtype='<f8')
    if wave.ndim != 2 or wave.shape[1] != 2 or not len(wave) or not np.isfinite(wave).all():
        raise ValueError('eq输入必须为非空、有限、双声道浮点样本')
    graph = ('equalizer=frequency={frequency_hz:.12g}:width_type=q:width={q:.12g}:'
             'gain={gain_db:.12g}:mix=1:channels=all:normalize=0:precision=f64:blocksize=0').format(**values)
    argv = ['ffmpeg', '-v', 'error', '-nostdin', '-filter_threads', '1', '-f', 'f64le',
            '-ar', str(sample_rate), '-ac', '2', '-i', 'pipe:0', '-af', graph,
            '-f', 'f64le', '-acodec', 'pcm_f64le', 'pipe:1']
    try:
        result = subprocess.run(argv, input=wave.tobytes(), capture_output=True, check=True, timeout=180)
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError('FFmpeg eq处理失败，不返回干声降级') from exc
    if result.stderr or len(result.stdout) != wave.size * 8:
        raise ValueError('FFmpeg eq输出帧数或错误流异常，拒绝静默降级')
    output = np.frombuffer(result.stdout, dtype='<f8').reshape(wave.shape)
    if not np.isfinite(output).all():
        raise ValueError('FFmpeg eq输出非有限数，拒绝交付')
    return output.copy(), {'backend': 'FFmpeg equalizer causal f64', 'parameters': values,
        'command_argv': argv, 'input_frames': len(wave), 'output_frames': len(output),
        'input_float64_sha256': hashlib.sha256(wave.tobytes()).hexdigest(),
        'output_float64_sha256': hashlib.sha256(output.tobytes()).hexdigest(),
        'order': '采样与原生混响/chorus并行→单段EQ→delay→duck→声部gain/pan→求和',
        'tail_policy': '因果IIR状态从零开始，输出保持声明帧数，滤波尾在同窗口内，不额外补时长',
        'automatic_normalization': False, 'blocksize': 0}


def probe_eq():
    rate = 48000
    time = np.arange(rate, dtype=np.float64) / rate
    signal = .1 * np.column_stack([np.sin(2 * np.pi * 200 * time), np.sin(2 * np.pi * 3500 * time)])
    effect = {'parameters': {'frequency_hz': 3500, 'q': .9, 'gain_db': -3.5}}
    output, details = apply_eq(signal, rate, effect)
    repeat, _ = apply_eq(signal, rate, effect)
    gain = 20 * np.log10(np.sqrt(np.mean(output[rate // 2:] ** 2, axis=0)) /
                        np.sqrt(np.mean(signal[rate // 2:] ** 2, axis=0)))
    impulse = np.zeros((rate, 2), dtype=np.float64)
    impulse[0] = [.1, .05]
    response, _ = apply_eq(impulse, rate, effect)
    if (abs(gain[1] + 3.5) > .02 or abs(gain[0]) > .03 or
            not np.array_equal(output, repeat) or not np.any(response[0])):
        raise ValueError('eq真实频响、无分块延迟或固定重复验证失败')
    version = subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True, timeout=10)
    return {'status': '通过', 'low_gain_db': float(gain[0]), 'center_gain_db': float(gain[1]),
            'float_repeat_equal': True, 'impulse_first_active_frame': 0,
            'ffmpeg_version': version.stdout.decode().splitlines()[0], **details}


def add_eq_capability(result):
    measured = probe_eq()
    cap = result['capabilities']
    cap['mixing']['eq'] = {'state': 'supported', 'evidence_ids': ['ffmpeg_peaking_cut_runtime'],
        'limitations': ['一段显式目标声部peaking衰减，50–18000Hz且低于Nyquist，Q .2–8、gain −12至0dB。',
                        '固定双声道f64/因果blocksize0，未提供动态EQ/自动去刺耳/多段/增益提升或循环验证。']}
    cap['evidence'] = [e for e in cap['evidence'] if e['evidence_id'] != 'ffmpeg_peaking_cut_runtime']
    cap['evidence'].append({'evidence_id': 'ffmpeg_peaking_cut_runtime', 'kind': 'command_output',
        'reference': 'doctor JSON的eq_probe', 'description': '真实200/3500Hz双声道频响、脉冲首帧及固定浮点重复；不签音乐听感。'})
    result['eq_probe'] = measured
    return result
