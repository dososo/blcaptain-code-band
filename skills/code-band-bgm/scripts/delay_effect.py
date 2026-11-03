"""复用FFmpeg aecho的有限抽头延迟；仅接受数字参数，不接受滤镜文本。"""
import math
import subprocess

import numpy as np


def delay_parameters(effect):
    values = {'delay_ms': 375., 'decay': .25, 'repeats': 2}
    parameters = effect.get('parameters', {})
    if not isinstance(parameters, dict) or set(parameters) - set(values):
        raise ValueError('延迟只支持delay_ms、decay、repeats参数')
    values.update(parameters)
    for name, lo, hi in [('delay_ms', 1, 2000), ('decay', 0, .75)]:
        value = values[name]
        if type(value) not in (int, float) or not math.isfinite(value) or not lo <= value <= hi:
            raise ValueError('延迟参数超范围或不是有限数：' + name)
    if type(values['repeats']) is not int or not 1 <= values['repeats'] <= 4:
        raise ValueError('延迟repeats必须为1至4的整数')
    if values['delay_ms'] * values['repeats'] > 4000:
        raise ValueError('有限延迟最长抽头不得超过4秒')
    return values


def validate_effects(effects, part_ids, loop=False, sample_rate=None):
    if not isinstance(effects, list) or len(effects) > 6:
        raise ValueError('最多支持一个混响、一个chorus、一段EQ、两个延迟和一个末端duck配置')
    from chorus_effect import validate_chorus_effects
    validate_chorus_effects(effects, part_ids, loop)
    reverb_count = 0
    delay_count = 0
    duck_count = 0
    eq_count = 0
    for i, effect in enumerate(effects):
        if not isinstance(effect, dict):
            raise ValueError('效果须为JSON对象')
        targets = effect.get('target_part_ids', [])
        if not isinstance(targets, list) or any(type(t) is not str or t not in part_ids for t in targets):
            raise ValueError('效果目标必须引用真实声部')
        if effect.get('kind') == 'reverb':
            reverb_count += 1
            if any(e.get('kind') != 'chorus' for e in effects[:i]) or reverb_count > 1:
                raise ValueError('混响只能有一项，须与原生chorus并行位于后级delay/duck之前')
        elif effect.get('kind') == 'chorus':
            # 参数、路由、唯一性与循环边界由原生效果校验器负责。
            continue
        elif effect.get('kind') == 'eq':
            from eq_effect import eq_parameters
            eq_count += 1
            if (eq_count > 1 or not targets or len(set(targets)) != len(targets)
                    or not isinstance(effect.get('effect_id'), str) or not effect['effect_id'].strip()
                    or any(e.get('kind') in ('delay', 'duck') for e in effects[:i])):
                raise ValueError('eq须为单段、非空唯一真实目标，位于delay/duck前且有effect_id')
            if loop:
                raise ValueError('eq尚未验证循环状态，不接受循环乐谱')
            eq_parameters(effect, sample_rate)
        elif effect.get('kind') == 'delay':
            if not isinstance(effect.get('effect_id'), str) or not effect['effect_id'].strip():
                raise ValueError('延迟须有非空effect_id以记录实际处理链')
            delay_count += 1
            if delay_count > 2:
                raise ValueError('最多支持两项有限抽头延迟配置')
            delay_parameters(effect)
            if loop:
                raise ValueError('有限抽头延迟尚未验证循环状态，不接受循环乐谱')
        elif effect.get('kind') == 'duck':
            from ducking_effect import duck_parameters
            duck_count += 1
            if duck_count > 1 or i != len(effects) - 1 or loop:
                raise ValueError('duck须为效果链末项且仅一项，目前不接受循环')
            duck_parameters(effect, part_ids)
        else:
            raise ValueError('当前仅支持混响、原生chorus、EQ、有限延迟和duck；不会静默忽略效果')


def apply_delay(audio, sample_rate, effect):
    values = delay_parameters(effect)
    audio = np.asarray(audio, dtype='<f8')
    if audio.ndim != 2 or audio.shape[1] != 2 or not np.isfinite(audio).all():
        raise ValueError('延迟输入必须为有限立体声浮点样本')
    delays = '|'.join(format(values['delay_ms'] * i, '.12g') for i in range(1, values['repeats'] + 1))
    decays = '|'.join(format(values['decay'] ** i, '.12g') for i in range(1, values['repeats'] + 1))
    graph = f'aecho=in_gain=1:out_gain=1:delays={delays}:decays={decays},atrim=end_sample={len(audio)}'
    argv = ['ffmpeg', '-v', 'error', '-nostdin', '-f', 'f64le', '-ar', str(sample_rate), '-ac', '2', '-i', 'pipe:0',
            '-af', graph, '-f', 'f64le', '-acodec', 'pcm_f64le', 'pipe:1']
    try:
        result = subprocess.run(argv, input=audio.tobytes(), capture_output=True, check=True, timeout=180)
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError('FFmpeg有限延迟处理失败；不忽略效果或返回干声') from exc
    output = np.frombuffer(result.stdout, dtype='<f8')
    if output.size != audio.size or not np.isfinite(output).all():
        raise ValueError('FFmpeg延迟输出帧数或有限性异常，拒绝静默降级')
    return output.reshape(audio.shape).copy(), {'backend': 'FFmpeg aecho', 'parameters': values,
        'command_argv': argv, 'input_frames': len(audio), 'output_frames': len(audio),
        'order': '采样及选定原生混响/chorus并行→有限延迟→duck→声部增益和声像→总线求和',
        'tail_policy': '延迟尾音包含于目标总帧数；超出窗口部分明确截断，不另加时长', 'automatic_normalization': False}


def probe_delay():
    signal = np.zeros((48000, 2), dtype=np.float64)
    signal[0] = [.25, .125]
    output, details = apply_delay(signal, 48000, {'parameters': {'delay_ms': 250, 'decay': .5, 'repeats': 2}})
    expected = np.zeros_like(signal)
    expected[0], expected[12000], expected[24000] = signal[0], signal[0] * .5, signal[0] * .25
    error = float(np.max(np.abs(output - expected)))
    if error > 1e-7:
        raise ValueError('FFmpeg延迟脉冲位置或衰减不符合声明')
    return {'status': '通过', 'maximum_error': error, 'impulse_frames': [0, 12000, 24000], **details}


def add_delay_capability(result):
    measured = probe_delay()
    cap = result['capabilities']
    cap['mixing']['delay'] = {'state': 'supported', 'evidence_ids': ['ffmpeg_aecho_runtime'],
        'limitations': ['有限抽头、最多两项声部延迟；不支持反馈、自动节拍同步、乒乓或循环状态。']}
    cap['evidence'] = [e for e in cap['evidence'] if e['evidence_id'] != 'ffmpeg_aecho_runtime']
    cap['evidence'].append({'evidence_id': 'ffmpeg_aecho_runtime', 'kind': 'command_output',
        'reference': 'doctor .probe.json的delay_probe', 'description': '真实双声道脉冲延迟位置、衰减和精确帧数已验证；不代表音乐空间听感验收。'})
    result['delay_probe'] = measured
    return result
