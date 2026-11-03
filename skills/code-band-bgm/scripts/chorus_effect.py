"""可选FluidSynth原生chorus；同一真实资源，不引入第三方DSP。"""
import ctypes as ct
import hashlib
import math

import numpy as np


def chorus_parameters(effect):
    values = {'voices': 3, 'level': .6, 'speed_hz': .2, 'depth_ms': 4.25,
              'send': 127, 'waveform': 'sine'}
    parameters = effect.get('parameters', {})
    if not isinstance(parameters, dict) or set(parameters) - set(values):
        raise ValueError('原生chorus参数存在未知字段或不是对象')
    values.update(parameters)
    for name, lo, hi in [('voices', 1, 8), ('send', 0, 127)]:
        if type(values[name]) is not int or not lo <= values[name] <= hi:
            raise ValueError('原生chorus整数参数超范围：' + name)
    for name, lo, hi in [('level', 0, 1), ('speed_hz', .1, 5), ('depth_ms', 0, 20)]:
        value = values[name]
        if type(value) not in (int, float) or not math.isfinite(value) or not lo <= value <= hi:
            raise ValueError('原生chorus参数超范围或不是有限数：' + name)
    if values['waveform'] != 'sine':
        raise ValueError('原生chorus首版仅支持sine调制')
    return values


def validate_chorus_effects(effects, part_ids, loop=False):
    if not isinstance(effects, list) or any(not isinstance(e, dict) for e in effects):
        raise ValueError('效果必须是对象列表')
    selected = [e for e in effects if e.get('kind') == 'chorus']
    if not selected:
        return None
    if len(selected) > 1:
        raise ValueError('最多支持一个原生chorus配置')
    if loop:
        raise ValueError('原生chorus尚未验证循环状态，不接受循环乐谱')
    effect = selected[0]
    if not isinstance(effect.get('effect_id'), str) or not effect['effect_id'].strip():
        raise ValueError('原生chorus须有非空effect_id以记录实际链路')
    targets = effect.get('target_part_ids', [])
    if (not isinstance(targets, list) or
            any(type(t) is not str or t not in part_ids for t in targets)):
        raise ValueError('原生chorus目标必须引用真实声部')
    if len(set(targets)) != len(targets):
        raise ValueError('原生chorus目标不能重复')
    index = effects.index(effect)
    if any(e.get('kind') in ('eq', 'delay', 'duck') for e in effects[:index]):
        raise ValueError('原生chorus须在后级delay/duck之前，与原生混响并行')
    chorus_parameters(effect)
    return effect


def apply_chorus(synth, values, channel=0):
    values = chorus_parameters({'parameters': values})
    lib = synth.lib
    declarations = {'fluid_synth_chorus_on': [ct.c_void_p, ct.c_int, ct.c_int],
                    'fluid_synth_get_cc': [ct.c_void_p, ct.c_int, ct.c_int, ct.POINTER(ct.c_int)]}
    for name, typ in [('nr', ct.c_int), ('level', ct.c_double), ('speed', ct.c_double),
                      ('depth', ct.c_double), ('type', ct.c_int)]:
        declarations['fluid_synth_set_chorus_group_' + name] = [ct.c_void_p, ct.c_int, typ]
        declarations['fluid_synth_get_chorus_group_' + name] = [ct.c_void_p, ct.c_int, ct.POINTER(typ)]
    try:
        for name, arguments in declarations.items():
            function = getattr(lib, name)
            function.restype, function.argtypes = ct.c_int, arguments
    except AttributeError as exc:
        raise ValueError('本机FluidSynth缺少原生chorus接口，不返回干声替代') from exc
    calls = []

    def call(name, *arguments):
        code = int(getattr(lib, name)(*arguments))
        calls.append({'api': name, 'return_code': code})
        if code < 0:
            raise ValueError('原生chorus接口调用失败：' + name)

    readback = {}
    for name, field, typ in [('nr', 'voices', ct.c_int), ('level', 'level', ct.c_double),
                              ('speed', 'speed_hz', ct.c_double), ('depth', 'depth_ms', ct.c_double),
                              ('type', 'waveform', ct.c_int)]:
        expected = 0 if field == 'waveform' else values[field]
        call('fluid_synth_set_chorus_group_' + name, synth.synth, 0, expected)
        actual = typ()
        call('fluid_synth_get_chorus_group_' + name, synth.synth, 0, ct.byref(actual))
        if not math.isclose(actual.value, expected, rel_tol=1e-6, abs_tol=1e-8):
            raise ValueError('原生chorus参数读回不一致：' + field)
        readback[field] = actual.value
    call('fluid_synth_chorus_on', synth.synth, 0, 1)
    call('fluid_synth_cc', synth.synth, channel, 93, values['send'])
    cc = ct.c_int()
    call('fluid_synth_get_cc', synth.synth, channel, 93, ct.byref(cc))
    if cc.value != values['send']:
        raise ValueError('原生chorus CC93发送读回不一致')
    return {'backend': 'FluidSynth native chorus', 'parameters': values,
            'group_readback': readback, 'channel': channel, 'cc93_readback': cc.value, 'api_calls': calls,
            'send_semantics': 'CC93叠加于SoundFont生成器的基底发送；send=0仍可能有湿声，不等于关闭效果。',
            'order': '采样→原生chorus与选定混响并行→后级delay/duck→声部增益和声像→总线',
            'tail_policy': '效果尾声包含在目标总帧数内；超窗截断，不另加时长',
            'automatic_normalization': False}


def probe_chorus(soundfont):
    from fluid_engine import render_events
    document = {'sample_rate_hz': 48000, 'total_frames': 48000,
                'parts': [{'part_id': 'probe', 'bank': 0, 'program': 5, 'gain_db': 0, 'pan': 0}],
                'events': [{'event_id': 'g4', 'part_id': 'probe', 'onset_frame': 0,
                            'gate_frames': 28800, 'midi': 67, 'velocity': 84}],
                'mix': {'master_gain_db': 0, 'effects': []}, 'render': {'loop': {'enabled': False}}}
    dry, _ = render_events(document, soundfont)
    document['mix']['effects'] = [{'effect_id': 'native_chorus_probe', 'kind': 'chorus',
                                  'target_part_ids': ['probe'], 'parameters': {}}]
    wet, diagnostic = render_events(document, soundfont)
    repeated, _ = render_events(document, soundfont)

    def side_fraction(audio):
        audio = audio - np.mean(audio, axis=0)
        mid, side = (audio[:, 0] + audio[:, 1]) / 2, (audio[:, 0] - audio[:, 1]) / 2
        return float(np.sum(side * side) / (np.sum(mid * mid) + np.sum(side * side)))

    dry_side, wet_side = side_fraction(dry), side_fraction(wet)
    if (wet.shape != (48000, 2) or not np.isfinite(wet).all() or
            not np.array_equal(wet, repeated) or wet_side <= dry_side + .05):
        raise ValueError('原生chorus代表探针的帧数、有限性、确定性或立体变化未通过')
    return {'status': '通过', 'sample_rate_hz': 48000, 'frames': 48000, 'channels': 2,
            'bank': 0, 'program': 5, 'midi': 67, 'velocity': 84,
            'dry_side_fraction': dry_side, 'wet_side_fraction': wet_side,
            'dry_audio_sha256': hashlib.sha256(dry.tobytes()).hexdigest(),
            'wet_audio_sha256': hashlib.sha256(wet.tobytes()).hexdigest(),
            'same_event_repeat_sample_exact': True,
            'diagnostic': diagnostic['parts'][0]['chorus'],
            'scope': '仅同一真实GM005 G4代表探针；不认原曲音源或效果，不代表音乐听感验收'}


def add_chorus_capability(result, soundfont):
    measured = probe_chorus(soundfont)
    cap = result['capabilities']
    cap['mixing']['chorus'] = {'state': 'supported', 'evidence_ids': ['fluidsynth_native_chorus_runtime'],
        'limitations': ['原生sine调制；最多一组，按声部路由；不支持循环状态。',
                        'CC93=0不一定零湿声；SoundFont可能有基底发送。',
                        'GM005代表探针通过，不代表任意资源或参考听感匹配。']}
    cap['evidence'] = [e for e in cap['evidence'] if e['evidence_id'] != 'fluidsynth_native_chorus_runtime']
    cap['evidence'].append({'evidence_id': 'fluidsynth_native_chorus_runtime', 'kind': 'command_output',
        'reference': 'doctor .probe.json的chorus_probe',
        'description': '同GM005默认原生chorus真实干湿、旁能量、帧数、有限性与同事件逐样重渲染检查；非听感签收。'})
    result['chorus_probe'] = measured
    return result
