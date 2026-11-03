"""SoundFont原生发生器偏移；相对原预设，不冒称毫秒、Hz或新的采样奏法。"""
import math

FIELDS = {
    'amp_attack_offset_timecents': (34, -3600, 0),
    'amp_release_offset_timecents': (38, -2400, 0),
    'filter_cutoff_offset_cents': (8, -1200, 0),
}


def validate_shaping(value):
    if not isinstance(value, dict) or not set(value) <= FIELDS.keys():
        raise ValueError('声音塑形必须为白名单字段对象，不接受任意发生器编号')
    for key, number in value.items():
        _, low, high = FIELDS[key]
        if type(number) not in (int, float) or not math.isfinite(number) or not low <= number <= high:
            raise ValueError('声音塑形偏移越界或非有限数：'+key)
    return value


def apply_shaping(synth, channel, value):
    validate_shaping(value)
    observed = {}
    for key in sorted(value):
        identifier = FIELDS[key][0]
        result = synth.lib.fluid_synth_set_gen(synth.synth, channel, identifier, float(value[key]))
        actual = float(synth.lib.fluid_synth_get_gen(synth.synth, channel, identifier))
        if result < 0 or not math.isfinite(actual) or abs(actual-value[key]) > 1e-3:
            raise ValueError('音频引擎未接受声音塑形，不能静默忽略：'+key)
        observed[key] = actual
    return observed


def add_shaping_capability(result, soundfont):
    import copy
    import numpy as np
    from fluid_engine import render_events
    cap = result['capabilities']
    from soundfont_inventory import inventory
    mapping = inventory(soundfont)
    representative = next((p for p in mapping['presets'] if p['bank'] == 0 and p['program'] == 89), None)
    if representative is None or not any(z['available_in_file'] and z['key_range'][0] <= 60 <= z['key_range'][1]
            and z['velocity_range'][0] <= 90 <= z['velocity_range'][1] for z in representative['mappings']):
        cap['performance']['sound_shaping'] = {'state':'unknown','evidence_ids':[],
            'limitations':['当前音色库缺少此探测采用的WarmPad89/键60/力度90区域，未运行代表塑形验证；不影响其他基础能力。']}
        cap['evidence'] = [e for e in cap['evidence'] if e['evidence_id'] != 'native_sound_shaping_probe']
        result['sound_shaping_probe'] = {'status':'未实测','reason':'缺少代表音色区域'}
        return result
    document = {'sample_rate_hz':48000,'total_frames':96000,
        'parts':[{'part_id':'shape_probe','bank':0,'program':89,'gain_db':0,'pan':0}],
        'events':[{'event_id':'shape','part_id':'shape_probe','onset_frame':0,'gate_frames':20480,'midi':60,'velocity':90}],
        'mix':{'master_gain_db':0,'effects':[]},'render':{'loop':{'enabled':False}}}
    original,_ = render_events(document, soundfont)
    changed = copy.deepcopy(document)
    changed['parts'][0]['sound_shaping'] = {'amp_attack_offset_timecents':-3600,
        'amp_release_offset_timecents':-2400,'filter_cutoff_offset_cents':-1200}
    shaped,diagnostic = render_events(changed, soundfont)
    again,_ = render_events(changed, soundfont)
    if not np.isfinite(shaped).all() or np.max(np.abs(shaped)) <= 1e-5 or np.array_equal(original, shaped) or not np.array_equal(shaped, again):
        raise ValueError('原生发生器实际发声/效果/双重渲染探测失败')
    identifier = 'native_sound_shaping_probe'
    cap = result['capabilities']
    cap['performance']['sound_shaping'] = {'state':'supported','evidence_ids':[identifier],
        'limitations':['仅静态声部级音量包络attack/release时间音分偏移及滤波截止音分偏移。',
                      '改变原预设参数，无法去掉源采样本体已录制的慢起音；未实现逐音符自动化或滑音。']}
    cap['evidence'] = [e for e in cap['evidence'] if e['evidence_id']!=identifier]
    cap['evidence'].append({'evidence_id':identifier,'kind':'command_output','reference':'doctor JSON的sound_shaping_probe',
        'description':'真实bank0/program89键60，三个原生发生器读回及实际波形变化；实际预设名称见sound_shaping_probe，两次独立渲染float完全一致。'})
    result['sound_shaping_probe'] = {'actual_preset_name':representative['name'], 'frames':len(shaped),'changed_audio':True,'float_repeat_equal':True,
        'accepted_offsets':diagnostic['parts'][0]['sound_shaping'],
        'scope':'代表配置真实发声；不等于所有偏移值或所有采样的听感已验收。'}
    return result
