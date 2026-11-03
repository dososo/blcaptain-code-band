"""显式打击事件触发的增益包络；不冒称音频侧链压缩器或原曲效果复原。"""
import hashlib
import math

import numpy as np


def duck_parameters(effect, part_ids=None):
    values = {'attack_ms': 4., 'hold_ms': 40., 'release_ms': 180., 'depth_db': -7.}
    parameters = effect.get('parameters', {})
    if not isinstance(parameters, dict) or set(parameters) - (set(values) | {'trigger_part_id', 'trigger_part_ids'}):
        raise ValueError('duck只接受trigger_part_id或trigger_part_ids、attack_ms、hold_ms、release_ms、depth_db')
    values.update(parameters)
    if ('trigger_part_id' in parameters) == ('trigger_part_ids' in parameters):
        raise ValueError('duck须且只能声明trigger_part_id或trigger_part_ids')
    if 'trigger_part_ids' in parameters:
        triggers = values['trigger_part_ids']
        if (not isinstance(triggers, list) or not 1 <= len(triggers) <= 8
                or any(not isinstance(t, str) or not t.strip() for t in triggers)
                or len(set(triggers)) != len(triggers)):
            raise ValueError('duck多触发须为1至8个非空、唯一声部ID')
    else:
        trigger = values.get('trigger_part_id')
        if not isinstance(trigger, str) or not trigger:
            raise ValueError('duck须有显式触发声部和独立、非空目标声部')
        triggers = [trigger]
    targets = effect.get('target_part_ids', [])
    if not isinstance(targets, list) or not targets or any(t in targets for t in triggers):
        raise ValueError('duck须有显式触发声部和独立、非空目标声部')
    if part_ids is not None and (any(t not in part_ids for t in triggers) or any(t not in part_ids for t in targets)):
        raise ValueError('duck触发或目标声部不存在')
    for key, low, high in [('attack_ms', .5, 50), ('hold_ms', 0, 250), ('release_ms', 1, 2000), ('depth_db', -24, 0)]:
        number = values[key]
        if type(number) not in (int, float) or not math.isfinite(number) or not low <= number <= high:
            raise ValueError('duck参数超范围或非有限数：' + key)
    return values


def apply_duck(audio, sample_rate, effect, events):
    wave = np.asarray(audio, dtype=np.float64)
    if wave.ndim != 2 or wave.shape[1] != 2 or not np.isfinite(wave).all():
        raise ValueError('duck输入须为有限立体声浮点音频')
    if type(sample_rate) is not int or not 8000 <= sample_rate <= 96000:
        raise ValueError('duck采样率无效')
    values = duck_parameters(effect)
    triggers = values.get('trigger_part_ids', [values.get('trigger_part_id')])
    if 'trigger_part_ids' in values and any(not any(e['part_id'] == t for e in events) for t in triggers):
        raise ValueError('duck每个多触发声部均须有真实触发事件')
    onsets = sorted({e['onset_frame'] for e in events if e['part_id'] in triggers})
    if not onsets or any(type(frame) is not int or not 0 <= frame < len(wave) for frame in onsets):
        raise ValueError('duck没有实际触发事件或触发时刻越界')
    attack = max(1, round(values['attack_ms'] * sample_rate / 1000))
    hold = round(values['hold_ms'] * sample_rate / 1000)
    release = max(1, round(values['release_ms'] * sample_rate / 1000))
    floor = 10 ** (values['depth_db'] / 20)
    contour = np.concatenate([np.linspace(1, floor, attack, endpoint=False), np.full(hold, floor), np.linspace(floor, 1, release + 1)])
    envelope = np.ones(len(wave), dtype=np.float64)
    for frame in onsets:
        size = min(len(contour), len(wave) - frame)
        envelope[frame:frame+size] = np.minimum(envelope[frame:frame+size], contour[:size])
    diagnostic = {'backend': 'explicit_event_gain_envelope', 'parameters': values,
        'trigger_count': len(onsets), 'attack_frames': attack, 'hold_frames': hold, 'release_frames': release,
        'minimum_gain': float(envelope.min()), 'envelope_sha256': hashlib.sha256(envelope.astype('<f8').tobytes()).hexdigest(),
        'overlap_rule': '取增益包络最小值，不相乘加深；线性幅度attack/hold/release',
        'tail_policy': '恢复包络在既定总帧数内处理，超界不延长文件；未验证循环',
        'scope': '自有制作手段；触发来自已编译音符，不检测原音频、不声称已识别原sidechain'}
    if 'trigger_part_ids' in values:
        diagnostic['trigger_source_event_counts'] = {t: sum(e['part_id'] == t for e in events) for t in triggers}
        diagnostic['trigger_source_onset_counts'] = {t: len({e['onset_frame'] for e in events if e['part_id'] == t}) for t in triggers}
    return wave * envelope[:, None], diagnostic


def add_duck_capability(result):
    wave = np.ones((48000, 2), dtype=np.float64) * .1
    effect = {'target_part_ids': ['bass'], 'parameters': {'trigger_part_id': 'kick', 'attack_ms': 10, 'hold_ms': 20, 'release_ms': 30, 'depth_db': -12}}
    output, evidence = apply_duck(wave, 48000, effect, [{'part_id': 'kick', 'onset_frame': 4800}])
    floor = 10 ** (-12 / 20)
    if not np.array_equal(output[:4800], wave[:4800]) or abs(output[5280, 0] - .1 * floor) > 1e-15 or abs(output[7680, 0] - .1) > 1e-15:
        raise ValueError('显式duck触发位置、深度或恢复验证失败')
    split_effect = {'target_part_ids': ['bass'], 'parameters': {
        'trigger_part_ids': ['kick_a', 'kick_b'], 'attack_ms': 10, 'hold_ms': 20, 'release_ms': 30, 'depth_db': -12}}
    events = [{'part_id': part, 'onset_frame': frame} for part, frame in
              [('kick_a', 4800), ('kick_a', 9600), ('kick_b', 11040), ('kick_b', 24000), ('kick_b', 9600)]]
    split, split_evidence = apply_duck(wave, 48000, split_effect, events)
    merged, merged_evidence = apply_duck(wave, 48000, effect, [{**e, 'part_id': 'kick'} for e in events])
    only_a, _ = apply_duck(wave, 48000, effect, [{**e, 'part_id': 'kick'} for e in events if e['part_id'] == 'kick_a'])
    if (not np.array_equal(split, merged) or np.array_equal(split, only_a)
            or split_evidence['trigger_count'] != 4 or split.shape != wave.shape or not np.isfinite(split).all()):
        raise ValueError('显式duck多来源并集与旧单来源合并对照失败')
    cap = result['capabilities']
    cap['mixing']['duck'] = {'state': 'supported', 'evidence_ids': ['explicit_duck_envelope', 'explicit_duck_multi_trigger'], 'limitations': ['单来源或1至8个来源的起点去重并集、最多一项目标包络；不是音频侧链压缩器；循环未验证。']}
    cap['evidence'] = [e for e in cap['evidence'] if e['evidence_id'] not in ('explicit_duck_envelope', 'explicit_duck_multi_trigger')]
    cap['evidence'].append({'evidence_id': 'explicit_duck_envelope', 'kind': 'command_output', 'reference': 'doctor JSON的duck_probe', 'description': '实际浮点控制验证触发、深度、恢复和精确帧数；不代表原FX识别或音乐验收。'})
    cap['evidence'].append({'evidence_id': 'explicit_duck_multi_trigger', 'kind': 'command_output', 'reference': 'doctor JSON的duck_multi_trigger_probe', 'description': '实际A/B分组、同起去重及重叠包络，与旧单触发合并逐样相等；不漏第二来源、不累计加深。'})
    result['duck_probe'] = {'status': '通过', **evidence}
    result['duck_multi_trigger_probe'] = {'status': '通过', 'split_equals_single_merged': True,
        'split_output_sha256': hashlib.sha256(split.astype('<f8').tobytes()).hexdigest(),
        'single_merged_output_sha256': hashlib.sha256(merged.astype('<f8').tobytes()).hexdigest(),
        'single_merged_envelope_sha256': merged_evidence['envelope_sha256'], 'input_frames': len(wave),
        'output_frames': len(split), **split_evidence}
    return result
