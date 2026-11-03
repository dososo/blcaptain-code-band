"""署名条件统计的有界演奏修改；不作曲、不补鼓、不加载模型。"""
import copy
from fractions import Fraction
import math

from score_tools import digest, require, round_fraction, tick_frame, validate_score


def apply_profile(score, profile, part_id):
    validate_score(score)
    require(isinstance(profile, dict) and profile.get('schema_version') == '1.0.0'
            and profile.get('hit_key') == 'closed_hat' and profile.get('steps_per_bar') == 16,
            'groove/profile', '本版本仅接受16格闭镲条件统计，不将其他鼓类静默重映射')
    require(type(profile.get('min_observations')) is int and profile['min_observations'] >= 3,
            'groove/profile', '条件统计至少需要3次观测')
    provenance = profile.get('provenance')
    require(isinstance(provenance, dict) and all(isinstance(provenance.get(k), str) and provenance[k]
            for k in ('dataset', 'author', 'url', 'license', 'midi_file', 'midi_sha256', 'method')),
            'groove/provenance', '需要来源、署名、许可及转换说明；字段齐全不代替来源核查')
    rows = profile.get('phases')
    require(isinstance(rows, list) and 1 <= len(rows) <= 16, 'groove/profile', '需要1至16个非重复拍位')
    phases = {}
    for row in rows:
        require(isinstance(row, dict) and type(row.get('step')) is int and 0 <= row['step'] < 16
                and row['step'] not in phases and type(row.get('observations')) is int and row['observations'] >= 0,
                'groove/phases', '拍位/样本数无效或重复')
        require(all(type(row.get(k)) in (int, float) and math.isfinite(row[k]) for k in ('velocity_delta', 'offset_step_fraction'))
                and abs(row['velocity_delta']) <= 127 and abs(row['offset_step_fraction']) <= .5,
                'groove/phases', '表达偏差无效或超范围')
        phases[row['step']] = row
    timeline = score['timeline']
    require(len(timeline['tempo_map']) == len(timeline['meter_map']) == 1
            and timeline['meter_map'][0]['numerator'] == timeline['meter_map'][0]['denominator'] == 4
            and timeline['ppq'] % 4 == 0, 'groove/timeline', '本统计适配只支持恒速4/4、可整分的16分网格')
    parts = [p for p in score['parts'] if p['part_id'] == part_id]
    require(len(parts) == 1 and not parts[0]['humanize']['enabled'], 'groove/part', '目标声部不存在或仍启用随机人性化')
    revised = copy.deepcopy(score)
    part = next(p for p in revised['parts'] if p['part_id'] == part_id)
    step_ticks = timeline['ppq'] // 4
    bpm = timeline['tempo_map'][0]['bpm']
    seconds_per_tick = Fraction(60*bpm['denominator'], timeline['ppq']*bpm['numerator'])
    starts = {s['start_tick'] for s in score['sections']}
    changes, unmatched = [], 0
    rate = score['render']['sample_rate_hz']
    max_frames = math.floor(.006*rate/64)*64
    frame = lambda tick: round_fraction(Fraction(tick_frame(tick,timeline,rate),64))*64
    for note in part['notes']:
        tick = note['onset_tick']
        row = phases.get((tick//step_ticks) % 16)
        if (note['pitch'].get('hit_key') != 'closed_hat' or tick % step_ticks
                or row is None or row['observations'] < profile['min_observations']):
            unmatched += 1
            continue
        old = copy.deepcopy(note)
        delta = max(-8, min(8, round_fraction(Fraction(str(row['velocity_delta'])))))
        desired = max(-.005, min(.005, row['offset_step_fraction']*float(seconds_per_tick)*step_ticks))
        ticks = round_fraction(Fraction(str(desired))/seconds_per_tick)
        target = tick + ticks
        # Section入口保网格；实际64帧偏移超界或时长越界时保留原起点。
        if tick in starts or target < 0 or target+note['duration_ticks'] > timeline['music_end_tick'] or abs(frame(target)-frame(tick)) > max_frames:
            target = tick
        note['onset_tick'] = target
        note['velocity'] = max(1, min(127, note['velocity']+delta))
        changes.append({'note_id':note['note_id'], 'phase':row['step'], 'observations':row['observations'],
                        'onset_tick_before':tick, 'onset_tick_after':target,
                        'compiled_onset_delta_frames':frame(target)-frame(tick),
                        'velocity_before':old['velocity'], 'velocity_after':note['velocity'],
                        'velocity_delta':note['velocity']-old['velocity']})
    validate_score(revised)
    return revised, {'status':'通过', 'scope':'指定闭镲的静态条件表现修改，不核准人味/风格',
        'parent_score_sha256':digest(score), 'score_sha256':digest(revised), 'profile_sha256':digest(profile),
        'provenance':copy.deepcopy(provenance), 'part_id':part_id,
        'eligible_note_count':len(changes), 'unmatched_note_count':unmatched, 'changes':changes,
        'limits':{'velocity_delta':8,'requested_time_ms':5,'compiled_onset_max_frames':max_frames},
        'preserved':'全部音高/音符ID/门长/种子、其他声部；不补删击打，未知拍位原样保留'}
