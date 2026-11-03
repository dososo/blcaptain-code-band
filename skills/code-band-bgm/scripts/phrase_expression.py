"""显式、确定性的乐句表现；只改指定音符的起点和力度，不作曲或渲染。"""
import argparse
import copy
from fractions import Fraction
import json
import math
from pathlib import Path
import sys


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(path):
    def reject(value):
        raise ValueError('JSON不允许非有限数：'+value)
    return json.loads(Path(path).read_text(encoding='utf-8'), parse_constant=reject)


def apply_phrase(score, plan, strength=1):
    from score_tools import digest, round_fraction, tick_frame, validate_score
    require(type(strength) in (int, float) and math.isfinite(strength) and 0 <= strength <= 1,
            '强度必须为0至1的有限数，不能用布尔或字符串')
    validate_score(score)
    require(type(plan) is dict and set(plan) == {'schema_version','start_tick','end_tick','edits'}
            and plan['schema_version'] == '1.0.0', '乐句计划字段或版本不符')
    start, end = plan['start_tick'], plan['end_tick']
    require(type(start) is int and type(end) is int and 0 <= start < end <= score['timeline']['music_end_tick'],
            '乐句区间必须位于主体内')
    require(type(plan['edits']) is list and bool(plan['edits']), '需要明确的声部与音符修改项')
    parts = {p['part_id']:p for p in score['parts']}
    factor = Fraction(str(strength))
    revised = copy.deepcopy(score)
    targets = {p['part_id']:p for p in revised['parts']}
    seen, changes = set(), []
    for edit in plan['edits']:
        require(type(edit) is dict and set(edit) == {'part_id','note_ids','offset_ticks','velocity_delta'},
                '修改项仅允许声部/音符ID、tick偏移及力度差')
        pid = edit['part_id']
        require(type(pid) is str and pid in parts, '目标声部不存在')
        require(not parts[pid]['humanize']['enabled'], '目标声部须关闭随机人性化，不能重抽既有表现')
        ids = edit['note_ids']
        require(type(ids) is list and bool(ids) and all(type(x) is str for x in ids)
                and len(ids) == len(set(ids)), '音符ID必须非空且不重复')
        require(type(edit['offset_ticks']) is int and type(edit['velocity_delta']) is int,
                'tick偏移与力度差必须为真整数')
        old = {n['note_id']:n for n in parts[pid]['notes']}
        new = {n['note_id']:n for n in targets[pid]['notes']}
        dt = round_fraction(factor * edit['offset_ticks'])
        dv = round_fraction(factor * edit['velocity_delta'])
        for nid in ids:
            require(nid in old and (pid,nid) not in seen, '音符ID不存在或被重复修改')
            seen.add((pid,nid))
            before = old[nid]
            on, off = before['onset_tick'], before['onset_tick'] + before['duration_ticks']
            require(start <= on < off <= end, '目标音符须完整位于显式乐句区间')
            target = on + dt
            velocity = before['velocity'] + dv
            require(start <= target and target + before['duration_ticks'] <= end,
                    '偏移越出乐句/主体边界，禁止静默夹取')
            require(1 <= velocity <= 127, '力度超出1至127，禁止静默夹取')
            # 固定原段落所有权；不为微表现移动段落边界或尾音预算。
            section = next(s for s in score['sections'] if s['start_tick'] <= on < s['end_tick'])
            require(section['start_tick'] <= target and target + before['duration_ticks'] <= section['end_tick'],
                    '偏移跨原段落边界')
            frame = lambda tick: round_fraction(Fraction(tick_frame(tick, score['timeline'], score['render']['sample_rate_hz']), 64)) * 64
            old_gate = frame(off) - frame(on)
            new_gate = frame(target + before['duration_ticks']) - frame(target)
            require(new_gate > 0 and abs(new_gate-old_gate) <= 64,
                    '偏移跨速度变化导致实际门长变化超过64帧或零门长')
            new[nid]['onset_tick'] = target
            new[nid]['velocity'] = velocity
            if before != new[nid]:
                changes.append({'part_id':pid, 'note_id':nid,
                    'onset_tick_before':on,'onset_tick_after':target,
                    'offset_ticks':dt,'velocity_before':before['velocity'],
                    'velocity_after':velocity,'velocity_delta':dv})
    validate_score(revised)
    for pid in {x[0] for x in seen}:
        before = parts[pid]['notes']
        after = {n['note_id']:n for n in targets[pid]['notes']}
        for x in before:
            for y in before:
                if x['onset_tick'] < y['onset_tick']:
                    require(after[x['note_id']]['onset_tick'] <= after[y['note_id']]['onset_tick'],
                            '偏移使声部原起点次序翻转')
        keys = {}
        for note in after.values():
            keys.setdefault(digest(note['pitch']), []).append(note)
        for notes in keys.values():
            ordered = sorted(notes, key=lambda n:n['onset_tick'])
            require(all(x['onset_tick']+x['duration_ticks'] <= y['onset_tick'] for x,y in zip(ordered,ordered[1:])),
                    '偏移造成同声部同键门控重叠')
    return revised, {'schema_version':'1.0.0','scope':'明确乐句的跨声部起点与力度编辑；不是人味/风格评分',
        'score_sha256_before':digest(score),'score_sha256_after':digest(revised),
        'plan_sha256':digest(plan),'strength':strength,'changes':changes,
        'preserved':'音高、音符ID/数量/时值、段落/总时长/尾音、其他字段及未选声部；timing_locked仍锁编译随机偏移，不禁止本次显式编辑',
        'limits':'原谱tick时值不变；按当前64帧调度核门长变化最多64帧，超出或零门长拒绝，父子实际编译另核；力度不保证音源具有相同响应。'}


def main():
    parser = argparse.ArgumentParser(description='有界确定性乐句表现研究；不渲染WAV')
    for name in ('skill-root','score','plan','out'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--strength', type=float, default=1)
    args = parser.parse_args()
    try:
        out = Path(args.out)
        require(not out.exists(), '输出目录已存在，拒绝覆盖')
        sys.path.insert(0, str(Path(args.skill_root)/'scripts'))
        score, report = apply_phrase(load(args.score), load(args.plan), args.strength)
        # 所有输入/语义检查完毕才创建新目录；失败不得发布假结果。
        out.mkdir(parents=True, exist_ok=False)
        for name, value in [('score.json',score),('change.json',report)]:
            with (out/name).open('x', encoding='utf-8') as stream:
                json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write('\n')
        print(json.dumps({'status':'通过','scope':report['scope'],'out':str(out),'changes':len(report['changes'])},ensure_ascii=False))
        return 0
    except (ValueError, OSError, ImportError, KeyError, TypeError, StopIteration) as exc:
        print(json.dumps({'status':'失败','error':str(exc),'scope':'未返回可演奏或音乐通过结论'},ensure_ascii=False), file=sys.stderr)
        return 2

if __name__ == '__main__':
    raise SystemExit(main())
