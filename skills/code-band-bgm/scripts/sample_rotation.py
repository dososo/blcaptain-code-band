"""在可证实的固定轮换采样预设之间，确定性拆分声部。

输出是新的可编辑乐谱和可重渲事件，原乐谱保持不变。只支持已登记的
VCSL独立轮换位；SoundFont多层zone不视为轮换样本。
"""
import copy

from score_tools import compile_score, digest, random_int, require

# 源库登记了两次独立录音；仍按每个音符实查音区/力度及sample ID。
VCSL_PACK_SHA256 = 'a5f223c7a587cf646fa4aef094bc132ee2d1eb1291bceb18527e78dcebe03052'
REGISTERED_PROGRAMS = {
    'vcsl_cajon_rr1': 4, 'vcsl_cajon_rr2': 5,
    'vcsl_frame_drum_rr1': 6, 'vcsl_frame_drum_rr2': 7,
    'vcsl_mbira_rr1': 9, 'vcsl_mbira_rr2': 10,
}

SUPPORTED_PAIRS = {
    ('vcsl_cajon_rr1', 'vcsl_cajon_rr2'),
    ('vcsl_frame_drum_rr1', 'vcsl_frame_drum_rr2'),
    ('vcsl_mbira_rr1', 'vcsl_mbira_rr2'),
}


def validate_rotation_parent(frozen, score, capabilities, asset_sha256, primary_inventory, additional_inventories):
    """检查原生冻结事件；外加控制或已有演奏与重新编译不一致时拒绝。"""
    from score_tools import validate_frozen
    require(isinstance(frozen, dict) and 'document_kind' not in frozen and 'events' in frozen
            and 'engine' in frozen, 'sample_rotation/events', '只接受原生冻结事件，不接受合奏封套')
    require('control_events' not in frozen or frozen['control_events'] == [],
            'sample_rotation/control_events', '原冻结事件包含外加控制，当前拆分会丢失控制，拒绝处理')
    inventories = {asset_sha256: primary_inventory, **(additional_inventories or {})}
    validate_frozen(frozen, score, capabilities, asset_sha256,
                    list((additional_inventories or {}).keys()), inventories)
    baseline = compile_score(score, capabilities, asset_sha256, primary_inventory, additional_inventories)
    require(frozen['events'] == baseline['events'] and frozen['parts'] == baseline['parts'],
            'sample_rotation/events', '原冻结事件与当前确定性编译不同，拒绝重抽或丢失已解析演奏')
    return frozen


def validate_rotation_output(parent_events, revised_events, original_part_id, generated_part_ids):
    """对比父冻结事件与输出；非目标声部严格相等，目标音符的音乐属性不变。"""
    before_list = [e for e in parent_events['events'] if e['part_id'] != original_part_id]
    after_list = [e for e in revised_events['events'] if e['part_id'] not in generated_part_ids]
    before = {e['event_id']: e for e in before_list}
    after = {e['event_id']: e for e in after_list}
    require(len(before) == len(before_list) == len(after_list) == len(after) and before == after,
            'sample_rotation/events', '非目标声部与父冻结事件不同或重复')
    original_list = [e for e in parent_events['events'] if e['part_id'] == original_part_id]
    revised_list = [e for e in revised_events['events'] if e['part_id'] in generated_part_ids]
    original = {e['note_id']: e for e in original_list}
    revised = {e['note_id']: e for e in revised_list}
    require(len(original) == len(original_list) == len(revised_list) == len(revised)
            and set(original) == set(revised), 'sample_rotation/events', '轮换后目标音符身份不同或重复')
    fields = ('onset_frame', 'gate_frames', 'midi', 'velocity', 'articulation',
              'timing_offset_frames', 'quantization_frames', 'notated_onset_tick')
    require(all(all(original[n][k] == revised[n][k] for k in fields) for n in original),
            'sample_rotation/events', '轮换改变了目标音符音乐事件')
    return True


def rotate_score(score, capabilities, asset_sha256, primary_inventory,
                 additional_inventories, part_id, resources):
    """返回(新谱, 冻结事件, 变更报告)。固定score/资源/seed时结果确定。

    前置条件：调用方已检查原冻结事件无外加控制，且其事件与当前编译结果一致。
    第一轮拒绝被任何效果/触发引用的声部，避免改变效果路由或duck语义；
    拒绝人性化声部，避免拆声部改变随机流；不支持循环边界轮换。
    """
    require(isinstance(resources, list) and len(resources) == 2
            and tuple(resources) in SUPPORTED_PAIRS, 'sample_rotation/resources',
            '仅支持登记的两枚独立VCSL轮换位且顺序须为rr1、rr2')
    registered = {r['instrument_id']: r for r in capabilities['resources']}
    for resource in resources:
        require(resource in registered and registered[resource]['engine_resource_id'] ==
                f"sf:0:{REGISTERED_PROGRAMS[resource]}:{VCSL_PACK_SHA256}",
                'sample_rotation/resources', '轮换预设不等于已核验的固定VCSL音色库摘要与映射')
    require(not score['render']['loop']['enabled'], 'sample_rotation/loop',
            '暂不支持循环的跨周期轮换')
    parts = [p for p in score['parts'] if p['part_id'] == part_id]
    require(len(parts) == 1, 'sample_rotation/part_id', '目标声部不存在或重复')
    part = parts[0]
    require(part['resource_binding']['instrument_id'] in resources,
            'sample_rotation/resources', '目标声部须已经绑定本组轮换资源')
    require(not part['humanize']['enabled'], 'sample_rotation/humanize',
            '拆声部会改变人性化随机流；请先关闭此声部人性化')
    require(len(part['notes']) >= 2, 'sample_rotation/notes', '至少两个音符才有轮换')
    for effect in score['mix']['effects']:
        params = effect.get('parameters', {})
        require(part_id not in effect.get('target_part_ids', [])
                and params.get('trigger_part_id') != part_id
                and part_id not in params.get('trigger_part_ids', []),
                'sample_rotation/effects', '该声部参与效果或触发，拆分可能改变路由，暂不支持')
    new_ids = [part_id + '_rr1', part_id + '_rr2']
    require(not any(p['part_id'] in new_ids for p in score['parts']),
            'sample_rotation/part_id', '生成的轮换声部ID与已有声部冲突')
    # 对每个预设编译全部原音符，逐音核最终力度、采样及映射。
    variants = []
    for resource in resources:
        candidate = copy.deepcopy(score)
        next(p for p in candidate['parts'] if p['part_id'] == part_id)['resource_binding']['instrument_id'] = resource
        variants.append(compile_score(candidate, capabilities, asset_sha256,
                                      primary_inventory, additional_inventories))
    source_events = [{e['note_id']: e for e in v['events'] if e['part_id'] == part_id} for v in variants]
    for note in part['notes']:
        a, b = (x[note['note_id']] for x in source_events)
        require(a['midi'] == b['midi'] and a['onset_frame'] == b['onset_frame']
                and a['gate_frames'] == b['gate_frames'] and a['velocity'] == b['velocity'],
                'sample_rotation/events', '轮换位产生不同的音乐事件')
        require(set(a['soundfont_sample_ids']).isdisjoint(b['soundfont_sample_ids']),
                'sample_rotation/samples', '两轮换位不是不同真实采样，不能称为轮换')
    ordered = sorted(part['notes'], key=lambda n: (n['onset_tick'], n['note_id']))
    start = random_int(score['randomness']['master_seed'], 'sample_rotation',
                       score['score_id'] + '/' + part_id, 2)
    buckets = [[], []]
    assignments = []
    for index, note in enumerate(ordered):
        slot = (start + index) % 2
        buckets[slot].append(copy.deepcopy(note))
        assignments.append({'note_id': note['note_id'], 'slot': slot + 1,
                            'resource_id': resources[slot],
                            'samples': source_events[slot][note['note_id']]['soundfont_sample_ids']})
    revised = copy.deepcopy(score)
    offset = next(i for i, p in enumerate(revised['parts']) if p['part_id'] == part_id)
    clones = []
    for slot in range(2):
        clone = copy.deepcopy(part)
        clone['part_id'] = new_ids[slot]
        clone['resource_binding']['instrument_id'] = resources[slot]
        clone['notes'] = buckets[slot]
        clones.append(clone)
    revised['parts'][offset:offset + 1] = clones
    compiled = compile_score(revised, capabilities, asset_sha256,
                             primary_inventory, additional_inventories)
    # 其他声部及其随机结果必须逐字段保持，目标声部保留原note_id和时序。
    original = variants[0]
    before = {e['event_id']: e for e in original['events'] if e['part_id'] != part_id}
    after = {e['event_id']: e for e in compiled['events'] if e['part_id'] not in new_ids}
    require(before == after, 'sample_rotation/events', '其他声部事件意外变化')
    for event in (e for e in compiled['events'] if e['part_id'] in new_ids):
        baseline = source_events[0][event['note_id']]
        require(all(event[k] == baseline[k] for k in ('onset_frame', 'gate_frames', 'midi', 'velocity', 'articulation', 'timing_offset_frames', 'quantization_frames', 'notated_onset_tick')),
                'sample_rotation/events', '轮换导致音乐事件意外变化')
    report = {'type': 'deterministic_fixed_preset_rotation',
              'parent_score_sha256': digest(score), 'score_sha256': digest(revised),
              'source_part_id': part_id,
              'generated_part_ids': new_ids, 'master_seed': score['randomness']['master_seed'],
              'namespace': 'sample_rotation', 'first_slot': start + 1, 'assignments': assignments,
              'original_score_unchanged': True, 'other_part_events_equal': True,
              'limitations': ['固定已登记两个预设；不随机选择SoundFont同预设zone',
                              '目标原note_id保留但event_id因拆分声部变化',
                              '不支持目标声部人性化、引用其效果/触发及循环',
                              '原冻结事件含外加CC控制时不得调用；其他事件必须与当前编译完全一致',
                              '不能证明听感真实或两段录音属于完整演奏法']}
    return revised, compiled, report


def main(argv=None):
    """独立入口；不调用shell，输出新乐谱、冻结事件和变更报告。"""
    import argparse
    import json
    import sys
    from pathlib import Path
    from soundfont_inventory import inventory
    from soundfont_packs import load_packs

    parser = argparse.ArgumentParser(description='已登记双采样预设的确定性轮换')
    parser.add_argument('--score', required=True)
    parser.add_argument('--events', required=True, help='父乐谱的冻结原生事件，外加控制事件暂不支持')
    parser.add_argument('--part', required=True)
    parser.add_argument('--resources', nargs=2, required=True, metavar=('RR1', 'RR2'))
    parser.add_argument('--capabilities', required=True)
    parser.add_argument('--soundfont', required=True)
    parser.add_argument('--soundfont-packs', required=True)
    parser.add_argument('--out', required=True, help='不存在的新输出目录')
    args = parser.parse_args(argv)
    try:
        output = Path(args.out)
        if output.exists():
            raise FileExistsError('输出目录已存在，拒绝覆盖')
        from score_tools import load
        score = load(Path(args.score))
        frozen = load(Path(args.events))
        caps = load(Path(args.capabilities))
        main_inventory = inventory(args.soundfont)
        packs = load_packs(args.soundfont_packs)
        validate_rotation_parent(frozen, score, caps, main_inventory['source']['sha256'], main_inventory, packs['inventories'])
        revised, events, report = rotate_score(score, caps, main_inventory['source']['sha256'],
                                               main_inventory, packs['inventories'], args.part, args.resources)
        validate_rotation_output(frozen, events, args.part, report['generated_part_ids'])
        report['parent_events_sha256'] = digest(frozen)
        report['parent_external_controls'] = 'none'
        output.mkdir(parents=True)
        for name, content in [('score.json', revised), ('events.json', events), ('rotation.json', report)]:
            with (output / name).open('x', encoding='utf-8') as stream:
                json.dump(content, stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write('\n')
        print(json.dumps({'status': '通过', 'scope': '乐谱和真实音区编译；音乐听感未验',
                          'output': str(output), 'slots': len(report['assignments'])}, ensure_ascii=False))
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({'status': '失败', 'error_type': type(exc).__name__, 'message': str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
