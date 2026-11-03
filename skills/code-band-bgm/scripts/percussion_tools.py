"""把真实逐键探测结果绑定到能力；标准名称不能替代声音证据。"""
from fluid_engine import probe_hits
from score_tools import ROOT, digest, load, require


def add_percussion_checks(result, soundfont):
    table = load(ROOT / 'assets/percussion-map.json')
    hits = table['hits']
    cap = result['capabilities']
    records = []
    for resource in cap['resources']:
        _, bank, program, sha = resource['engine_resource_id'].split(':')
        if int(bank) != 128 or sha != result['asset_sha256']:
            continue
        measured = probe_hits(soundfont, int(bank), int(program), hits)
        passed = {str(p['midi']) for p in measured if p['status'] == '通过'}
        resource['unpitched_hits'] = {name: midi for name, midi in hits.items() if midi in passed}
        failed = sorted(set(hits.values()) - passed, key=int)
        records.append({'instrument_id': resource['instrument_id'], 'hits': measured})
        if failed:
            message = resource['instrument_id'] + '以下GM打击键实测未通过，未注册：' + ','.join(failed)
            resource['limitations'].append(message)
            cap['unresolved'].append(message)
    require(records, 'percussion', '当前基础能力没有可探测的鼓组')
    cap['evidence'] = [e for e in cap['evidence'] if e['evidence_id'] != 'percussion_keys_runtime']
    cap['evidence'].append({'evidence_id': 'percussion_keys_runtime', 'kind': 'command_output',
        'reference': 'doctor .probe.json的percussion_probe_results',
        'description': '每鼓组独立逐键清声、关闭效果后真实渲染，登记通过的键；不等于全部力度、轮换或人工音色验收。'})
    cap['resource_manifest_sha256'] = digest(cap['resources'])
    result['percussion_probe_results'] = records
    result['percussion_map_sha256'] = digest(table)
    return result


def validate_cached_percussion(cap, original):
    if not any(e['evidence_id'] == 'percussion_keys_runtime' for e in cap['evidence']):
        return
    records = original.get('percussion_probe_results')
    require(isinstance(records, list) and all(isinstance(r, dict) and isinstance(r.get('hits'), list) for r in records), 'capabilities/evidence', '缺少逐打击键缓存证据')
    indexed = {r.get('instrument_id'): r['hits'] for r in records}
    require(len(indexed) == len(records), 'capabilities/evidence', '逐打击键缓存资源重复')
    for resource in cap['resources']:
        if ':128:' not in resource['engine_resource_id']:
            continue
        entries = indexed.get(resource['instrument_id'], [])
        require(all(isinstance(p, dict) for p in entries), 'capabilities/evidence', '逐打击键记录格式错误')
        passed = {str(p.get('midi')) for p in entries if p.get('status') == '通过' and
                  type(p.get('frames')) is int and p['frames'] == 4800 and type(p.get('velocity')) is int and p['velocity'] == 90 and
                  type(p.get('peak_linear')) in (int, float) and .0001 < p['peak_linear'] < float('inf')}
        require(set(resource['unpitched_hits'].values()) <= passed, 'capabilities/evidence', '缓存缺少已注册打击键的有效发声记录')
