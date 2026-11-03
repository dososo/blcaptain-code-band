"""无模型的本地角色证据校验与打包；结构通过不代表配器或风格验收。"""
import copy
import hashlib
import json
import math
from pathlib import Path
import shutil

ROLES = ('drums', 'bass', 'harmonic_rhythm', 'foreground')
KINDS = ('measurement', 'source_separation', 'controlled_render', 'official_comparison')


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def _digest(path):
    try:
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError('角色证据文件必须真实存在且非空')
        result = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                result.update(block)
        return result.hexdigest()
    except OSError as exc:
        raise ValueError('角色证据文件无法读取') from exc


def validate_role_evidence(evidence_path, source_sha256):
    """相对文件路径以证据JSON父目录解析；返回原报告，不自动签收听感。"""
    if not _sha(source_sha256):
        raise ValueError('本次真实参考的SHA256无效')
    try:
        path = Path(evidence_path).resolve()
        def reject_constant(_):
            raise ValueError('角色证据JSON不能包含非有限数值')
        report = json.loads(path.read_text(encoding='utf-8'), parse_constant=reject_constant)
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError('角色证据文件不存在或不是有效JSON') from exc
    if not isinstance(report, dict) or report.get('analysis_scope') != 'local_role_measurement_evidence':
        raise ValueError('角色证据必须声明本地角色测量范围')
    source = report.get('source')
    if not isinstance(source, dict) or source.get('file_sha256') != source_sha256:
        raise ValueError('角色证据与本次真实参考源SHA256不匹配')
    if report.get('composition_ready') is not False:
        raise ValueError('角色测量不能自动标记创作或风格已核准')
    if report.get('manual_listening') not in ('未进行', '待人工验收'):
        raise ValueError('人工听感必须明确标记未进行或待人工验收')
    evidence = report.get('evidence')
    if not isinstance(evidence, list) or not evidence:
        raise ValueError('至少需要一份真实测量或对照证据文件')
    identities = set()
    for item in evidence:
        if not isinstance(item, dict) or not _text(item.get('evidence_id')) or item['evidence_id'] in identities:
            raise ValueError('证据标识必须非空且不能重复')
        identities.add(item['evidence_id'])
        if item.get('kind') not in KINDS:
            raise ValueError('证据类型仅支持测量、源分离、控制演奏或官方对照')
        name = item.get('path')
        if not _text(name) or Path(name).is_absolute():
            raise ValueError('证据文件必须使用相对证据包父目录的路径')
        if not _sha(item.get('sha256')) or _digest(path.parent / name) != item['sha256']:
            raise ValueError('证据文件SHA256缺失、无效或内容已改变')
    observations = report.get('observations')
    if not isinstance(observations, dict):
        raise ValueError('角色证据缺少鼓、低音、和声节奏与前景观察')
    for role in ROLES:
        item = observations.get(role)
        if not isinstance(item, dict) or not _text(item.get('observation')):
            raise ValueError('四种角色都必须包含非空观察')
        limits, refs = item.get('limitations'), item.get('evidence_ids')
        if not isinstance(limits, list) or not limits or not all(_text(v) for v in limits):
            raise ValueError('每个角色都必须记录非空限制列表')
        if not isinstance(refs, list) or not refs or not all(_text(v) and v in identities for v in refs):
            raise ValueError('每个角色都必须引用已存在的非空证据标识')
    windows = report.get('windows')
    if not isinstance(windows, list) or not windows:
        raise ValueError('角色证据必须声明至少一个真实分析窗口')
    for window in windows:
        if not isinstance(window, dict):
            raise ValueError('分析窗口必须包含起点和时长')
        start, duration = window.get('start_seconds'), window.get('duration_seconds')
        try:
            valid = (type(start) in (int, float) and type(duration) in (int, float)
                     and start >= 0 and duration > 0 and math.isfinite(start + duration))
        except OverflowError:
            valid = False
        if not valid:
            raise ValueError('分析窗口必须是有限非负起点和有限正时长')
    return report


def copy_role_evidence(evidence_path, source_sha256, output_directory):
    """复制调用方选定的小报告并重写相对引用；不覆盖旧证据或修改输入。"""
    report = validate_role_evidence(evidence_path, source_sha256)
    source = Path(evidence_path).resolve()
    source_hash = _digest(source)
    output = Path(output_directory)
    package, files = output / 'reference_evidence.json', output / 'reference_role_sources'
    if package.exists() or files.exists():
        raise ValueError('作品角色证据或来源目录已存在，禁止覆盖')
    try:
        output.mkdir(parents=True, exist_ok=True)
        files.mkdir(exist_ok=False)
        published = copy.deepcopy(report)
        for index, item in enumerate(published['evidence']):
            original = source.parent / item['path']
            name = str(index) + '_' + item['sha256'] + original.suffix
            target = files / name
            with original.open('rb') as reader, target.open('xb') as writer:
                shutil.copyfileobj(reader, writer)
            if _digest(target) != item['sha256']:
                raise ValueError('证据复制期间内容改变，拒绝打包')
            item['path'] = str(Path('reference_role_sources') / name)
        if _digest(source) != source_hash:
            raise ValueError('打包期间角色证据包改变，拒绝交付')
        with package.open('x', encoding='utf-8') as stream:
            json.dump(published, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
        validate_role_evidence(package, source_sha256)
        return package
    except OSError as exc:
        raise ValueError('角色证据无法复制或写入作品目录') from exc
