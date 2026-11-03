"""封存操作者已核来源的本地分析资产；不下载、不安装、不加载模型。"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import re


def register(source_root, dependencies, checkpoint, expected_sha256, source_commit, out):
    source, deps, model, output = map(lambda p: Path(p).resolve(), (source_root, dependencies, checkpoint, out))
    if output.exists():
        raise ValueError('配置已存在，禁止覆盖')
    if not source.is_dir() or not deps.is_dir() or not model.is_file() or not re.fullmatch('[a-f0-9]{64}', expected_sha256) or not re.fullmatch('[a-f0-9]{40}', source_commit):
        raise ValueError('须提供本地源码/隔离依赖/模型、完整SHA256及固定源码提交')
    def digest(path):
        result = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024*1024), b''):
                result.update(chunk)
        return result.hexdigest()
    if digest(model) != expected_sha256:
        raise ValueError('模型SHA256与显式期望不一致，未登记')
    if any(p.is_symlink() for root in (source, deps) for p in root.rglob('*')):
        raise ValueError('源码/依赖内部含未验证符号链接')
    files = lambda root, suffix=None: sorted(p for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc' and (suffix is None or p.suffix == suffix))
    code, packages = files(source, '.py'), files(deps)
    if not code or not packages or any(p.is_symlink() for p in code + packages):
        raise ValueError('源码/依赖为空或含未验证文件符号链接')
    value = {'source_root': os.path.relpath(source, output.parent), 'target_dependencies': os.path.relpath(deps, output.parent), 'checkpoint': os.path.relpath(model, output.parent),
             'checkpoint_sha256': expected_sha256, 'source_commit': source_commit,
             'source_sha256': {str(p.relative_to(source)): digest(p) for p in code},
             'target_dependency_sha256': {str(p.relative_to(deps)): digest(p) for p in packages},
             'hash_lock_scope': '只封存指定源码与隔离依赖目录；不推断上游许可、原配器或其他系统包已验证'}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
    return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ('source-root', 'dependencies', 'checkpoint', 'expected-sha256', 'source-commit', 'out'):
        parser.add_argument('--' + field, required=True)
    args = parser.parse_args()
    result = register(args.source_root, args.dependencies, args.checkpoint, args.expected_sha256, args.source_commit, args.out)
    print(json.dumps({'状态': '本地资产已封存，未运行模型', '源码数': len(result['source_sha256']), '依赖文件数': len(result['target_dependency_sha256']), '输出': args.out}, ensure_ascii=False))
