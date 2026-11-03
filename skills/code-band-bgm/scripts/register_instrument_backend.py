"""登记已取得的官方Cnn10及隔离依赖；不下载、安装或加载模型。"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess

MODEL_SHA256 = '5240bdc47444e331bbeb54fd741d2b3f933a4526e84b0c17bf3593df94b13962'
SOURCE_SHA256 = {
    'models.py': '7f9af440395ace5160bbb51d654a0dc35fb887fbf5edecb12da61ff6efb306d9',
    'pytorch_utils.py': '1464fcfbfc0fe4c55f690f6b39e1c80eeed5de1e7fd1b7fd30334d304de7dbe9',
    'class_labels_indices.csv': 'cdd1049833c4b86127c2773ac0d14a2754b6a6d0d1798002ed5c66e699708429',
}
VERSIONS = {'torch': '2.9.1', 'torchlibrosa': '0.1.0', 'librosa': '0.11.0',
            'numpy': '2.2.6', 'scipy': '1.15.3', 'soundfile': '0.13.1'}
PARAMETERS = dict(sample_rate=32000, window_size=1024, hop_size=320, mel_bins=64,
                  fmin=50, fmax=14000, classes_num=527)
SCORE_SEMANTICS = '527类独立sigmoid原始标签分数；未校准，不是乐器存在概率或物理身份核准'
LIMITATIONS = ['FM可能误为拨弦或吉他；候选标签不能直接绑定真实音源。',
               '静音仍有Music残分；合奏中的弱源可能漏检，分数受增益和片段影响。',
               '分离估计轨不是物理分轨；多标签不等于独立声源数、情绪或风格验收。']


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            result.update(block)
    return result.hexdigest()


def dependency_hashes(directory):
    directory = Path(directory)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError('依赖须为已安装的普通本地目录')
    files = list(directory.rglob('*'))
    if any(p.is_symlink() for p in files):
        raise ValueError('隔离依赖目录不能包含符号链接')
    result = {str(p.relative_to(directory)): digest(p) for p in files
              if p.is_file() and p.suffix != '.pyc' and '__pycache__' not in p.parts}
    if not result or 'torchlibrosa/stft.py' not in result:
        raise ValueError('隔离目录缺少真实torchlibrosa实现')
    return result


def check_assets(checkpoint, source_root, dependency_root):
    checkpoint, source_root = Path(checkpoint), Path(source_root)
    if checkpoint.is_symlink() or not checkpoint.is_file() or digest(checkpoint) != MODEL_SHA256:
        raise ValueError('须提供固定SHA256的官方Cnn10本地权重')
    if source_root.is_symlink() or not source_root.is_dir():
        raise ValueError('官方源码须为普通本地目录')
    for name, expected in SOURCE_SHA256.items():
        file = source_root / name
        if file.is_symlink() or not file.is_file() or digest(file) != expected:
            raise ValueError('官方Cnn10源码或527标签表缺失、摘要不符')
    return dependency_hashes(dependency_root)


def read_labels(source_root):
    with (Path(source_root) / 'class_labels_indices.csv').open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 527 or [int(r['index']) for r in rows] != list(range(527)):
        raise ValueError('官方标签表必须唯一顺序覆盖527类')
    return [{'index': int(r['index']), 'mid': r['mid'], 'label': r['display_name']} for r in rows]


def validate_config(config_path):
    path = Path(config_path).resolve()
    cfg = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(cfg, dict) or cfg.get('backend_id') != 'panns_cnn10_cpu' or cfg.get('checkpoint_sha256') != MODEL_SHA256:
        raise ValueError('不是固定官方Cnn10后端配置')
    if cfg.get('versions') != VERSIONS or cfg.get('source_sha256') != SOURCE_SHA256:
        raise ValueError('配置源码或依赖版本不在已测范围内')
    if cfg.get('worker_sha256') != digest(Path(__file__).with_name('panns_instruments.py')):
        raise ValueError('候选运行器源码已改变，请重新审查登记')
    paths = {}
    for key in ('checkpoint', 'source_root', 'dependency_root'):
        name = cfg.get(key)
        if not isinstance(name, str) or not name.strip() or Path(name).is_absolute():
            raise ValueError('权重、源码、依赖必须使用相对配置文件的路径')
        paths[key] = path.parent / name
    hashes = check_assets(paths['checkpoint'], paths['source_root'], paths['dependency_root'])
    if cfg.get('dependency_sha256') != hashes:
        raise ValueError('隔离依赖文件集合或内容已经改变')
    return cfg, paths


def register_backend(checkpoint, source_root, dependency_root, backend_python, out):
    output = Path(out).resolve()
    if output.exists():
        raise ValueError('登记文件已存在，禁止覆盖')
    dependencies = check_assets(checkpoint, source_root, dependency_root)
    if not Path(backend_python).is_file():
        raise ValueError('须显式提供独立本地Python')
    code = ('import sys,json,importlib.metadata;sys.path.insert(0,sys.argv[1]);'
            'print(json.dumps({n:importlib.metadata.version(n) for n in json.loads(sys.argv[2])}))')
    process = subprocess.run([os.path.abspath(backend_python), '-B', '-c', code,
                              str(Path(dependency_root).resolve()), json.dumps(list(VERSIONS))],
                             capture_output=True, text=True, timeout=30, check=False)
    if process.returncode or json.loads(process.stdout) != VERSIONS:
        raise ValueError('独立运行库不符合已实测的六项固定依赖版本')
    cfg = {'schema_version': '1.0.0', 'backend_id': 'panns_cnn10_cpu',
           'checkpoint_sha256': MODEL_SHA256, 'source_sha256': SOURCE_SHA256,
           'dependency_sha256': dependencies, 'versions': VERSIONS,
           'worker_sha256': digest(Path(__file__).with_name('panns_instruments.py')),
           'registrar_source_sha256': digest(__file__),
           'license': {'code': 'MIT', 'checkpoint': 'CC-BY-4.0',
                       'publisher': 'Qiuqiang Kong等PANNs作者；官方Zenodo记录3987831',
                       'checkpoint_url': 'https://zenodo.org/records/3987831',
                       'source_commit': 'd2f4b8c18eab44737fcc0de1248ae21eb43f6aa4'},
           'limits': ['仅多标签候选；不确认原乐器、声源数、情绪或风格。',
                      'FM会误为拨弦/吉他；静音仍有Music残分；合奏与增益变化影响候选。']}
    for key, value in [('checkpoint', checkpoint), ('source_root', source_root), ('dependency_root', dependency_root)]:
        cfg[key] = os.path.relpath(Path(value).resolve(), output.parent)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(cfg, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    return cfg


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'source-root', 'dependency-root', 'backend-python', 'out'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    try:
        cfg = register_backend(args.checkpoint, args.source_root, args.dependency_root, args.backend_python, args.out)
        print(json.dumps({'状态': '已登记候选后端，尚未运行模型', '输出': args.out,
                          '模型SHA256': cfg['checkpoint_sha256']}, ensure_ascii=False))
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        print(json.dumps({'状态': '失败', '错误': str(exc)}, ensure_ascii=False))
        raise SystemExit(2)
