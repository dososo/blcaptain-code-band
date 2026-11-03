"""登记已有官方6源模型和隔离运行库；不下载、不安装、不加载Torch。"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess

MODEL_SHA256 = '34c22ccb381c6f9fdbf324f04e1e2fe21aaaf293f5ded163a162697ff9a02ddd'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def register_backend(checkpoint, backend_python, out):
    model, output = Path(checkpoint), Path(out).resolve()
    if output.exists():
        raise ValueError('配置已存在，禁止覆盖')
    if model.is_symlink() or not model.is_file() or digest(model) != MODEL_SHA256:
        raise ValueError('须提供已核SHA256的官方htdemucs_6s本地普通文件')
    if not Path(backend_python).is_file():
        raise ValueError('须显式提供已安装Demucs的独立Python')
    probe = ('import importlib.util,importlib.metadata,json;'
             's=importlib.util.find_spec("demucs");'
             'assert s and s.origin;'
             'print(json.dumps({"source_root":str(__import__("pathlib").Path(s.origin).parent),'
             '"versions":{k:importlib.metadata.version(k) for k in '
             '["demucs","torch","torchaudio","numpy","scipy","soundfile"]}}))')
    process = subprocess.run([os.path.abspath(backend_python), '-B', '-c', probe],
                             capture_output=True, text=True, timeout=30, check=False)
    if process.returncode:
        raise ValueError('独立运行库探测失败：' + process.stderr[-2000:])
    runtime = json.loads(process.stdout)
    if runtime['versions']['demucs'] != '4.0.1' or runtime['versions']['torch'] != '2.2.2':
        raise ValueError('本后端仅验证Demucs4.0.1/Torch2.2.2；不静默使用其他版本')
    source = Path(runtime['source_root']).resolve()
    if any(p.is_symlink() for p in source.rglob('*')):
        raise ValueError('Demucs源码含未登记符号链接')
    files = sorted(source.rglob('*.py'))
    if not files:
        raise ValueError('Demucs源码为空')
    value = {'schema_version': '1.0.0', 'backend_id': 'demucs_htdemucs_6s_cpu',
             'checkpoint': os.path.relpath(model.resolve(), output.parent), 'checkpoint_sha256': MODEL_SHA256,
             'source_root': os.path.relpath(source, output.parent),
             'source_sha256': {str(p.relative_to(source)): digest(p) for p in files},
             'versions': runtime['versions'], 'worker_sha256': digest(Path(__file__).with_name('demucs_local.py')),
             'license': {'project': 'MIT', 'model_publisher': 'Meta官方Demucs预训练分发',
                         'model_url': 'https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/5c90dfd2-34c22ccb.th',
                         'documentation_commit': 'e976d93ecc3865e5757426930257e200846a520a',
                         'scope': '发布项目许可不扩大为参考录音或训练录音的再分发授权；不自动打包参考。'},
             'limits': ['6s官方提醒piano质量有限；FM也可能进入piano轨，轨名不证明物理身份。',
                        '只登记本机已存在的源码/版本和固定模型；尚未运行分离。']}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
    return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'backend-python', 'out'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    try:
        result = register_backend(args.checkpoint, args.backend_python, args.out)
        print(json.dumps({'状态': '本地分离后端已登记，未运行模型', '输出': args.out,
                          '源码数': len(result['source_sha256']), '模型SHA256': result['checkpoint_sha256']}, ensure_ascii=False))
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        print(json.dumps({'状态': '失败', '错误': str(exc)}, ensure_ascii=False))
        raise SystemExit(2)
