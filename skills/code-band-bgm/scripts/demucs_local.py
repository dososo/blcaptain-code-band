"""隔离CPU六源估计；唯一已验证本地权重，不下载、不把轨名当乐器身份。"""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import os
import socket
import sys
import threading
import time

from register_separation_backend import MODEL_SHA256, digest

SOURCES = ('drums', 'bass', 'other', 'vocals', 'guitar', 'piano')


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def run(input_path, config_path, out):
    output = Path(out).resolve()
    output.mkdir(exist_ok=False)
    try:
        import resource
        def peak_rss():
            n = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            return int(n if sys.platform == 'darwin' else n * 1024)
        started = time.monotonic()
        def guard():
            while True:
                if peak_rss() > 3 * 1024**3 or time.monotonic() - started > 180:
                    write(output / 'resource_stop.json', {'status': '失败', 'reason': '3GiB/180秒门禁', 'peak_rss_bytes': peak_rss()})
                    os._exit(73)
                time.sleep(.1)
        threading.Thread(target=guard, daemon=True).start()
        for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            os.environ[key] = '2'
        os.environ.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
        def deny(*args, **kwargs):
            raise RuntimeError('本地分离后端禁止网络连接及下载回退')
        socket.socket.connect = deny
        socket.socket.connect_ex = deny
        socket.create_connection = deny
        config = Path(config_path).resolve()
        cfg = json.loads(config.read_text())
        if not isinstance(cfg, dict) or cfg.get('backend_id') != 'demucs_htdemucs_6s_cpu' or cfg.get('checkpoint_sha256') != MODEL_SHA256:
            raise ValueError('不是已验证的固定6源本地配置')
        if cfg.get('worker_sha256') != digest(__file__):
            raise ValueError('分离worker源码已变化，请重新审查并登记')
        for key in ('checkpoint', 'source_root'):
            if not isinstance(cfg.get(key), str) or Path(cfg[key]).is_absolute():
                raise ValueError('模型/源码须按配置文件保存相对路径')
        checkpoint, source = config.parent / cfg['checkpoint'], config.parent / cfg['source_root']
        if checkpoint.is_symlink() or not checkpoint.is_file() or source.is_symlink() or not source.is_dir():
            raise ValueError('模型/源码必须是已存在的本地普通资产')
        locked = cfg.get('source_sha256')
        if not isinstance(locked, dict) or not locked or any(p.is_symlink() for p in source.rglob('*')):
            raise ValueError('Demucs源码摘要缺失或含符号链接')
        actual = {str(p.relative_to(source)) for p in source.rglob('*.py')}
        if actual != set(locked) or any(digest(source / name) != value for name, value in locked.items()):
            raise ValueError('Demucs实现源码缺失、多出或摘要改变')
        versions = cfg.get('versions')
        required = ('demucs', 'torch', 'torchaudio', 'numpy', 'scipy', 'soundfile')
        if not isinstance(versions, dict) or set(versions) != set(required) or versions.get('demucs') != '4.0.1' or versions.get('torch') != '2.2.2':
            raise ValueError('分离依赖版本配置无效')
        if any(importlib.metadata.version(name) != versions[name] for name in required):
            raise ValueError('独立运行库版本与登记不匹配')
        source = source.resolve()
        sys.path.insert(0, str(source.parent))
        import numpy as np
        import soundfile as sf
        import torch
        import demucs
        from demucs.states import load_model
        from demucs.apply import apply_model
        if Path(demucs.__file__).resolve().parent != source:
            raise ValueError('实际导入Demucs位置与固定源码不符')
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        torch.manual_seed(20261031)
        torch.hub.load_state_dict_from_url = deny
        path = Path(input_path)
        if path.is_symlink() or not path.is_file():
            raise ValueError('输入必须是本地普通分析WAV')
        input_hash = digest(path)
        info = sf.info(path)
        if info.samplerate != 44100 or info.channels != 2 or info.subtype != 'FLOAT' or not 22050 <= info.frames <= 30 * 44100:
            raise ValueError('仅接0.5–30秒、44.1k双声道FLOAT32分析窗')
        wave, _ = sf.read(path, dtype='float32', always_2d=True)
        if not np.isfinite(wave).all():
            raise ValueError('输入含NaN/Inf')
        # 同一已打开文件读SHA再seek，防缺失路径触发任何网络后备。
        with checkpoint.open('rb') as stream:
            h = hashlib.sha256()
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                h.update(block)
            if h.hexdigest() != MODEL_SHA256:
                raise ValueError('固定模型SHA256不匹配')
            stream.seek(0)
            package = torch.load(stream, map_location='cpu')
        model = load_model(package, strict=True).eval().cpu().float()
        state_keys, model_keys = set(package['state']), set(model.state_dict())
        missing, unexpected = sorted(model_keys - state_keys), sorted(state_keys - model_keys)
        if missing or unexpected or tuple(model.sources) != SOURCES or model.samplerate != 44100:
            raise ValueError('权重键/六轨名/采样率不匹配')
        if not all(p.device.type == 'cpu' and p.dtype == torch.float32 for p in model.parameters()):
            raise ValueError('禁止非CPU FP32推理')
        del package
        tensor = torch.from_numpy(np.ascontiguousarray(wave.T))
        ref = tensor.mean(dim=0)
        mean, std = ref.mean(), ref.std()
        scale = std if float(std) > 1e-8 else torch.tensor(1.)
        with torch.inference_mode():
            estimate = apply_model(model, ((tensor - mean) / scale)[None], device='cpu',
                                   shifts=0, split=True, overlap=.25, segment=4,
                                   progress=False, num_workers=0)[0] * scale + mean
        if estimate.device.type != 'cpu' or estimate.shape != (6, 2, len(wave)) or not torch.isfinite(estimate).all():
            raise ValueError('输出设备/帧数/有限性不符')
        values = {name: value.T for name, value in zip(SOURCES, estimate.numpy())}
        reconstructed = sum(value.astype(np.float64) for value in values.values())
        values.update(reconstruction=reconstructed, residual=wave.astype(np.float64) - reconstructed)
        outputs = {}
        for name, value in values.items():
            file = output / (name + '.wav')
            sf.write(file, value, 44100, subtype='FLOAT')
            outputs[name] = {'file': file.name, 'sha256': digest(file), 'frames': len(value),
                             'sample_rate_hz': 44100, 'channels': 2, 'sample_format': 'float32'}
        if digest(path) != input_hash or digest(checkpoint) != MODEL_SHA256 or any(digest(source / name) != value for name, value in locked.items()):
            raise ValueError('执行期间输入/模型/实现源码改变')
        energy = {name: float(np.mean(values[name].astype(np.float64)**2)) for name in SOURCES}
        total = sum(energy.values())
        original_rms = float(np.sqrt(np.mean(wave.astype(np.float64)**2)))
        residual_rms = float(np.sqrt(np.mean(values['residual']**2)))
        report = {'status': 'separation_estimates_only', 'ready_for_composition': False, 'labels_are_physical_identities': False,
                  'source': {'sha256': input_hash, 'frames': len(wave), 'sample_rate_hz': 44100, 'channels': 2},
                  'runtime': {'checkpoint_sha256': MODEL_SHA256, 'device': 'cpu', 'dtype': 'float32', 'torch_threads': 2,
                              'network_blocked': True, 'strict_state': True, 'missing_keys': missing, 'unexpected_keys': unexpected,
                              'model_key_count': len(model_keys), 'checkpoint_key_count': len(state_keys), 'versions': versions,
                              'source_sha256': locked, 'worker_sha256': digest(__file__), 'peak_rss_bytes': peak_rss(),
                              'elapsed_seconds': time.monotonic() - started},
                  'parameters': {'shifts': 0, 'split': True, 'overlap': .25, 'segment_seconds': 4, 'num_workers': 0},
                  'outputs': outputs, 'license': cfg.get('license'),
                  'measurements': {'squared_energy_fraction_not_source_truth': {k: v / total if total else 0 for k, v in energy.items()},
                                   'reconstruction_relative_rms_not_accuracy': residual_rms / original_rms if original_rms else None},
                  'limits': ['轨名非物理乐器：FM也能进入piano；官方说明piano分离质量有限。',
                             '输出未逐轨归一/限幅；恢复均值可给弱轨带入DC。', '残差、能量份额不是准确率，未人工试听。',
                             '参考片段只本地分析，不自动公开分发或作为新主题/采样。']}
        write(output / 'report.json', report)
        return report
    except BaseException as exc:
        write(output / 'failure.json', {'status': '失败', 'type': type(exc).__name__, 'message': str(exc)})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'input', 'out'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    try:
        result = run(args.input, args.config, args.out)
        print(json.dumps({'状态': '分离估计已完成', 'ready_for_composition': False,
                          '输出': args.out, '峰RSS字节': result['runtime']['peak_rss_bytes']}, ensure_ascii=False))
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        print(json.dumps({'状态': '失败', '错误': str(exc)}, ensure_ascii=False))
        raise SystemExit(2)
