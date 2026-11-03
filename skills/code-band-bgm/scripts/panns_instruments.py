"""独立CPU运行官方Cnn10；输出完整多标签候选，不核准原曲乐器。"""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import socket
import sys
import threading
import time

from register_instrument_backend import PARAMETERS, SCORE_SEMANTICS, LIMITATIONS, digest, read_labels, validate_config


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def run(config_path, inputs_path, out):
    output = Path(out)
    output.mkdir(exist_ok=False)
    started = time.monotonic()
    def peak_rss():
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(value if sys.platform == 'darwin' else value * 1024)
    def guard():
        while True:
            if peak_rss() > 3 * 1024**3 or time.monotonic() - started > 300:
                write(output / 'resource_stop.json', {'状态': '失败', '原因': '3GiB或300秒门禁', 'peak_rss_bytes': peak_rss()})
                os._exit(73)
            time.sleep(.1)
    threading.Thread(target=guard, daemon=True).start()
    try:
        for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            os.environ[name] = '2'
        os.environ.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        def deny(*args, **kwargs):
            raise RuntimeError('本地候选后端禁止网络及下载后备')
        socket.socket.connect = deny
        socket.socket.connect_ex = deny
        socket.create_connection = deny
        config, inputs = Path(config_path).resolve(), Path(inputs_path).resolve()
        config_hash, inputs_hash = digest(config), digest(inputs)
        cfg, paths = validate_config(config)
        sys.path.insert(0, str(paths['dependency_root'].resolve()))
        if any(importlib.metadata.version(name) != version for name, version in cfg['versions'].items()):
            raise ValueError('实际独立依赖版本与登记不一致')
        import numpy as np
        import soundfile as sf
        import torch
        import librosa
        import torchlibrosa
        if Path(torchlibrosa.__file__).resolve().parent != (paths['dependency_root'] / 'torchlibrosa').resolve():
            raise ValueError('实际torchlibrosa实现不在已锁隔离依赖目录')
        sys.path.insert(0, str(paths['source_root'].resolve()))
        import models
        import pytorch_utils
        from models import Cnn10
        if Path(models.__file__).resolve() != (paths['source_root'] / 'models.py').resolve():
            raise ValueError('实际导入模型实现与固定官方源码不符')
        if Path(pytorch_utils.__file__).resolve() != (paths['source_root'] / 'pytorch_utils.py').resolve():
            raise ValueError('实际模型工具实现与固定官方源码不符')
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        torch.manual_seed(20261031)
        torch.hub.load_state_dict_from_url = deny
        with paths['checkpoint'].open('rb') as stream:
            # 验证并从同一已打开文件加载，禁止网络后备或宽松反序列化。
            import hashlib
            h = hashlib.sha256()
            for block in iter(lambda: stream.read(1048576), b''): h.update(block)
            if h.hexdigest() != cfg['checkpoint_sha256']:
                raise ValueError('实际模型摘要不匹配')
            stream.seek(0)
            checkpoint = torch.load(stream, map_location='cpu', weights_only=True)
        model = Cnn10(**PARAMETERS).cpu().float().eval()
        state, actual = checkpoint['model'], model.state_dict()
        if len(actual) != 60 or set(state) != set(actual) or any(state[k].shape != actual[k].shape for k in actual):
            raise ValueError('固定Cnn10实际权重键或形状不匹配')
        loaded = model.load_state_dict(state, strict=True)
        if loaded.missing_keys or loaded.unexpected_keys or any(p.device.type != 'cpu' or p.dtype != torch.float32 for p in model.parameters()):
            raise ValueError('模型必须严格加载并只在CPU以FP32执行')
        del state, checkpoint
        manifest = json.loads(inputs.read_text(encoding='utf-8'))
        windows = manifest.get('windows') if isinstance(manifest, dict) else None
        if not isinstance(windows, list) or not 1 <= len(windows) <= 8:
            raise ValueError('后端须接收一至八个准备窗口')
        ids, results, total = set(), [], 0
        for item in windows:
            if not isinstance(item, dict) or not isinstance(item.get('window_id'), str) or not item['window_id'].strip() or item['window_id'] in ids:
                raise ValueError('准备窗口标识必须非空且唯一')
            ids.add(item['window_id'])
            frames = item.get('frames')
            if type(frames) is not int or not 16000 <= frames <= 960000:
                raise ValueError('后端窗口仅支持0.5至30秒')
            total += frames
            if total > 120 * 32000:
                raise ValueError('后端窗口总长不能超过120秒')
            name = item.get('input_file')
            if not isinstance(name, str) or Path(name).is_absolute() or name != 'analysis_inputs/' + str(len(results)).zfill(3) + '.wav':
                raise ValueError('准备输入必须引用固定本地分析副本')
            file = inputs.parent / name
            if file.is_symlink() or not file.is_file() or digest(file) != item['sha256']:
                raise ValueError('准备输入缺失或摘要改变')
            info = sf.info(file)
            if (info.samplerate, info.channels, info.frames, info.subtype) != (32000, 1, frames, 'FLOAT'):
                raise ValueError('分析副本须为精确32k单声道FLOAT32')
            wave, _ = sf.read(file, dtype='float32')
            if not np.isfinite(wave).all():
                raise ValueError('分析副本含非有限样本')
            with torch.inference_mode():
                scores = model(torch.from_numpy(wave)[None], None)['clipwise_output'].numpy()[0]
            if scores.shape != (527,) or not np.isfinite(scores).all() or not ((scores >= 0) & (scores <= 1)).all():
                raise ValueError('实际模型输出不是527个有限标签分数')
            if digest(file) != item['sha256']:
                raise ValueError('执行期间分析输入改变')
            results.append({'window_id': item['window_id'], 'input_sha256': item['sha256'], 'frames': frames,
                            'sample_rate_hz': 32000, 'channels': 1, 'scores': scores.tolist()})
        validate_config(config)
        if digest(config) != config_hash or digest(inputs) != inputs_hash:
            raise ValueError('执行期间配置或输入清单改变')
        report = {'analysis_scope': 'local_multi_instrument_candidates', 'physical_identity_confirmed': False,
                  'composition_ready': False, 'score_semantics': SCORE_SEMANTICS,
                  'source_sha256': manifest['source_sha256'], 'config_sha256': config_hash,
                  'labels_sha256': cfg['source_sha256']['class_labels_indices.csv'], 'labels': read_labels(paths['source_root']),
                  'windows': results, 'limitations': LIMITATIONS, 'manual_listening': '未进行',
                  'runtime': {key: cfg[key] for key in ('checkpoint_sha256', 'worker_sha256', 'source_sha256', 'dependency_sha256', 'versions')}}
        report['runtime'].update(device='cpu', dtype='float32', threads=2, batch_size=1, network_blocked=True,
                                 weights_only=True, strict_state=True, model_key_count=len(actual),
                                 missing_keys=list(loaded.missing_keys), unexpected_keys=list(loaded.unexpected_keys),
                                 peak_rss_bytes=peak_rss(), elapsed_seconds=time.monotonic() - started,
                                 parameters=PARAMETERS, state_shapes={k: list(v.shape) for k, v in actual.items()})
        write(output / 'report.json', report)
        return report
    except BaseException as exc:
        write(output / 'failure.json', {'状态': '失败', '错误': str(exc)})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'inputs', 'out'): parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    try:
        report = run(args.config, args.inputs, args.out)
        print(json.dumps({'状态': '候选执行完成，未核准原乐器', '窗口数': len(report['windows'])}, ensure_ascii=False))
    except (OSError, ValueError, RuntimeError, ImportError, KeyError) as exc:
        print(json.dumps({'状态': '失败', '错误': str(exc)}, ensure_ascii=False))
        raise SystemExit(2)
