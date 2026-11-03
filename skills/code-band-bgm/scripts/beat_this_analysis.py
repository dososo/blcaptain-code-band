"""可选BeatThis本地CPU拍点；数值候选不代替拍号、鼓件或听感验收。"""
import hashlib
from importlib.metadata import PackageNotFoundError, version
import os
from pathlib import Path
import pickle
import re
import stat

import numpy as np


def _times(value, duration, label, allow_empty=False):
    times = np.asarray(value)
    if times.ndim != 1 or times.dtype.kind not in 'fiu':
        raise ValueError('BeatThis返回非有限或非一维时间数组：' + label)
    with np.errstate(over='ignore', invalid='ignore'):
        times = times.astype(np.float64)
    if not np.isfinite(times).all():
        raise ValueError('BeatThis返回非有限或非一维时间数组：' + label)
    if (not allow_empty and not len(times)) or np.any(np.diff(times) <= 0):
        raise ValueError('BeatThis返回空或无序时间数组：' + label)
    if np.any(times < 0) or np.any(times > duration):
        raise ValueError('BeatThis拍点超出输入音频时段：' + label)
    return times


def analyze_beats(audio, sample_rate, checkpoint, expected_sha256):
    """校验同一权重句柄后惰性加载CPU后端；没有下载、GPU或唯一BPM推断。"""
    audio = np.asarray(audio)
    if audio.ndim != 1 or audio.dtype.kind not in 'fiu' or type(sample_rate) is not int or sample_rate != 22050:
        raise ValueError('BeatThis要求22050Hz实数单声道波形')
    if not sample_rate * 5 <= len(audio) <= sample_rate * 600:
        raise ValueError('BeatThis只验证5至600秒音频')
    with np.errstate(over='ignore', invalid='ignore'):
        wave = audio.astype(np.float32)
    if not np.isfinite(audio).all() or not np.isfinite(wave).all() or float(np.max(np.abs(wave))) <= 1e-7:
        raise ValueError('BeatThis输入须有限且非静音')
    if not isinstance(checkpoint, (str, Path)) or not str(checkpoint).strip():
        raise ValueError('必须显式提供已有本地普通模型文件')
    if not isinstance(expected_sha256, str) or not re.fullmatch(r'[0-9a-fA-F]{64}', expected_sha256):
        raise ValueError('必须显式提供模型SHA256摘要')
    try:
        descriptor = os.open(checkpoint, os.O_RDONLY | getattr(os, 'O_NONBLOCK', 0))
    except OSError as error:
        raise ValueError('BeatThis模型本地文件不可读；不会下载或回退') from error
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise ValueError('BeatThis模型必须是本地普通文件')
    with os.fdopen(descriptor, 'rb') as stream:
        checksum = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            checksum.update(block)
        actual_sha256 = checksum.hexdigest()
        if actual_sha256 != expected_sha256.lower():
            raise ValueError('BeatThis模型SHA256不符')
        stream.seek(0)
        try:
            beat_version, torch_version = version('beat-this'), version('torch')
            torch_major = re.match(r'^(\d+)\.', torch_version)
            if beat_version != '1.1.0' or not torch_major or int(torch_major[1]) < 2:
                raise ValueError('当前只验证beat-this 1.1.0和PyTorch>=2')
            import torch
            from beat_this.inference import Audio2Beats
        except (ImportError, OSError, PackageNotFoundError) as error:
            raise ValueError('缺少BeatThis可选本地依赖；不会自动安装或下载模型') from error
        previous_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(4)
            # 同一已校验句柄交给torch.load，路径消失也不进入上游短名下载分支。
            tracker = Audio2Beats(checkpoint_path=stream, device='cpu', float16=False, dbn=False)
            parameters = list(tracker.model.parameters())
            if not parameters or any(p.device.type != 'cpu' or p.dtype != torch.float32 for p in parameters):
                raise ValueError('BeatThis权重未实际使用CPU FP32，拒绝继续')
            with torch.inference_mode():
                beats, downbeats = tracker(wave, sample_rate)
        except (OSError, RuntimeError, TypeError, KeyError, pickle.UnpicklingError) as error:
            raise ValueError('BeatThis本地CPU加载或推理失败；不会联网或切换后端') from error
        finally:
            torch.set_num_threads(previous_threads)
    duration = len(wave) / sample_rate
    beats = _times(beats, duration, '拍点')
    downbeats = _times(downbeats, duration, '强拍', allow_empty=True)
    if len(downbeats) and not np.isin(downbeats, beats).all():
        raise ValueError('BeatThis强拍不是官方后处理拍点的子集')
    return {'status': '通过', 'status_scope': 'rhythm_measurement_candidates',
        'backend': {'library': 'beat_this', 'version': beat_version, 'torch_version': torch_version,
            'device': 'cpu', 'dtype': 'float32', 'threads': 4, 'dbn': False,
            'method': '官方Audio2Beats与minimal后处理', 'model_frame_rate_hz': 50,
            'model_sha256': actual_sha256, 'checkpoint_access': '同一已校验二进制句柄，无路径重开或短名下载'},
        'observation': {'start_seconds': 0.0, 'end_seconds': duration, 'frames': len(wave),
            'sample_rate_hz': sample_rate, 'channels': 1, 'time_basis': '相对输入波形起点的秒；窗口绝对偏移由调用者保留'},
        'beats_seconds': beats.tolist(), 'downbeats_seconds': downbeats.tolist(),
        'downbeat_status': '模型强拍候选' if len(downbeats) else '未知：模型未检出强拍',
        'confidence': '未校准；原始模型时间不是人工逐拍标注',
        'limits': ['可能漏拍、半倍/倍拍或换拍级，不能由原始拍数机械给出唯一速度。',
            '不推断4/4、具体鼓件、摇摆/切分标签、原曲和弦或旋律。',
            '强拍已由官方后处理对齐最近拍点；50Hz量化不代表真实鼓起音精度。',
            '测量通过只指本输入范围；不代表参考风格理解或人工听感通过。']}
