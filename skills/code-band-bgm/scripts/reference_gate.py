"""参考理解前置门禁：信号存在（G2）与相位（G3）。

阈值来自 2026-11-03 负控实验（work/Claude交付_代码乐团_20261103/evidence/reference_controls），
是看到结果后定的建议值，待真实语料复验。没有信号或相位异常时，节拍、调性、乐器和语义模型一律不运行，
返回结构化拒绝原因，而不是让模型对噪声编出内容。
"""
import math

import numpy as np

SIGNAL_FLOOR_DBFS = -60.0
FOLDDOWN_LIMIT_DB = -6.0
GATE_VERSION = "reference_gate_v1_20261103"


def _db(value):
    return round(20 * math.log10(value), 2) if value > 0 and math.isfinite(value) else None


def measure(frames):
    """frames: (样本, 声道) 浮点数组。峰值在原声道上算，不受降混增益影响。"""
    frames = np.asarray(frames, dtype=np.float64)
    if frames.ndim == 1:
        frames = frames[:, None]
    peak = float(np.max(np.abs(frames))) if frames.size else 0.0
    channel_rms = np.sqrt(np.mean(frames ** 2, axis=0)) if frames.size else np.zeros(frames.shape[1])
    stereo_rms = float(np.sqrt(np.mean(channel_rms ** 2))) if frames.size else 0.0
    result = {"channels": int(frames.shape[1]), "peak_dbfs": _db(peak), "rms_dbfs": _db(stereo_rms),
              "lr_correlation": None, "mono_folddown_loss_db": None}
    if frames.shape[1] >= 2 and stereo_rms > 0:
        left, right = frames[:, 0], frames[:, 1]
        denominator = math.sqrt(float(np.dot(left, left)) * float(np.dot(right, right)))
        result["lr_correlation"] = round(float(np.dot(left, right)) / denominator, 4) if denominator > 0 else None
        mid = frames[:, :2].mean(axis=1)
        mid_rms = float(np.sqrt(np.mean(mid ** 2)))
        # 相对两声道平均 RMS：相关信号约 0 dB，反相信号趋近负无穷
        reference = float(np.sqrt(np.mean(channel_rms[:2] ** 2)))
        result["mono_folddown_loss_db"] = round(20 * math.log10(mid_rms / reference), 2) if mid_rms > 0 else -120.0
    return result


def decide(measured):
    reasons = []
    signal_ok = measured["rms_dbfs"] is not None and measured["rms_dbfs"] >= SIGNAL_FLOOR_DBFS
    if not signal_ok:
        reasons.append({"code": "no_signal", "message": f"整段 RMS 低于 {SIGNAL_FLOOR_DBFS} dBFS，视为没有可分析的信号"})
    correlation, loss = measured["lr_correlation"], measured["mono_folddown_loss_db"]
    phase_ok = not ((correlation is not None and correlation < 0) or (loss is not None and loss < FOLDDOWN_LIMIT_DB))
    if signal_ok and not phase_ok:
        reasons.append({"code": "phase_anomaly",
                        "message": f"左右声道相关 {correlation}、单声道折叠损失 {loss} dB；单声道路径的分析不可信"})
    return {"gate_version": GATE_VERSION, "signal": "通过" if signal_ok else "拒绝", "phase": "通过" if phase_ok else "异常",
            "mono_models_allowed": signal_ok and phase_ok, "reasons": reasons,
            "thresholds": {"signal_floor_dbfs": SIGNAL_FLOOR_DBFS, "folddown_limit_db": FOLDDOWN_LIMIT_DB,
                           "status": "建议值，待真实语料复验"}}


def check_file(path, rate=22050):
    """解码立体声后判定；供 describe、review-reference 等模型命令在加载模型前调用。"""
    from audio_tools import _decode, _probe
    channels = min(_probe(path)["channels"], 2)
    frames = _decode(path, rate=rate, channels=channels).reshape(-1, channels)
    measured = measure(frames)
    return measured, decide(measured)


def require_mono_models(path):
    from audio_tools import AudioError
    measured, decision = check_file(path)
    if not decision["mono_models_allowed"]:
        raise AudioError("reference_signal_refused", "参考没有通过信号或相位门禁，未运行模型",
                         {"measurements": measured, "gate": decision})
    return measured, decision


SEMANTIC_LABEL = {"evidence_level": "无等级", "brief_allowed": False,
                  "notes": "模型自述，未经负控校准；负控上会对纯音、噪声编出乐器和情绪，只供人工复核，不进入简报（G8）。"}
