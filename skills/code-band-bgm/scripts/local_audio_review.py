"""可选本地音频语言适配器；权重显式提供，禁止自动联网。"""
from __future__ import annotations

import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from audio_tools import INPUT_FLAGS


QUESTIONS = [
    "请用中文描述这段音乐的主奏、伴奏与低频声音，说明乐器、起音和延音质感；不确定的音色请保留未知。",
    "请用中文描述这段音乐的鼓点、重音、律动和风格，不要只给流派名称，不要编造精确BPM或调性。",
    "请用中文描述这段音乐的空间质感与声部疏密，区分可听到的混响或延迟与不确定推测。",
    "请用中文描述这段音乐的情绪与发展，以及创作不同旋律但同风格新曲应保留的三项声音特征，不要转录旋律或歌词。",
]


def file_sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            result.update(chunk)
    return result.hexdigest()


def segment_starts(seconds):
    """短曲完整读取；长曲取头、中、尾三个窗口，不冒称覆盖全曲。"""
    if seconds <= 30:
        return [0.0]
    return sorted(set([0.0, (seconds - 30) / 2, seconds - 30]))


def decode_segment(path, start):
    try:
        result = subprocess.run([
            "ffmpeg", "-v", "error", "-nostdin", *INPUT_FLAGS,
            "-ss", str(start), "-i", str(path), "-map", "0:a:0", "-vn", "-t", "30",
            "-ar", "16000", "-ac", "1", "-f", "f32le", "pipe:1",
        ], capture_output=True, check=True, timeout=60)
    except subprocess.SubprocessError as exc:
        raise ValueError("参考片段无法在预算内解码，不返回风格描述") from exc
    waveform = np.frombuffer(result.stdout, dtype="<f4").copy()
    if not len(waveform) or not np.isfinite(waveform).all():
        raise ValueError("分析片段为空或含非有限样本")
    return waveform


def use_official_features(model, mx):
    """按HF Qwen2-Audio前处理屏蔽补齐帧，只向语言模型送有效音频token。

    对照transformers/models/qwen2_audio/modeling_qwen2_audio.py的长度公式与
    audio_attention_mask；编码层复用mlx-audio 0.6.3的已有权重和投影。
    """
    state = {}

    def extract(audio):
        wave = np.asarray(audio, dtype=np.float32).reshape(-1)
        values = model._processor.feature_extractor(
            wave, sampling_rate=16000, padding="max_length", return_attention_mask=True, return_tensors="np")
        features = values["input_features"]
        if features.shape != (1, 128, 3000) or not np.isfinite(features).all():
            raise ValueError("官方音频特征维度或数值异常")
        mask = np.asarray(values["attention_mask"])
        if mask.shape != (1, 3000) or not np.isin(mask, [0, 1]).all() or np.any(np.diff(mask[0]) > 0):
            raise ValueError("官方有效帧掩码无效")
        conv, tokens = audio_feature_lengths(int(mask.sum()))
        state.update(conv=conv, tokens=tokens)
        return mx.array(features, dtype=model.audio_tower.conv1.weight.dtype), tokens

    def encode(features):
        import mlx.nn as nn
        encoder = model.audio_tower
        x = nn.gelu(encoder.conv1(features.transpose(0, 2, 1)))
        x = nn.gelu(encoder.conv2(x))
        x = x + encoder.embed_positions[:x.shape[1]]
        mask = mx.where(mx.arange(x.shape[1]) < state["conv"], 0., -float("inf"))
        mask = mask.astype(x.dtype)[None, None, None, :]
        for layer in encoder.layers:
            residual, attention = x, layer.self_attn
            z = layer.self_attn_layer_norm(x)
            batch, frames, _ = z.shape
            def heads(value):
                return value.reshape(batch, frames, attention.num_heads, attention.head_dim).transpose(0, 2, 1, 3)
            q, k, v = heads(attention.q_proj(z)), heads(attention.k_proj(z)), heads(attention.v_proj(z))
            z = mx.fast.scaled_dot_product_attention(q, k, v, scale=attention.scale, mask=mask)
            z = z.transpose(0, 2, 1, 3).reshape(batch, frames, attention.embed_dim)
            x = residual + attention.out_proj(z)
            x = x + layer.fc2(nn.gelu(layer.fc1(layer.final_layer_norm(x))))
        batch, frames, width = x.shape
        x = x[:, :frames // 2 * 2, :].reshape(batch, frames // 2, 2, width).mean(axis=2)
        x = encoder.layer_norm(x)
        x = model.multi_modal_projector(x[:, :state["tokens"], :])
        return x.astype(model.language_model.model.embed_tokens.weight.dtype)

    model._extract_features = extract
    model.get_audio_features = encode


def audio_feature_lengths(valid_mel_frames):
    if type(valid_mel_frames) is not int or not 3 <= valid_mel_frames <= 3000:
        raise ValueError("音频有效特征帧数必须介于3和3000之间")
    conv = (valid_mel_frames - 1) // 2 + 1
    return conv, (conv - 2) // 2 + 1


def describe_audio(input_path, model_path):
    source, model_dir = Path(input_path).resolve(), Path(model_path).resolve()
    if not source.is_file():
        raise ValueError("参考必须是已取得的本地音频文件")
    if source.stat().st_size > 100 * 1024 * 1024:
        raise ValueError("本地参考文件超过100MiB分析预算")
    if not model_dir.is_dir():
        raise ValueError("模型必须是已安装的本地目录；本命令不自动下载或上传")
    config = json.loads((model_dir / "config.json").read_text())
    if not isinstance(config, dict) or config.get("model_type") != "qwen2_audio" or config.get("auto_map"):
        raise ValueError("当前可选适配器仅验证Qwen2-Audio本地权重，不执行自定义模型代码")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    try:
        import mlx.core as mx
        from mlx_audio.stt.utils import load_model
    except ImportError as exc:
        raise ValueError("缺少可选本地音频理解依赖；请在隔离环境安装requirements-audio-understanding.txt") from exc
    if version("mlx-audio") != "0.6.3":
        raise ValueError("当前适配器仅验证mlx-audio 0.6.3，请使用已锁定的可选依赖")
    try:
        probe = subprocess.run([
            "ffprobe", "-v", "error", *INPUT_FLAGS, "-show_format",
            "-of", "json", str(source),
        ], capture_output=True, text=True, check=True, timeout=30)
    except subprocess.SubprocessError as exc:
        raise ValueError("参考文件无法识别为有效音频，不返回风格描述") from exc
    seconds = float(json.loads(probe.stdout)["format"]["duration"])
    if not np.isfinite(seconds) or not 0 < seconds <= 600:
        raise ValueError("本地理解首版只接受0至600秒可解码音频")
    windows = [(start, decode_segment(source, start)) for start in segment_starts(seconds)]
    if not any(np.max(np.abs(wave)) >= 1e-7 for _, wave in windows):
        raise ValueError("参考分析窗口全部静音，不能形成音乐风格证据")
    started = time.monotonic()
    print("正在本地加载音频理解模型，不上传音频。", file=sys.stderr, flush=True)
    try:
        model = load_model(str(model_dir), strict=True)
    except RuntimeError as exc:
        raise ValueError("本地模型加载失败：" + str(exc)) from exc
    use_official_features(model, mx)
    segments = []
    for start, wave in windows:
        answers = []
        if np.max(np.abs(wave)) < 1e-7:
            text, tokens = "该窗口静音，无法推断音乐风格。", 0
        else:
            for question in QUESTIONS:
                mx.random.seed(0)
                try:
                    result = model.generate(wave, prompt=question, max_tokens=280, temperature=0.0)
                except RuntimeError as exc:
                    raise ValueError("本地音频推理失败：" + str(exc)) from exc
                if not result.text.strip():
                    raise ValueError("本地模型未返回描述，不能标记参考理解完成")
                answers.append({"question": question, "answer": result.text.strip(),
                                "generation_tokens": result.generation_tokens,
                                "token_limit_reached": result.generation_tokens >= 280})
                mx.clear_cache()
            text = "\n".join(a["answer"] for a in answers)
            tokens = sum(a["generation_tokens"] for a in answers)
        segments.append({"start_seconds": start, "end_seconds": start + len(wave) / 16000,
                         "decoded_mono_16khz_sha256": hashlib.sha256(wave.tobytes()).hexdigest(),
                         "description": text, "answers": answers, "generation_tokens": tokens})
        print(f"本地描述已完成：{start:.2f}秒起窗口。", file=sys.stderr, flush=True)
        mx.clear_cache()
    return {"status": "通过", "analysis_scope": "local_audio_language_model_estimates",
            "source": {"file_name": source.name, "file_sha256": file_sha256(source), "duration_seconds": seconds},
            "backend": {"library": "mlx-audio", "version": version("mlx-audio"), "mlx_version": version("mlx"),
                        "model_type": "qwen2_audio", "config_sha256": file_sha256(model_dir / "config.json"),
                        "weights_sha256": file_sha256(model_dir / "weights.safetensors"),
                        "processing": "官方WhisperFeatureExtractor，16kHz单声道，30秒补齐；编码注意力屏蔽补齐帧，仅送有效音频token",
                        "adapter_revision": "qwen2_audio_valid_frames_v2",
                        "network": "离线模式；仅显式本地权重与文件", "seed": 0, "temperature": 0.0},
            "questions": QUESTIONS, "segments": segments, "elapsed_seconds": round(time.monotonic() - started, 3),
            "limits": ["模型描述不是人工试听，也不是已校准的乐器概率；须与数值测量及可用资源核对。",
                       "超过30秒时只分析头、中、尾窗口，不能推断未覆盖段落。",
                       "量化MLX实现未证明与原生Transformers数值一致；有效长度修复已通过本地已知讲话对照，不代表音乐细分类准确率已验收。"],
            "style_review": "待人工验收"}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="离线参考音乐语义描述，不上传录音")
    parser.add_argument("--input", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    try:
        report = describe_audio(args.input, args.model)
        with Path(args.out).open("x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        print(json.dumps({"status": "通过", "out": args.out, "style_review": report["style_review"]}, ensure_ascii=False))
    except Exception as exc:
        print(json.dumps({"status": "失败", "message": str(exc)}, ensure_ascii=False))
        sys.exit(2)
