"""本地音频检查与保守参考分析；不上传、不转录旋律、不处理登录态。"""
from __future__ import annotations

import hashlib
import http.client
import ipaddress
import json
import math
import re
import socket
import ssl
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.parse import urljoin, urlsplit
from urllib import error as urlerror, request as urlrequest

import numpy as np
from scipy import signal

# 按文件路径加载本模块时（测试用 spec_from_file_location），同目录的门禁模块也要可导入
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

VERSION = "0.1.0"
# 不允许媒体播放列表再发起网络请求或间接读取其他文件。
INPUT_FLAGS = ["-protocol_whitelist", "file,pipe", "-format_whitelist", "wav,mp3,aac,flac,ogg,mov,matroska,aiff,au"]
LIMITS = {"max_bytes": 100 * 1024 * 1024, "max_seconds": 600,
          "network_timeout_seconds": 20, "process_timeout_seconds": 90,
          "max_redirects": 5, "retries": 0, "download_total_seconds": 60,
          "max_decoded_samples": 64000000, "estimate_segment_seconds": 90}


class AudioError(ValueError):
    def __init__(self, code, message, details=None):
        super().__init__(message)
        self.code, self.message, self.details = code, message, details or {}


def _run(args):
    try:
        result = subprocess.run(args, capture_output=True, timeout=LIMITS["process_timeout_seconds"], check=False)
    except FileNotFoundError:
        raise AudioError("dependency_missing", f"缺少本地工具：{args[0]}") from None
    except subprocess.TimeoutExpired:
        raise AudioError("process_timeout", "本地音频工具超时；未生成成功结果") from None
    if result.returncode:
        # FFmpeg 的原始错误可能含带凭据链接或个人路径，不转发它。
        raise AudioError("decode_failed", "媒体无法完整解码或测量", {"tool": args[0], "exit_code": result.returncode})
    return result


def _tools():
    return {tool: _run([tool, "-version"]).stdout.decode().splitlines()[0] for tool in ("ffmpeg", "ffprobe")} | {
        "audio_tools": VERSION, "numpy": np.__version__}


def _hash_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _local(path):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise AudioError("file_missing", "参考或成品文件不存在，或不是普通文件")
    if path.stat().st_size > LIMITS["max_bytes"]:
        raise AudioError("file_too_large", "文件超过本地分析大小上限", LIMITS)
    return path


def _probe(path):
    result = _run(["ffprobe", "-v", "error", *INPUT_FLAGS, "-show_streams", "-show_format", "-of", "json", str(path)])
    doc = json.loads(result.stdout)
    streams = [s for s in doc.get("streams", []) if s.get("codec_type") == "audio"]
    if not streams:
        raise AudioError("no_audio", "媒体中没有可解码音频流")
    s = streams[0]
    duration = float(s.get("duration", doc.get("format", {}).get("duration", 0)))
    if not math.isfinite(duration) or duration <= 0 or duration > LIMITS["max_seconds"]:
        raise AudioError("duration_limit", "无法确认时长，或媒体超过分析时长上限", LIMITS)
    rate, channels = int(s["sample_rate"]), int(s["channels"])
    if not (1 <= channels <= 8 and 8000 <= rate <= 192000):
        raise AudioError("media_limit", "音频采样率或声道数超出当前工具范围")
    return {"container": doc["format"]["format_name"], "codec": s["codec_name"],
            "sample_rate_hz": rate, "channels": channels,
            "bits_per_sample": int(s.get("bits_per_raw_sample") or s.get("bits_per_sample") or 0),
            "container_duration_seconds": duration, "audio_stream_index": s["index"]}


def _decode(path, rate=None, channels=None):
    meta = _probe(path)
    sample_count = meta["container_duration_seconds"] * (rate or meta["sample_rate_hz"]) * (channels or meta["channels"])
    if sample_count > LIMITS["max_decoded_samples"]:
        raise AudioError("decode_memory_limit", "媒体解码样本数超过当前内存预算", LIMITS)
    args = ["ffmpeg", "-v", "error", "-nostdin", *INPUT_FLAGS, "-i", str(path), "-map", "0:a:0", "-vn",
            "-t", str(LIMITS["max_seconds"] + 1)]
    if rate:
        args += ["-ar", str(rate)]
    if channels:
        args += ["-ac", str(channels)]
    args += ["-f", "f64le", "-acodec", "pcm_f64le", "pipe:1"]
    result = _run(args)
    if result.stderr.strip():
        raise AudioError("decode_failed", "解码器报告错误；不把部分解码结果视为完整媒体")
    return np.frombuffer(result.stdout, dtype="<f8")


def _db(value):
    return round(20 * math.log10(value), 4) if value > 0 and math.isfinite(value) else None


def _windows(audio, rate, seconds=0.25):
    size = max(1, round(rate * seconds))
    return [{"start_frame": start, "end_frame": min(start + size, len(audio)),
             "start_seconds": start / rate, "end_seconds": min(start + size, len(audio)) / rate,
             "rms_dbfs": _db(float(np.sqrt(np.mean(audio[start:start + size] ** 2))))}
            for start in range(0, len(audio), size)]


def _activity_report(audio, rate, expected=None):
    """逐秒去DC活动诊断；近静音不自动判音乐失败，也不推断尾音。"""
    expected = expected or {}
    total, threshold = len(audio), -70
    boundary = {"status": "未知", "start_frame": None,
                "notes": "未明确声明append尾音，不能把末段自动视为合法尾音"}
    if expected.get("tail_policy") == "append":
        tail = expected.get("tail_frames")
        if type(tail) is int and 0 <= tail <= total:
            boundary = {"status": "已声明", "start_frame": total - tail,
                        "tail_frames": tail, "notes": "仅来自expected，不证明该区间应该静音"}
        else:
            boundary["status"] = "声明无效，需核"
    windows, intervals = [], []
    unassessed = 0
    for start in range(0, total, rate):
        end = min(start + rate, total)
        block = audio[start:end]
        if len(block) < 2:
            unassessed += end - start
            windows.append({"start_frame": start, "end_frame": end,
                            "start_seconds": start / rate, "end_seconds": end / rate,
                            "partial_window": True, "near_silent": None,
                            "raw_rms_dbfs": [_db(float(abs(v))) for v in block[0]],
                            "reason": "单帧无法区分直流与瞬态，活动未评估"})
            continue
        scale = np.max(np.abs(block), axis=0)
        normalized = np.divide(block, scale, out=np.zeros_like(block), where=scale > 0)
        # 先移除首样本再求均值，避免纯常数受均值舍入产生假AC。
        # 归一后求平方再还原量级，避免有限FLOAT输入平方溢出。
        shifted = normalized - normalized[0]
        mean = np.mean(shifted, axis=0)
        ac = shifted - mean
        rms = np.sqrt(np.mean(ac ** 2, axis=0)) * scale
        dc = (normalized[0] + mean) * scale
        raw_rms = np.sqrt(np.mean(normalized ** 2, axis=0)) * scale
        if not all(np.isfinite(x).all() for x in (rms, dc, raw_rms)):
            return {"status": "未评估", "reason": "去DC测量数值不可靠，未发布非有限活动诊断",
                    "analyzed_frames": start, "requested_frames": total}
        maximum = float(np.max(rms))
        quiet = maximum < 10 ** (threshold / 20)
        windows.append({"start_frame": start, "end_frame": end,
                        "start_seconds": start / rate, "end_seconds": end / rate,
                        "partial_window": end - start < rate,
                        "dc_mean": [float(v) for v in dc],
                        "raw_rms_dbfs": [_db(float(v)) for v in raw_rms],
                        "ac_rms_dbfs": [_db(float(v)) for v in rms],
                        "max_channel_ac_rms": maximum, "near_silent": quiet})
        if quiet:
            if intervals and intervals[-1]["end_frame"] == start:
                intervals[-1]["end_frame"] = end
            else:
                intervals.append({"start_frame": start, "end_frame": end})
    for interval in intervals:
        start, end = interval["start_frame"], interval["end_frame"]
        split = boundary["start_frame"]
        body = max(0, min(end, split) - start) if split is not None else None
        tail = end - start - body if body is not None else None
        interval.update({"start_seconds": start / rate, "end_seconds": end / rate,
                         "duration_seconds": (end - start) / rate,
                         "body_overlap_frames": body, "tail_overlap_frames": tail,
                         "region": "尾音边界未知" if split is None else
                         "跨主体与尾音" if body and tail else "主体内" if body else "声明尾音内"})
    return {"status": "需核" if intervals or unassessed or boundary["status"] == "声明无效，需核" else "未发现1秒近静音窗",
            "method": "每窗逐声道减均值后RMS；取最大声道，避免反相或多声道混合抵消",
            "window_seconds": 1, "threshold_dbfs": threshold,
            "classification": "所有声道AC RMS均严格低于阈值为近静音；不含等于阈值",
            "zero_dbfs_representation": "零能量以null表示，不输出负无穷",
            "analyzed_frames": total - unassessed, "unassessed_frames": unassessed, "sample_rate_hz": rate,
            "near_silent_frames": sum(x["end_frame"] - x["start_frame"] for x in intervals),
            "windows": windows, "near_silent_intervals": intervals, "tail_boundary": boundary,
            "limits": "仅逐秒活动诊断；末窗仅有一帧时不能区分直流与瞬态，标记未评估。不能发现窗内所有短空隙。休止、尾音及低动态可能有意，需核谱/用途/听感，不判音乐质量或自动失败"}


def _loudness(path):
    args = ["ffmpeg", "-hide_banner", "-nostdin", *INPUT_FLAGS, "-i", str(path), "-map", "0:a:0", "-vn",
            "-af", "ebur128=peak=true", "-f", "null", "-"]
    log = _run(args).stderr.decode(errors="replace")
    summary = log.rsplit("Summary:", 1)[-1]
    def field(pattern):
        match = re.search(pattern, summary)
        value = float(match.group(1)) if match else None
        return value if value is not None and math.isfinite(value) else None
    return {"method": "FFmpeg ebur128=peak=true（过采样真峰值）",
            "integrated_lufs": field(r"I:\s+([-\d.inf]+) LUFS"),
            "loudness_range_lu": field(r"LRA:\s+([-\d.inf]+) LU"),
            "true_peak_dbtp": field(r"Peak:\s+([-\d.inf]+) dBFS"),
            "raw_summary": summary.strip()}


def _continuous_tail_check(audio, rate, max_tail_seconds):
    """显式用于预期持续的短样；不能判断有意休止或音乐质量。"""
    if type(max_tail_seconds) not in (int, float) or not math.isfinite(max_tail_seconds) or max_tail_seconds < 0 or max_tail_seconds >= len(audio) / rate:
        raise AudioError("coverage_contract", "允许尾音须为小于总时长的非负有限秒数")
    scale = float(np.max(np.abs(audio)))
    block = max(1, round(rate * 0.1))
    if scale == 0:
        return {"accepted": False, "trailing_low_ac_seconds": len(audio) / rate, "reason": "无交流活动"}
    normalized = audio / scale
    powers, ends = [], []
    for begin in range(0, len(audio), block):
        chunk = normalized[begin:begin + block]
        chunk = chunk - np.mean(chunk, axis=0, keepdims=True)
        powers.append(float(np.mean(chunk ** 2)))
        ends.append(min(begin + block, len(audio)))
    reference = float(np.percentile(powers, 75))
    threshold = max(reference * 1e-4, 1e-24)
    active = [i for i, power in enumerate(powers) if power >= threshold] if reference > 1e-24 else []
    last = ends[active[-1]] if active else 0
    trailing = (len(audio) - last) / rate
    return {"accepted": bool(active and trailing <= max_tail_seconds + block / rate),
            "trailing_low_ac_seconds": trailing, "allowed_tail_seconds": max_tail_seconds,
            "window_seconds": block / rate, "relative_threshold_db": -40,
            "notes": "持续短样末端低活动门禁；有意静默和极低动态需人工判断，不评价风格。"}


def check_audio(path, expected=None):
    """只读检查最终文件；expected 可给输出规格、帧数、活动区间及循环配置。"""
    expected = expected or {}
    path = _local(path)
    meta = _probe(path)
    rate, channels = meta["sample_rate_hz"], meta["channels"]
    data = _decode(path)
    if len(data) % channels:
        raise AudioError("invalid_frames", "解码样本数不能整除声道数")
    audio = data.reshape(-1, channels)
    if len(audio) > LIMITS["max_seconds"] * rate:
        raise AudioError("duration_limit", "实际解码时长超过分析上限")
    finite = bool(np.isfinite(audio).all())
    checks = []
    def add(name, passed, detail):
        checks.append({"name": name, "status": "通过" if passed else "失败", "detail": detail})
    add("完整解码与有限数", bool(len(audio)) and finite, {"frames": len(audio)})
    add("WAV 容器", meta["container"] == "wav", meta["container"])
    formats = {"pcm_s16": "pcm_s16le", "pcm_s24": "pcm_s24le", "pcm_s32": "pcm_s32le", "float32": "pcm_f32le"}
    desired = expected.get("sample_format", "pcm_s24")
    add("PCM 编码", meta["codec"] == formats.get(desired, desired), {"actual": meta["codec"], "expected": desired, "bits": meta["bits_per_sample"]})
    for key, default in (("sample_rate_hz", 48000), ("channels", 2)):
        add(key, meta[key] == expected.get(key, default), {"actual": meta[key], "expected": expected.get(key, default)})
    if expected.get("target_file_frames") is not None:
        add("精确总帧数（包含尾音）", len(audio) == expected["target_file_frames"], {"actual": len(audio), "expected": expected["target_file_frames"]})
    measurements = meta | {"decoded_frames": len(audio), "duration_seconds": len(audio) / rate,
                           "file_sha256": _hash_file(path), "decoded_float64_sha256": hashlib.sha256(data.tobytes()).hexdigest(),
                           "decoded_hash_method": "FFmpeg 解码为 little-endian float64；相同版本下比较有效样本",
                           "quantization_input_finite": "未知：当前检查仅覆盖最终文件，不覆盖引擎量化前浮点流",
                           "activity": {"status": "未评估", "reason": "文件为空或存在非有限数"}}
    if finite and len(audio):
        measurements["activity"] = _activity_report(audio, rate, expected)
        if "continuous_tail_max_seconds" in expected:
            if type(expected.get("target_file_frames")) is not int:
                raise AudioError("coverage_contract", "持续短样门禁必须显式提供目标帧数")
            coverage = _continuous_tail_check(audio, rate, expected["continuous_tail_max_seconds"])
            measurements["continuous_tail_coverage"] = coverage
            add("持续短样末端覆盖", coverage["accepted"], coverage)
        windows = _windows(audio, rate)
        active_frames = sum(w["end_frame"] - w["start_frame"] for w in windows if w["rms_dbfs"] is not None and w["rms_dbfs"] > -80)
        min_active_frames = min(rate, math.ceil(len(audio) * 0.05))
        rms = np.sqrt(np.mean(audio ** 2, axis=0))
        peaks = np.max(np.abs(audio), axis=0)
        bits = meta["bits_per_sample"]
        positive_full_scale = 1 - 2.0 ** (1 - bits) if meta["codec"].startswith("pcm_s") and bits else 1.0
        full_scale = (audio >= positive_full_scale) | (audio <= -1.0)
        add("持续有声", active_frames >= min_active_frames, {"active_frames": active_frames, "minimum": min_active_frames, "threshold_dbfs": -80, "window_seconds": 0.25, "notes": "仅最低整体发声量初筛，不证明全时段覆盖；需结合activity逐秒去DC诊断、乐谱活动区间与人工判断"})
        add("各声道有声", bool(np.all(rms > 1e-4)), {"channel_rms_dbfs": [_db(float(v)) for v in rms], "threshold_dbfs": -80})
        measurements.update({"sample_peak_dbfs": [_db(float(v)) for v in peaks],
                             "full_scale_samples": [int(np.sum(full_scale[:, c])) for c in range(channels)],
                             "energy_windows": windows, "stereo_design": "未由声道字段证明，需人工及声部对照",
                             "loudness": _loudness(path)})
        add("最终文件无满幅样本", not bool(np.any(full_scale)), {"positive_full_scale": positive_full_scale, "negative_full_scale": -1.0, "notes": "按整数PCM不对称正负边界计数；此项不能证明上游从未削波"})
        for region in expected.get("active_regions", []):
            start, end = region["start_frame"], region["end_frame"]
            valid = 0 <= start < end <= len(audio)
            region_rms = float(np.sqrt(np.mean(audio[start:end] ** 2))) if valid else 0
            add("声明活动区间有声", valid and region_rms > 1e-4, region | {"rms_dbfs": _db(region_rms)})
        if expected.get("loop", {}).get("enabled"):
            width = min(max(1, rate // 100), len(audio))
            joined = np.concatenate([audio[-width:], audio[:width]], axis=0)
            measurements["loop_boundary"] = {"join_jump_peak": float(np.max(np.abs(audio[0] - audio[-1]))),
                "join_20ms_rms_dbfs": _db(float(np.sqrt(np.mean(joined ** 2)))), "end_fade_accepted_as_loop": False,
                "status": "仅测量；未校准接缝阈值，未试听，不能据此认定无缝",
                "three_cycle_frames": len(audio) * 3, "three_cycle_boundary_count": 2,
                "notes": "三个相同周期的两处边界数值相同；未导出三周期试听文件，音乐结构与效果状态需另验"}
    return {"status": "通过" if all(c["status"] == "通过" for c in checks) else "失败", "status_scope": "基础技术检查；活动覆盖和听感另行判断", "checks": checks,
            "activity_review": measurements["activity"]["status"],
            "measurements": measurements, "tools": _tools(), "human_review": "未试听，等待人工审核",
            "commands": [["ffprobe", "-v", "error", *INPUT_FLAGS, "-show_streams", "-show_format", "-of", "json", "<输入文件>"],
                         ["ffmpeg", "-v", "error", "-nostdin", *INPUT_FLAGS, "-i", "<输入文件>", "-map", "0:a:0", "-vn", "-t", "601", "-f", "f64le", "-acodec", "pcm_f64le", "pipe:1"],
                         ["ffmpeg", "-hide_banner", "-nostdin", *INPUT_FLAGS, "-i", "<输入文件>", "-map", "0:a:0", "-vn", "-af", "ebur128=peak=true", "-f", "null", "-"]]}


def _public_target(url):
    parsed = urlsplit(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password:
        raise AudioError("url_scheme", "仅支持无账号信息的公开 HTTP/HTTPS 直接媒体链接")
    try:
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        addresses = socket.getaddrinfo(parsed.hostname, port, type=socket.SOCK_STREAM)
    except (ValueError, socket.gaierror):
        raise AudioError("url_resolution", "链接地址或域名无法解析") from None
    ips = sorted({a[4][0] for a in addresses})
    if not ips or any(not ipaddress.ip_address(ip).is_global for ip in ips):
        raise AudioError("url_private", "公开链接下载器拒绝内网、回环、链路本地或保留地址")
    return parsed, port, ips


def _fetch(url, dest):
    """校验每次跳转并把连接固定到已验证 IP，避免二次 DNS 解析绕过地址检查。"""
    current = url
    deadline = time.monotonic() + LIMITS["download_total_seconds"]
    for _ in range(LIMITS["max_redirects"] + 1):
        if time.monotonic() >= deadline:
            raise AudioError("url_timeout", "媒体下载超过总耗时上限", LIMITS)
        parsed, port, ips = _public_target(current)
        conn = http.client.HTTPConnection(parsed.hostname, port, timeout=LIMITS["network_timeout_seconds"])
        try:
            raw = socket.create_connection((ips[0], port), timeout=LIMITS["network_timeout_seconds"])
            conn.sock = ssl.create_default_context().wrap_socket(raw, server_hostname=parsed.hostname) if parsed.scheme == "https" else raw
            request_path = parsed.path or "/"
            if parsed.query:
                request_path += "?" + parsed.query
            conn.request("GET", request_path, headers={"User-Agent": "code-band-bgm/" + VERSION, "Accept": "audio/*,video/*,application/octet-stream"})
            response = conn.getresponse()
            if response.status in (301, 302, 303, 307, 308):
                location = response.getheader("Location")
                if not location:
                    raise AudioError("url_redirect", "链接跳转缺少目标")
                current = urljoin(current, location)
                continue
            if response.status != 200:
                code = "url_restricted" if response.status in (401, 402, 403, 451) else "url_http"
                raise AudioError(code, "公开入口拒绝访问或不可用；可提供有权使用的本地音频", {"http_status": response.status})
            content_type = response.getheader("Content-Type", "").split(";", 1)[0].lower()
            if not (content_type.startswith(("audio/", "video/")) or content_type == "application/octet-stream"):
                raise AudioError("url_not_media", "入口不是受支持的直接媒体；网页、登录页和 DRM 页面不作为音频分析成功")
            length = response.getheader("Content-Length")
            if length and int(length) > LIMITS["max_bytes"]:
                raise AudioError("file_too_large", "下载媒体超过大小上限", LIMITS)
            size = 0
            with open(dest, "xb") as stream:
                while chunk := response.read(65536):
                    if time.monotonic() >= deadline:
                        raise AudioError("url_timeout", "媒体下载超过总耗时上限", LIMITS)
                    size += len(chunk)
                    if size > LIMITS["max_bytes"]:
                        raise AudioError("file_too_large", "实际下载媒体超过大小上限", LIMITS)
                    stream.write(chunk)
            # URL 只保留公开来源主机，路径和查询均可能含访问凭据。
            return {"kind": "公开直接媒体", "origin_host": parsed.hostname, "url_sha256": hashlib.sha256(url.encode()).hexdigest(),
                    "downloaded_bytes": size, "content_type": content_type, "credentials": "未读取 cookie 或账号凭据"}
        except AudioError:
            raise
        except (OSError, ValueError, http.client.HTTPException):
            raise AudioError("url_network", "公开媒体网络获取失败；未绕过访问控制") from None
        finally:
            conn.close()
    raise AudioError("url_redirect_limit", "链接跳转次数超过上限", LIMITS)



def _proxy_target(url):
    """代理解析公开域名；本地名字、地址字面量与绕过代理的目标仍拒绝。"""
    try:
        parsed = urlsplit(url)
        host = (parsed.hostname or "").rstrip(".").lower()
        port = parsed.port or 443
    except ValueError:
        raise AudioError("url_scheme", "链接地址格式无效") from None
    if parsed.scheme != "https" or not host or parsed.username or parsed.password or port != 443:
        raise AudioError("url_scheme", "系统代理模式仅支持无账号信息、默认443端口的HTTPS直接媒体链接")
    if any(ord(c) < 33 for c in url) or "%" in host:
        raise AudioError("url_scheme", "链接包含非法控制字符或地址作用域")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        # inet_aton额外识别127.1、十六进制等IPv4旧写法，避免伪装成域名。
        try:
            address = ipaddress.ip_address(socket.inet_ntoa(socket.inet_aton(host)))
        except (OSError, ValueError):
            address = None
    if address is not None:
        if not address.is_global:
            raise AudioError("url_private", "系统代理模式仍拒绝内网、回环和保留地址")
    elif "." not in host or host.startswith(".") or ".." in host or host.endswith((".local", ".localhost", ".lan", ".home", ".internal", ".onion", ".test", ".invalid")):
        raise AudioError("url_private", "系统代理模式拒绝本地名字或保留域名")
    if urlrequest.proxy_bypass(parsed.netloc):
        raise AudioError("proxy_bypass", "当前代理排除规则会使该目标直连；拒绝静默绕过系统代理")
    return parsed


def _fetch_system_proxy(url, dest):
    """显式使用现有代理；不改变配置，不自动降级，不附加账号或Cookie。"""
    configured = urlrequest.getproxies().get("https")
    if not configured:
        raise AudioError("proxy_missing", "系统未配置HTTPS代理；不能使用system_proxy模式")
    try:
        proxy = urlsplit(configured if "://" in configured else "http://" + configured)
        if proxy.scheme not in ("http", "https") or not proxy.hostname or proxy.username or proxy.password:
            raise ValueError("代理格式或凭据不受支持")
        proxy.port  # 检查端口格式，不把代理地址或额外凭据写入报告。
    except ValueError:
        raise AudioError("proxy_configuration", "当前HTTPS代理格式不受支持，或包含未授权的额外账号信息") from None
    first = _proxy_target(url)
    deadline = time.monotonic() + LIMITS["download_total_seconds"]
    class Guard(urlrequest.HTTPRedirectHandler):
        redirects = 0
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            self.redirects += 1
            if self.redirects > LIMITS["max_redirects"]:
                raise AudioError("url_redirect_limit", "链接跳转次数超过上限", LIMITS)
            if time.monotonic() >= deadline:
                raise AudioError("url_timeout", "媒体下载超过总耗时上限", LIMITS)
            _proxy_target(newurl)
            return super().redirect_request(req, fp, code, msg, headers, newurl)
    opener = urlrequest.build_opener(urlrequest.ProxyHandler({"https": configured}), Guard(),
                                    urlrequest.HTTPSHandler(context=ssl.create_default_context()))
    request = urlrequest.Request(url, headers={"User-Agent": "code-band-bgm/" + VERSION,
                                              "Accept": "audio/*,video/*,application/octet-stream"})
    try:
        with opener.open(request, timeout=LIMITS["network_timeout_seconds"]) as response:
            if response.status != 200:
                raise AudioError("url_http", "入口未返回完整媒体；不接受部分响应", {"http_status": response.status})
            final = _proxy_target(response.geturl())
            content_type = response.headers.get("Content-Type", "").split(";", 1)[0].lower()
            if not (content_type.startswith(("audio/", "video/")) or content_type == "application/octet-stream"):
                raise AudioError("url_not_media", "入口不是受支持的直接媒体；网页和登录页不作为音频分析成功")
            length = response.headers.get("Content-Length")
            if length and int(length) > LIMITS["max_bytes"]:
                raise AudioError("file_too_large", "下载媒体超过大小上限", LIMITS)
            size = 0
            with open(dest, "xb") as stream:
                while chunk := response.read(65536):
                    if time.monotonic() >= deadline:
                        raise AudioError("url_timeout", "媒体下载超过总耗时上限", LIMITS)
                    size += len(chunk)
                    if size > LIMITS["max_bytes"]:
                        raise AudioError("file_too_large", "实际下载媒体超过大小上限", LIMITS)
                    stream.write(chunk)
            if length and size != int(length):
                raise AudioError("url_incomplete", "媒体下载字节数与服务器声明不符；不接受部分结果")
            return {"kind": "公开直接媒体", "origin_host": first.hostname, "final_host": final.hostname,
                    "url_sha256": hashlib.sha256(url.encode()).hexdigest(), "downloaded_bytes": size,
                    "content_type": content_type, "network_mode": "system_proxy", "tls_certificate_validation": True,
                    "credentials": "未读取Cookie、浏览器凭据或额外账号信息",
                    "destination_ip_verification": "无法验证代理后真实目的IP；仅供操作者显式信任的现有代理解析公开域名，不等同默认直连的公网IP校验"}
    except AudioError:
        raise
    except urlerror.HTTPError as exc:
        code = "url_restricted" if exc.code in (401, 402, 403, 451) else "url_http"
        raise AudioError(code, "公开媒体入口拒绝访问或不可用；未绕过访问控制", {"http_status": exc.code}) from None
    except (OSError, ValueError, http.client.HTTPException, urlerror.URLError):
        raise AudioError("url_network", "系统代理媒体获取失败；没有自动切换网络或绕过访问控制") from None


def _estimates(audio, rate):
    audio = audio[:rate * LIMITS["estimate_segment_seconds"]]
    hop = 220
    _, _, spec = signal.stft(audio, fs=rate, nperseg=2048, noverlap=2048 - hop, boundary=None)
    power = np.abs(spec)
    onset = np.maximum(np.diff(power, axis=1), 0).sum(axis=0)
    onset -= onset.mean() if len(onset) else 0
    candidates = []
    if len(onset) > 4 and np.max(np.abs(onset)) > 1e-9:
        ac = signal.fftconvolve(onset, onset[::-1], mode="full")[len(onset) - 1:]
        low, high = max(1, round(rate * 60 / (hop * 240))), min(len(ac) - 1, round(rate * 60 / (hop * 40)))
        peaks, _ = signal.find_peaks(ac[low:high + 1])
        best = sorted((int(p + low) for p in peaks), key=lambda lag: ac[lag], reverse=True)[:5]
        for lag in best:
            if ac[lag] > 0:
                bpm = round(60 * rate / (hop * lag), 2)
                # G4：单一后端不输出单值，半速/倍速一起给出，由第二后端交叉或人工确认
                candidates.append({"bpm": bpm, "octave_alternatives": [round(bpm / 2, 2), bpm, round(bpm * 2, 2)],
                                   "single_value_allowed": False, "autocorrelation_ratio": round(float(ac[lag] / max(ac[0], 1e-12)), 4),
                                   "confidence": "低", "reason": "谱通量自相关；可能为半速/倍速，未校准统计概率"})
    # 聚合频谱为十二音级能量；不导出逐音符旋律。
    freqs = np.fft.rfftfreq(2048, 1 / rate)
    mask = (freqs >= 65.4) & (freqs <= 2093)
    midi = np.rint(69 + 12 * np.log2(freqs[mask] / 440)).astype(int)
    chroma = np.zeros(12)
    for pitch, energy in zip(midi, np.mean(power[mask] ** 2, axis=1)):
        chroma[pitch % 12] += energy
    keys = []
    if chroma.sum() > 1e-16:
        chroma /= chroma.sum()
        names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        # G5：音阶覆盖率只能区分音集，关系大小调得分恒等；不输出主音和大小调，留给人工确认（L4）
        scores = sorted(((float(sum(chroma[(root + i) % 12] for i in (0, 2, 4, 5, 7, 9, 11))), root) for root in range(12)), reverse=True)
        for score, root in scores[:3]:
            keys.append({"pitch_set": [names[(root + i) % 12] for i in (0, 2, 4, 5, 7, 9, 11)],
                         "relative_pair": [names[root] + " 大调", names[(root + 9) % 12] + " 自然小调"], "tonic": None,
                         "scale_energy_ratio": round(score, 4), "confidence": "低",
                         "reason": "十二音级频谱与七声音集覆盖率；只能给出音集，主音与大小调未知，和声模糊、谐波与打击乐可能干扰"})
    return {"tempo_candidates": candidates or [{"bpm": None, "confidence": "未知", "reason": "没有稳定起音周期"}],
            "key_candidates": keys or [{"tonic": None, "confidence": "未知", "reason": "没有可靠音级能量"}],
            "beat_positions": {"confidence": "未知", "reason": "自相关不确定节拍相位与拍号，未推断逐拍位置"},
            "groove": {"confidence": "未知", "reason": "未运行摇摆或节奏分组分类器"},
            "pitch_class_energy": [round(float(v), 6) for v in chroma],
            "method": {"analysis_rate_hz": rate, "fft_size": 2048, "hop_frames": hop, "tempo_range_bpm": [40, 240], "melody_transcription": False, "segment_start_seconds": 0, "segment_end_seconds": len(audio) / rate}}


def analyze(source, output_path=None, network_mode="direct", rhythm_backend="basic", rhythm_model=None, rhythm_model_sha256=None):
    """参考只作测量与抽象特征；下载副本在本地临时目录内，任务后移除。"""
    if network_mode not in ("direct", "system_proxy"):
        raise AudioError("network_mode", "网络模式只支持direct或显式system_proxy")
    if rhythm_backend not in ("basic", "librosa", "beat_this"):
        raise AudioError("rhythm_backend", "节拍后端只支持基础候选、librosa或显式本地BeatThis")
    if rhythm_backend == "beat_this":
        if not rhythm_model or not rhythm_model_sha256:
            raise AudioError("rhythm_model", "BeatThis须显式提供本地模型和SHA256，不会自动下载")
    elif rhythm_model is not None or rhythm_model_sha256 is not None:
        raise AudioError("rhythm_model", "模型参数仅用于beat_this后端，不能静默忽略")
    source = str(source)
    with tempfile.TemporaryDirectory(prefix="code-band-reference-") as work:
        if urlsplit(source).scheme and not Path(source).exists():
            path = Path(work) / "reference.media"
            origin = _fetch_system_proxy(source, path) if network_mode == "system_proxy" else _fetch(source, path)
            origin.setdefault("network_mode", "direct")
        else:
            path = _local(source)
            origin = {"kind": "本地文件", "file_name": path.name}
        meta = _probe(path)
        from reference_gate import decide, measure
        channels = min(meta["channels"], 2)
        frames = _decode(path, rate=22050, channels=channels).reshape(-1, channels)
        if not np.isfinite(frames).all() or not len(frames):
            raise AudioError("nonfinite_audio", "参考音频为空或包含非有限样本")
        # 显式中声道：ffmpeg -ac 1 降混对相关立体声会让峰值偏高约 3 dB（负控 H1）
        mono = frames.mean(axis=1)
        level = measure(frames)
        gate = decide(level)
        if len(mono) > LIMITS["max_seconds"] * 22050:
            raise AudioError("duration_limit", "实际解码时长超过分析上限", LIMITS)
        energy = _windows(mono, 22050, 5.0)
        report = {"status": "通过", "analysis_scope": "objective_measurements_only", "measurement_status": "通过",
                  "reference_understanding": {"status": "待语义证据", "notes": "已完成真实音频测量；尚无音频感知或用户音乐描述证据，不表示音频无法分析，也不表示已理解参考风格。"},
                  "source": origin | {"file_sha256": _hash_file(path), "usage": "仅本地参考分析，不作为成品采样", "license": "由调用者登记真实来源与使用依据；未自动判定授权"},
                  "measurements": meta | {"analyzed_seconds": len(mono) / 22050, "analysis_sample_peak_dbfs": level["peak_dbfs"],
                                          "analysis_peak_basis": "解码后各声道样本峰值（22050 Hz），不是降混后的峰值",
                                          "stereo_rms_dbfs": level["rms_dbfs"], "lr_correlation": level["lr_correlation"],
                                          "mono_folddown_loss_db": level["mono_folddown_loss_db"], "energy_sections": energy,
                                          "energy_sections_timebase": {"sample_rate_hz": 22050, "origin_seconds": 0,
                                                                       "frame_interval": "start_inclusive_end_exclusive", "window_seconds": 5.0}},
                  "signal_gate": gate,
                  "estimates": _estimates(mono, 22050) if gate["mono_models_allowed"] else
                               {"status": "拒绝", "reasons": gate["reasons"], "tempo_candidates": [], "key_candidates": []},
                  "interpretation": {"instrumentation": {"confidence": "未知", "reason": "没有实际听辨或乐器分类；频谱不能证明具体乐器"},
                                     "structure": {"confidence": "低", "reason": "仅给五秒窗能量变化，不能等同曲式段落"}, "human_review": "未试听，等待人工审核"},
                  "original_composition_policy": {"reuse_reference_audio": False, "reuse_transcribed_melody": False, "notes": "宿主仅提取抽象速度、密度和能量走向；另写主题与编排，不保证绝对原创或权利安全"},
                  "tools": _tools(), "limits": LIMITS,
                  "commands": [["ffprobe", "-v", "error", *INPUT_FLAGS, "-show_streams", "-show_format", "-of", "json", "<参考文件>"],
                               ["ffmpeg", "-v", "error", "-nostdin", *INPUT_FLAGS, "-i", "<参考文件>", "-map", "0:a:0", "-vn", "-t", "601", "-ar", "22050", "-ac", "<1或2>", "-f", "f64le", "-acodec", "pcm_f64le", "pipe:1"]]}
        if not gate["mono_models_allowed"]:
            report["reference_understanding"] = {"status": "拒绝", "notes": "未通过信号或相位门禁，没有运行节拍、调性或模型分析。", "reasons": gate["reasons"]}
            if rhythm_backend != "basic":
                report["rhythm_analysis"] = {"status": "拒绝", "backend": rhythm_backend, "reasons": gate["reasons"]}
        elif rhythm_backend == "librosa":
            from rhythm_analysis import analyze_rhythm
            report["rhythm_analysis"] = analyze_rhythm(mono, 22050)
        elif rhythm_backend == "beat_this":
            from beat_this_analysis import analyze_beats
            report["rhythm_analysis"] = analyze_beats(mono, 22050, rhythm_model, rhythm_model_sha256)
    if output_path:
        destination = Path(output_path)
        with destination.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    return report
