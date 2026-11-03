"""临时合成信号仅用于工具回归；不计为乐团作品或引擎接入证据。"""
import importlib.util
import json
import math
import socket
import subprocess
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.io import wavfile

MODULE = Path(__file__).resolve().parents[1] / "skills/code-band-bgm" / "scripts" / "audio_tools.py"
spec = importlib.util.spec_from_file_location("audio_tools", MODULE)
audio_tools = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audio_tools)


def pcm24(path, audio, rate=48000):
    data = np.clip(np.asarray(audio) * 8388607, -8388608, 8388607).astype("<i4")
    packed = data.reshape(-1).view(np.uint8).reshape(-1, 4)[:, :3].tobytes()
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(audio.shape[1])
        stream.setsampwidth(3)
        stream.setframerate(rate)
        stream.writeframes(packed)


class AudioToolsTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="code-band-audio-test-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.path = self.root / '信号 $(touch 不应执行).wav'
        time = np.arange(48000) / 48000
        self.signal = np.column_stack([0.2 * np.sin(2 * np.pi * 440 * time), 0.15 * np.sin(2 * np.pi * 660 * time)])
        pcm24(self.path, self.signal)

    def test_exact_pcm24_final_file_and_distinct_measurements(self):
        report = audio_tools.check_audio(self.path, {"target_file_frames": 48000})
        self.assertEqual(report["status"], "通过")
        self.assertEqual(report["measurements"]["decoded_frames"], 48000)
        self.assertEqual(report["measurements"]["bits_per_sample"], 24)
        self.assertAlmostEqual(report["measurements"]["sample_peak_dbfs"][0], 20 * math.log10(.2), places=3)
        loudness = report["measurements"]["loudness"]
        self.assertIsNotNone(loudness["integrated_lufs"])
        self.assertIsNotNone(loudness["true_peak_dbtp"])
        self.assertIn("未试听", report["human_review"])
        self.assertIn("未知", report["measurements"]["quantization_input_finite"])
        json.dumps(report, allow_nan=False)
        self.assertFalse((Path.cwd() / "不应执行").exists())

    def test_wrong_frames_and_wrong_format_do_not_pass(self):
        report = audio_tools.check_audio(self.path, {"target_file_frames": 47999, "sample_format": "pcm_s16"})
        self.assertEqual(report["status"], "失败")
        self.assertEqual(sum(c["status"] == "失败" for c in report["checks"]), 2)

    def test_positive_integer_full_scale_is_not_mistaken_for_headroom(self):
        pcm24(self.path, np.ones((48000, 2)))
        report = audio_tools.check_audio(self.path)
        self.assertEqual(report["status"], "失败")
        self.assertEqual(report["measurements"]["full_scale_samples"], [48000, 48000])
        self.assertTrue(any(c["name"] == "最终文件无满幅样本" and c["status"] == "失败" for c in report["checks"]))

    def test_silence_and_sparse_transient_are_reported(self):
        for name, data in (("静音", np.zeros((48000, 2))), ("只有瞬态", np.zeros((48000 * 8, 2)))):
            if name == "只有瞬态":
                data[100:200] = .3
            path = self.root / (name + ".wav")
            pcm24(path, data)
            report = audio_tools.check_audio(path)
            self.assertEqual(report["status"], "失败")
            self.assertTrue(any(c["name"] == "持续有声" and c["status"] == "失败" for c in report["checks"]))
            json.dumps(report, allow_nan=False)

    def test_nonfinite_float_source_is_rejected_in_check_and_analysis(self):
        path = self.root / "非有限.wav"
        data = self.signal.astype(np.float32)
        data[100, 0] = np.nan
        wavfile.write(path, 48000, data)
        report = audio_tools.check_audio(path, {"sample_format": "float32"})
        self.assertEqual(report["status"], "失败")
        self.assertEqual(report["checks"][0]["status"], "失败")
        with self.assertRaises(audio_tools.AudioError) as caught:
            audio_tools.analyze(path)
        self.assertEqual(caught.exception.code, "nonfinite_audio")

    def test_invalid_file_no_fake_result(self):
        invalid = self.root / "损坏.wav"
        invalid.write_text("并非音频", encoding="utf-8")
        with self.assertRaises(audio_tools.AudioError) as caught:
            audio_tools.check_audio(invalid)
        self.assertEqual(caught.exception.code, "decode_failed")
        self.assertNotIn(str(self.root), str(caught.exception.details))

    def test_playlist_cannot_cause_indirect_network_or_file_reads(self):
        path = self.root / "间接媒体.m3u8"
        path.write_text("#EXTM3U\n#EXT-X-TARGETDURATION:1\n#EXTINF:1,\nhttp://127.0.0.1:9/private.wav\n#EXT-X-ENDLIST\n", encoding="utf-8")
        with self.assertRaises(audio_tools.AudioError) as caught:
            audio_tools.analyze(path)
        self.assertEqual(caught.exception.code, "decode_failed")

    def test_activity_interval_and_loop_are_separate(self):
        report = audio_tools.check_audio(self.path, {"active_regions": [{"start_frame": 0, "end_frame": 1000}], "loop": {"enabled": True}})
        self.assertEqual(report["status"], "通过")
        boundary = report["measurements"]["loop_boundary"]
        self.assertIn("不能据此认定无缝", boundary["status"])
        self.assertEqual(boundary["three_cycle_frames"], 144000)

    def test_reference_tempo_uses_actual_waveform_and_no_transcription(self):
        rate = 22050
        data = np.zeros(rate * 8)
        length = rate // 10
        pulse = .5 * np.sin(2 * np.pi * 440 * np.arange(length) / rate) * np.exp(-np.arange(length) / (rate * .02))
        for frame in range(0, len(data) - length, rate // 2):
            data[frame:frame + length] += pulse
        path = self.root / "节拍夹具.wav"
        pcm24(path, np.column_stack([data, data]), rate)
        report = audio_tools.analyze(path)
        self.assertEqual(report["status"], "通过")
        self.assertEqual(report["measurement_status"], "通过")
        self.assertEqual(report["analysis_scope"], "objective_measurements_only")
        self.assertEqual(report["reference_understanding"]["status"], "待语义证据")
        tempos = report["estimates"]["tempo_candidates"]
        self.assertTrue(any(abs(c["bpm"] - 120) < 2 for c in tempos if c["bpm"]))
        self.assertEqual(report["interpretation"]["instrumentation"]["confidence"], "未知")
        self.assertFalse(report["estimates"]["method"]["melody_transcription"])
        self.assertFalse(report["original_composition_policy"]["reuse_reference_audio"])
        self.assertEqual(report["source"]["file_sha256"], audio_tools._hash_file(path))

    def test_reference_energy_sections_have_explicit_decoded_timebase(self):
        for rate in (44100, 48000):
            with self.subTest(rate=rate):
                time = np.arange(rate * 12) / rate
                data = .1 * np.sin(2 * np.pi * 440 * time)
                path = self.root / (str(rate) + "_时基.wav")
                pcm24(path, np.column_stack([data, data]), rate)
                measurements = audio_tools.analyze(path)["measurements"]
                self.assertEqual(measurements["sample_rate_hz"], rate)
                self.assertEqual(measurements["energy_sections_timebase"], {
                    "sample_rate_hz": 22050, "origin_seconds": 0,
                    "frame_interval": "start_inclusive_end_exclusive", "window_seconds": 5.0})
                windows = measurements["energy_sections"]
                self.assertEqual([(w["start_seconds"], w["end_seconds"]) for w in windows],
                                 [(0, 5), (5, 10), (10, 12)])
                self.assertEqual([(w["start_frame"], w["end_frame"]) for w in windows],
                                 [(0, 110250), (110250, 220500), (220500, 264600)])

    def test_analysis_cli_energy_sections_preserve_five_second_boundaries(self):
        for rate in (44100, 48000):
            with self.subTest(rate=rate):
                time = np.arange(rate * 12) / rate
                data = .1 * np.sin(2 * np.pi * 440 * time)
                path = self.root / (str(rate) + "_CLI时基.wav")
                pcm24(path, np.column_stack([data, data]), rate)
                output = self.root / (str(rate) + "_分析.json")
                process = subprocess.run([sys.executable, str(MODULE.with_name("band.py")), "analyze",
                                          "--input", str(path), "--out", str(output)],
                                         capture_output=True, text=True, timeout=30)
                self.assertEqual(process.returncode, 0, process.stderr)
                measurements = json.loads(output.read_text(encoding="utf-8"))["measurements"]
                self.assertEqual(measurements["sample_rate_hz"], rate)
                self.assertEqual(measurements["energy_sections_timebase"]["sample_rate_hz"], 22050)
                self.assertEqual([(w["start_seconds"], w["end_seconds"]) for w in measurements["energy_sections"]],
                                 [(0, 5), (5, 10), (10, 12)])

    def test_final_file_energy_windows_keep_native_rate_and_quarter_second_measurements(self):
        report = audio_tools.check_audio(self.path)
        windows = report["measurements"]["energy_windows"]
        self.assertEqual([(w["start_frame"], w["end_frame"]) for w in windows],
                         [(0, 12000), (12000, 24000), (24000, 36000), (36000, 48000)])
        decoded = audio_tools._decode(self.path).reshape(-1, 2)
        for window in windows:
            self.assertEqual(window["start_seconds"], window["start_frame"] / 48000)
            self.assertEqual(window["end_seconds"], window["end_frame"] / 48000)
            expected = audio_tools._db(float(np.sqrt(np.mean(decoded[window["start_frame"]:window["end_frame"]] ** 2))))
            self.assertEqual(window["rms_dbfs"], expected)

    def test_analysis_never_overwrites_report(self):
        report_path = self.root / "分析.json"
        report_path.write_text("原有内容", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            audio_tools.analyze(self.path, report_path)
        self.assertEqual(report_path.read_text(encoding="utf-8"), "原有内容")

    def test_restricted_local_schemes_and_private_ips(self):
        for url, code in (("file:///etc/passwd", "url_scheme"), ("http://127.0.0.1/audio.wav", "url_private"), ("http://user:secret@example.com/audio.wav", "url_scheme")):
            with self.assertRaises(audio_tools.AudioError) as caught:
                audio_tools.analyze(url)
            self.assertEqual(caught.exception.code, code)
            self.assertNotIn("secret", str(caught.exception))

    def test_safe_downloader_success_limits_and_rejections(self):
        payload = self.path.read_bytes()
        class Response:
            status = 200
            headers = {"Content-Type": "audio/wav"}
            def __init__(self):
                self.data = payload
            def getheader(self, name, default=None):
                return self.headers.get(name, default)
            def read(self, size):
                data, self.data = self.data[:size], self.data[size:]
                return data
        response = Response()
        class Connection:
            def __init__(self, *args, **kwargs):
                self.sock = None
            def request(self, *args, **kwargs):
                pass
            def getresponse(self):
                return response
            def close(self):
                pass
        def resolve(host, port, **kwargs):
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1" if host == "private.test" else "93.184.216.34", port))]
        with patch.object(audio_tools.socket, "getaddrinfo", side_effect=resolve), patch.object(audio_tools.socket, "create_connection", return_value=object()), patch.object(audio_tools.http.client, "HTTPConnection", Connection):
            report = audio_tools.analyze("http://public.test/reference.wav?token=不应入日志")
            self.assertEqual(report["source"]["origin_host"], "public.test")
            self.assertNotIn("不应入日志", json.dumps(report, ensure_ascii=False))
            self.assertEqual(report["source"]["downloaded_bytes"], len(payload))
            response.status = 403
            with self.assertRaises(audio_tools.AudioError) as caught:
                audio_tools.analyze("http://public.test/private.wav?token=不应入日志")
            self.assertEqual(caught.exception.code, "url_restricted")
            response.status = 200
            response.headers = {"Content-Type": "text/html"}
            with self.assertRaises(audio_tools.AudioError) as caught:
                audio_tools.analyze("http://public.test/page")
            self.assertEqual(caught.exception.code, "url_not_media")
            response.headers = {"Content-Type": "audio/wav", "Content-Length": str(audio_tools.LIMITS["max_bytes"] + 1)}
            with self.assertRaises(audio_tools.AudioError) as caught:
                audio_tools.analyze("http://public.test/large.wav")
            self.assertEqual(caught.exception.code, "file_too_large")
            response.status = 302
            response.headers = {"Location": "http://private.test/audio.wav"}
            with self.assertRaises(audio_tools.AudioError) as caught:
                audio_tools.analyze("http://public.test/redirect")
            self.assertEqual(caught.exception.code, "url_private")


    def test_system_proxy_requires_explicit_mode_and_existing_configuration(self):
        url = "https://example.com/audio.wav"
        with patch.object(audio_tools.urlrequest, "getproxies", return_value={}):
            with self.assertRaises(audio_tools.AudioError) as caught:
                audio_tools.analyze(url, network_mode="system_proxy")
            self.assertEqual(caught.exception.code, "proxy_missing")
        with patch.object(audio_tools, "_fetch", side_effect=audio_tools.AudioError("url_private", "拒绝")), patch.object(audio_tools, "_fetch_system_proxy") as proxy:
            with self.assertRaises(audio_tools.AudioError):
                audio_tools.analyze(url)
            proxy.assert_not_called()
        with self.assertRaises(audio_tools.AudioError) as caught:
            audio_tools.analyze(self.path, network_mode="自动回退")
        self.assertEqual(caught.exception.code, "network_mode")

    def test_system_proxy_rejects_private_names_credentials_and_bypass(self):
        configured = {"https": "http://127.0.0.1:7897"}
        with patch.object(audio_tools.urlrequest, "getproxies", return_value=configured), patch.object(audio_tools.urlrequest, "proxy_bypass", return_value=False):
            for url in ("https://127.0.0.1/a.wav", "https://127.1/a.wav", "https://localhost/a.wav", "https://device.local/a.wav", "https://[::1]/a.wav", "https://user:secret@example.com/a.wav", "http://example.com/a.wav"):
                with self.assertRaises(audio_tools.AudioError):
                    audio_tools.analyze(url, network_mode="system_proxy")
            with patch.object(audio_tools.urlrequest, "proxy_bypass", return_value=True):
                with self.assertRaises(audio_tools.AudioError) as caught:
                    audio_tools.analyze("https://example.com/a.wav", network_mode="system_proxy")
                self.assertEqual(caught.exception.code, "proxy_bypass")
        with patch.object(audio_tools.urlrequest, "getproxies", return_value={"https": "http://user:secret@127.0.0.1:7897"}):
            with self.assertRaises(audio_tools.AudioError) as caught:
                audio_tools.analyze("https://example.com/a.wav", network_mode="system_proxy")
            self.assertEqual(caught.exception.code, "proxy_configuration")
            self.assertNotIn("secret", str(caught.exception))

    def test_system_proxy_transfer_tls_redirect_guard_and_restricted_http(self):
        payload = self.path.read_bytes()
        handlers = []
        class Response:
            status = 200
            headers = {"Content-Type": "audio/wav"}
            def __init__(self):self.data = payload
            def __enter__(self):return self
            def __exit__(self, *args):pass
            def geturl(self):return "https://example.com/audio.wav"
            def read(self, size):
                chunk, self.data = self.data[:size], self.data[size:]
                return chunk
        class Opener:
            def open(self, request, timeout):return Response()
        def build(*items):
            handlers.extend(items)
            return Opener()
        with patch.object(audio_tools.urlrequest, "getproxies", return_value={"https": "http://127.0.0.1:7897"}), patch.object(audio_tools.urlrequest, "proxy_bypass", return_value=False), patch.object(audio_tools.urlrequest, "build_opener", side_effect=build):
            report = audio_tools.analyze("https://example.com/audio.wav?token=不能泄露", network_mode="system_proxy")
            self.assertEqual(report["source"]["network_mode"], "system_proxy")
            self.assertTrue(report["source"]["tls_certificate_validation"])
            self.assertIn("无法验证", report["source"]["destination_ip_verification"])
            self.assertNotIn("不能泄露", json.dumps(report, ensure_ascii=False))
            tls = next(h for h in handlers if isinstance(h, audio_tools.urlrequest.HTTPSHandler))
            self.assertTrue(tls._context.check_hostname)
            self.assertEqual(tls._context.verify_mode, audio_tools.ssl.CERT_REQUIRED)
            guard = next(h for h in handlers if isinstance(h, audio_tools.urlrequest.HTTPRedirectHandler))
            request = audio_tools.urlrequest.Request("https://example.com/a.wav")
            for target in ("https://127.0.0.1/private.wav", "http://example.com/a.wav", "https://user:secret@example.com/a.wav"):
                with self.assertRaises(audio_tools.AudioError):
                    guard.redirect_request(request, None, 302, "", {}, target)
            guard.redirects = audio_tools.LIMITS["max_redirects"]
            with self.assertRaises(audio_tools.AudioError) as caught:
                guard.redirect_request(request, None, 302, "", {}, "https://example.org/a.wav")
            self.assertEqual(caught.exception.code, "url_redirect_limit")
            with patch.object(Response, "status", 206):
                with self.assertRaises(audio_tools.AudioError) as caught:
                    audio_tools.analyze("https://example.com/a.wav", network_mode="system_proxy")
                self.assertEqual(caught.exception.code, "url_http")
            with patch.object(Response, "headers", {"Content-Type":"audio/wav", "Content-Length":str(len(payload)+1)}):
                with self.assertRaises(audio_tools.AudioError) as caught:
                    audio_tools.analyze("https://example.com/a.wav", network_mode="system_proxy")
                self.assertEqual(caught.exception.code, "url_incomplete")
            error = audio_tools.urlerror.HTTPError("https://example.com/?token=不能泄露", 403, "拒绝", {}, None)
            with patch.object(Opener, "open", side_effect=error):
                with self.assertRaises(audio_tools.AudioError) as caught:
                    audio_tools.analyze("https://example.com/a.wav", network_mode="system_proxy")
                self.assertEqual(caught.exception.code, "url_restricted")
                self.assertEqual(caught.exception.details["http_status"], 403)
                self.assertNotIn("不能泄露", str(caught.exception.details))


if __name__ == "__main__":
    unittest.main(verbosity=2)
