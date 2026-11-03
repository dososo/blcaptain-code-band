"""活动诊断夹具不是新乐团作品，不使用声学活动判音乐质量。"""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.io import wavfile

MODULE=Path(__file__).resolve().parents[1]/'skills/code-band-bgm/scripts/audio_tools.py'
spec=importlib.util.spec_from_file_location('audio_activity_tools',MODULE)
audio_tools=importlib.util.module_from_spec(spec)
spec.loader.exec_module(audio_tools)


class AudioActivityTests(unittest.TestCase):
    def test_explicit_continuous_tail_guard(self):
        sr = 8000
        t = np.arange(8 * sr) / sr
        body = np.sin(2 * np.pi * 220 * t)[:, None] * .2
        valid = np.concatenate([body, np.zeros((2 * sr, 1))])
        rejected = np.concatenate([body, np.zeros((7 * sr, 1))])
        self.assertTrue(audio_tools._continuous_tail_check(valid, sr, 2)['accepted'])
        self.assertFalse(audio_tools._continuous_tail_check(rejected, sr, 2)['accepted'])
        self.assertFalse(audio_tools._continuous_tail_check(rejected * .1 + .03, sr, 2)['accepted'])
        self.assertFalse(audio_tools._continuous_tail_check(np.ones((sr, 1)) * .1, sr, .2)['accepted'])
        with self.assertRaises(audio_tools.AudioError):
            audio_tools._continuous_tail_check(valid, sr, float('nan'))
    def test_single_frame_window_cannot_be_classified_as_silence(self):
        data = np.zeros((9, 1))
        data[-1, 0] = 1.0
        report = audio_tools._activity_report(data, 8)
        self.assertIsNone(report['windows'][-1]['near_silent'])
        self.assertEqual(report['unassessed_frames'], 1)
        self.assertEqual(report['status'], '需核')
        self.assertEqual(report['windows'][-1]['raw_rms_dbfs'], [0.0])
    def test_nine_seconds_audio_then_seven_seconds_dc_is_reported(self):
        rate=8000;t=np.arange(rate*16)/rate
        data=np.column_stack([.1*np.sin(2*np.pi*440*t),.1*np.sin(2*np.pi*660*t)])
        data[9*rate:]=.002
        with tempfile.TemporaryDirectory()as temp:
            path=Path(temp)/'有声后低电平直流.wav'
            wavfile.write(path,rate,data.astype(np.float32))
            report=audio_tools.check_audio(path,{'sample_format':'float32','sample_rate_hz':rate,'target_file_frames':rate*16})
        self.assertIn('activity',report['measurements'])
        activity=report['measurements']['activity']
        self.assertEqual(report['status'],'通过')
        self.assertEqual(report['activity_review'],'需核')
        self.assertEqual(activity['window_seconds'],1)
        self.assertEqual(activity['threshold_dbfs'],-70)
        self.assertEqual([(x['start_frame'],x['end_frame'])for x in activity['near_silent_intervals']],[(rate*9,rate*16)])
        self.assertEqual(activity['tail_boundary']['status'],'未知')
        self.assertEqual(activity['near_silent_intervals'][0]['region'],'尾音边界未知')
        self.assertLess(activity['windows'][-1]['max_channel_ac_rms'],1e-10)
        self.assertGreater(activity['windows'][-1]['raw_rms_dbfs'][0],-60)
        json.dumps(report,allow_nan=False)

    def test_constant_and_silence_are_finite_diagnostics(self):
        for value in [0.,.02]:
            with self.subTest(value=value):
                d=audio_tools._activity_report(np.full((24,2),value),8)
                self.assertEqual(d['near_silent_frames'],24)
                self.assertEqual(d['windows'][0]['ac_rms_dbfs'],[None,None])
                json.dumps(d,allow_nan=False)

    def test_antiphase_and_one_active_channel_do_not_cancel(self):
        t=np.arange(8000)/8000;s=.1*np.sin(2*np.pi*440*t)
        for data in [np.column_stack([s,-s]),np.column_stack([s,np.zeros_like(s)])]:
            d=audio_tools._activity_report(data,8000)
            self.assertEqual(d['near_silent_intervals'],[])
            self.assertEqual(d['status'],'未发现1秒近静音窗')

    def test_partial_final_window_and_separated_intervals(self):
        data=np.zeros((28,1));data[8:16,0]=[1,-1]*4
        d=audio_tools._activity_report(data,8)
        self.assertEqual([(x['start_frame'],x['end_frame'])for x in d['windows']],[(0,8),(8,16),(16,24),(24,28)])
        self.assertEqual([(x['start_frame'],x['end_frame'])for x in d['near_silent_intervals']],[(0,8),(16,28)])
        self.assertEqual(d['analyzed_frames'],28)
        self.assertTrue(d['windows'][-1]['partial_window'])

    def test_threshold_is_absolute_ac_rms(self):
        wave=np.array([1,-1]*4,dtype=float)[:,None]
        low=audio_tools._activity_report(wave*1e-4,8)
        high=audio_tools._activity_report(wave*1e-3,8)
        self.assertEqual(low['near_silent_frames'],8)
        self.assertEqual(high['near_silent_frames'],0)

    def test_explicit_append_tail_and_crossing_span(self):
        d=audio_tools._activity_report(np.zeros((32,2)),8,{'tail_policy':'append','tail_frames':8})
        self.assertEqual(d['tail_boundary']['start_frame'],24)
        interval=d['near_silent_intervals'][0]
        self.assertEqual(interval['body_overlap_frames'],24)
        self.assertEqual(interval['tail_overlap_frames'],8)
        self.assertEqual(interval['region'],'跨主体与尾音')
        tail=np.array([1,-1]*12+[0]*8,dtype=float)[:,None]
        d=audio_tools._activity_report(tail,8,{'tail_policy':'append','tail_frames':8})
        self.assertEqual(d['near_silent_intervals'][0]['region'],'声明尾音内')
        self.assertEqual(d['status'],'需核')

    def test_invalid_or_unused_tail_not_inferred(self):
        for expected in [{'tail_frames':8},{'tail_policy':'none','tail_frames':8},{'tail_policy':'append','tail_frames':True},{'tail_policy':'append','tail_frames':99}]:
            d=audio_tools._activity_report(np.zeros((16,1)),8,expected)
            self.assertNotEqual(d['tail_boundary']['status'],'已声明')
            self.assertIsNone(d['tail_boundary']['start_frame'])
            json.dumps(d,allow_nan=False)

    def test_nonfinite_file_activity_is_explicitly_unassessed(self):
        with tempfile.TemporaryDirectory()as temp:
            path=Path(temp)/'非有限.wav'
            data=np.zeros((8000,2),np.float32);data[0,0]=np.nan
            wavfile.write(path,8000,data)
            report=audio_tools.check_audio(path,{'sample_format':'float32','sample_rate_hz':8000})
        self.assertEqual(report['status'],'失败')
        self.assertEqual(report['activity_review'],'未评估')
        json.dumps(report,allow_nan=False)

    def test_finite_extreme_values_cannot_publish_infinite_activity(self):
        with np.errstate(over='ignore',invalid='ignore'):
            report=audio_tools._activity_report(np.array([1e200,-1e200]*4)[:,None],8)
        self.assertTrue(np.isfinite(report['windows'][0]['max_channel_ac_rms']))
        json.dumps(report,allow_nan=False)

if __name__=='__main__':unittest.main()
