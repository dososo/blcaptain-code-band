"""局部投影的可辨识范围与文件协议，不能替代真实音源和人工听感。"""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('harmonics', ROOT/'skills/code-band-bgm/scripts/harmonic_observation.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class HarmonicObservationTests(unittest.TestCase):
    def setUp(self):
        self.sr = 16000
        t = np.arange(self.sr)/self.sr
        tone = sum(a*np.sin(2*np.pi*440*h*t+.3) for h,a in enumerate([.1,.03,.01],1))
        background = .04*np.sin(2*np.pi*573*t+.7)
        self.audio = np.column_stack((tone+background,tone+background))

    def test_joint_projection_recovers_known_harmonics(self):
        row, signal = module.project(self.audio,self.sr,.1,.127,[440,573])
        np.testing.assert_allclose(row['target_ratios'][:3],[1,.3,.1],atol=1e-7)
        self.assertTrue(row['numerically_identifiable'])
        self.assertFalse(row['physical_source_identified'])
        self.assertLess(row['weighted_residual_power_ratio'],1e-20)
        self.assertTrue(np.isfinite(signal).all())

    def test_common_fundamental_is_unknown_not_double_assigned(self):
        with self.assertRaisesRegex(ValueError,'共频'):
            module.project(self.audio,self.sr,.1,.15,[440,220])

    def test_gain_does_not_change_normalized_profile(self):
        a,_ = module.project(self.audio,self.sr,.1,.14,[440,573])
        b,_ = module.project(self.audio*.123,self.sr,.1,.14,[440,573])
        np.testing.assert_allclose(a['target_ratios'][:3],b['target_ratios'][:3],atol=1e-10)

    def test_invalid_window_and_nonfinite_audio_fail(self):
        for start,end in [(-.1,.1),(.1,.11),(.9,1.1)]:
            with self.assertRaises(ValueError):
                module.project(self.audio,self.sr,start,end,[440])
        self.audio[1700,0] = np.nan
        with self.assertRaises(ValueError):
            module.project(self.audio,self.sr,.1,.15,[440])

    def make_request(self, folder):
        source = folder/'输入.wav'
        sf.write(source,self.audio,self.sr,subtype='FLOAT')
        request = {'schema_version':'1.0.0','audio_file':source.name,'audio_sha256':module.sha(source),
                   'observations':[{'observation_id':'早观察','start_seconds':.1,'end_seconds':.14,
                                    'f0_candidates_hz':[440,573]}]}
        path = folder/'请求.json'
        path.write_text(json.dumps(request,ensure_ascii=False))
        return path,request

    def test_bound_request_and_failure_paths(self):
        with tempfile.TemporaryDirectory() as td:
            path,request = self.make_request(Path(td))
            result = module.inspect_harmonics(path)
            self.assertEqual(result['status_scope'],'local_harmonic_observation')
            self.assertFalse(result['dry_source_recovered'])
            invalid = [dict(request,audio_sha256='0'*64),dict(request,extra=True)]
            duplicate = copy.deepcopy(request)
            duplicate['observations'] *= 2
            invalid.append(duplicate)
            long_window = copy.deepcopy(request)
            long_window['observations'][0]['end_seconds'] = .8
            invalid.append(long_window)
            for bad in invalid:
                path.write_text(json.dumps(bad,ensure_ascii=False))
                with self.assertRaises(ValueError):
                    module.inspect_harmonics(path)

    def test_fractional_frame_boundaries_are_not_silently_repaired(self):
        for start,end in [(-.000001,.1),(.9,1.000001),(.1,.119999)]:
            with self.assertRaises(ValueError):
                module.project(self.audio,self.sr,start,end,[440])

    def test_request_changed_during_read_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            path,request=self.make_request(Path(td))
            real_read=module.read_audio
            def changed_read(source):
                request['observations'][0]['observation_id']='运行中变更'
                path.write_text(json.dumps(request))
                return real_read(source)
            with patch.object(module,'read_audio',side_effect=changed_read):
                with self.assertRaisesRegex(ValueError,'请求JSON改变'):
                    module.inspect_harmonics(path)

    def test_cli_from_other_cwd_refuses_overwrite_and_fake_output(self):
        with tempfile.TemporaryDirectory() as td:
            folder=Path(td)
            request,data = self.make_request(folder)
            out=folder/'观察 ; $(不能执行).json'
            command=[sys.executable,str(ROOT/'skills/code-band-bgm/scripts/band.py'),'inspect-harmonics',
                     '--request',str(request),'--out',str(out)]
            first=subprocess.run(command,cwd=folder,capture_output=True,text=True)
            self.assertEqual(first.returncode,0,first.stderr+first.stdout)
            before=out.read_bytes()
            self.assertNotEqual(subprocess.run(command,cwd=folder,capture_output=True).returncode,0)
            self.assertEqual(out.read_bytes(),before)
            data['audio_sha256']='0'*64
            request.write_text(json.dumps(data))
            command[-1]=str(folder/'失败不得输出.json')
            self.assertNotEqual(subprocess.run(command,cwd=folder,capture_output=True).returncode,0)
            self.assertFalse((folder/'失败不得输出.json').exists())


if __name__ == '__main__':
    unittest.main()
