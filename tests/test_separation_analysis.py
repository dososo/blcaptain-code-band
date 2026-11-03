"""分离契约/失败发布回归；模型真实烟测另存，不把模拟后端当音色验收。"""
import hashlib
import importlib
import importlib.util
import json
import os
import py_compile
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'skills/code-band-bgm/scripts'))
from audio_tools import AudioError


class SeparationAnalysisTests(unittest.TestCase):
    def module(self,name='separate_analysis'):
        self.assertIsNotNone(importlib.util.find_spec(name),'应提供独立可选分离模块')
        return importlib.import_module(name)

    def files(self):
        base=Path(tempfile.mkdtemp(prefix='code-band-separation-test-'))
        source=base/'输入 $(禁止执行) 空格.wav'
        y=np.column_stack([np.sin(np.arange(22050)*.02),np.cos(np.arange(22050)*.02)]).astype(np.float32)*.05
        sf.write(source,y,44100,subtype='FLOAT')
        config=base/'模型.json';config.write_text(json.dumps({'checkpoint_sha256':'b'*64}))
        target=base/'基础解释器';target.write_text('测试仅检查venv路径')
        python=base/'venv/bin/python';python.parent.mkdir(parents=True);python.symlink_to(target)
        return base,source,config,python,base/'输出'

    def report_files(self):
        mod=self.module();base=Path(tempfile.mkdtemp(prefix='code-band-separation-report-'))
        inp=base/'input.wav';x=np.full((22050,2),.01,dtype=np.float32);sf.write(inp,x,44100,subtype='FLOAT')
        outputs={}
        for name in (*mod.SOURCES,'reconstruction','residual'):
            a=x/6 if name in mod.SOURCES else x if name=='reconstruction' else np.zeros_like(x)
            file=base/(name+'.wav');sf.write(file,a,44100,subtype='FLOAT')
            outputs[name]={'file':file.name,'sha256':hashlib.sha256(file.read_bytes()).hexdigest()}
        report={'status':'separation_estimates_only','ready_for_composition':False,'labels_are_physical_identities':False,
                'source':{'sha256':hashlib.sha256(inp.read_bytes()).hexdigest(),'frames':22050,'sample_rate_hz':44100,'channels':2},
                'runtime':{'checkpoint_sha256':'b'*64,'device':'cpu','dtype':'float32','torch_threads':2,'network_blocked':True,
                           'strict_state':True,'missing_keys':[],'unexpected_keys':[], 'peak_rss_bytes':1000,
                           'model_key_count':525,'checkpoint_key_count':525},'outputs':outputs}
        return mod,base,inp,report

    def test_valid_six_track_report_is_only_an_estimate(self):
        mod,base,inp,r=self.report_files()
        self.assertIs(mod.validate_separation(r,base,22050,hashlib.sha256(inp.read_bytes()).hexdigest(),'b'*64,inp),r)
        self.assertFalse(r['ready_for_composition'])

    def test_wrong_binding_gpu_or_identity_claim_is_rejected(self):
        for section,key,value in [('source','sha256','c'*64),('source','frames',22049),
                                  ('runtime','device','mps'),('runtime','torch_threads',8),
                                  ('runtime','checkpoint_sha256','c'*64),('runtime','network_blocked',False),
                                  ('runtime','network_blocked',1),('runtime','torch_threads',2.0),('runtime','strict_state',1),
                                  ('runtime','missing_keys',['weight']),('runtime','strict_state',False),
                                  ('runtime','model_key_count',524),('runtime','model_key_count',525.0),
                                  ('runtime','checkpoint_key_count',True),('runtime','checkpoint_key_count',None),
                                  ('root','ready_for_composition',True),('root','labels_are_physical_identities',True)]:
            with self.subTest(section=section,key=key):
                mod,base,inp,r=self.report_files()
                (r if section=='root' else r[section])[key]=value
                with self.assertRaises(AudioError):mod.validate_separation(r,base,22050,r['source']['sha256'] if key!='sha256' else 'a'*64,'b'*64,inp)

    def test_output_path_hash_format_and_nonfinite_are_rejected(self):
        for mode in ['escape','hash','nan','rate','frames','subtype','symlink','missing']:
            with self.subTest(mode=mode):
                mod,base,inp,r=self.report_files();item=r['outputs']['piano'];path=base/item['file']
                if mode=='escape':item['file']='../outside.wav'
                elif mode=='hash':item['sha256']='0'*64
                elif mode=='missing':del r['outputs']['piano']
                elif mode=='symlink':
                    alias=base/'alias.wav';alias.symlink_to(path);item['file']=alias.name;item['sha256']=hashlib.sha256(alias.read_bytes()).hexdigest()
                else:
                    a=np.zeros((22050,2),dtype=np.float32)
                    if mode=='nan':a[0,0]=np.nan
                    if mode=='frames':a=a[:-1]
                    sf.write(path,a,48000 if mode=='rate' else 44100,subtype='PCM_24' if mode=='subtype' else 'FLOAT')
                    item['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
                with self.assertRaises(AudioError):mod.validate_separation(r,base,22050,r['source']['sha256'],'b'*64,inp)

    def test_reconstruction_must_match_saved_stems_and_input(self):
        mod,base,inp,r=self.report_files();path=base/'reconstruction.wav'
        sf.write(path,np.zeros((22050,2)),44100,subtype='FLOAT');r['outputs']['reconstruction']['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
        with self.assertRaises(AudioError):mod.validate_separation(r,base,22050,r['source']['sha256'],'b'*64,inp)

    def test_bad_window_and_existing_output_never_call_backend(self):
        mod=self.module();base,source,config,python,out=self.files()
        for kwargs in [dict(seconds=.49),dict(seconds=31),dict(start=-1),dict(seconds=float('nan')),dict(seconds=True)]:
            with self.subTest(kwargs=kwargs),self.assertRaises(AudioError):mod.separate_reference(source,config,python,out,**kwargs)
        out.mkdir();marker=out/'保留.txt';marker.write_text('不覆盖')
        with patch.object(mod,'_probe') as probe,self.assertRaises(AudioError):mod.separate_reference(source,config,python,out,seconds=.5)
        probe.assert_not_called();self.assertEqual(marker.read_text(),'不覆盖')

    def test_backend_failure_preserves_log_and_venv_path_without_publish(self):
        mod=self.module();base,source,config,python,out=self.files();calls=[]
        def fail(argv,**kwargs):
            calls.append((argv,kwargs));return SimpleNamespace(returncode=2,stdout='已开始',stderr='受控失败')
        with patch.object(mod,'subprocess',SimpleNamespace(run=fail,TimeoutExpired=subprocess.TimeoutExpired)):
            with self.assertRaises(AudioError) as err:mod.separate_reference(source,config,python,out,seconds=.5)
        self.assertEqual(err.exception.code,'separation_backend_failed');self.assertFalse(out.exists())
        self.assertEqual(calls[0][0][0],os.path.abspath(python));self.assertNotEqual(calls[0][0][0],str(python.resolve()))
        self.assertFalse(calls[0][1].get('shell',False));self.assertEqual(calls[0][1]['timeout'],210)
        failure=list(base.glob('.separate-*/failure.json'));self.assertEqual(len(failure),1)
        self.assertIn('受控失败',(failure[0].parent/'backend.log').read_text())

    def test_timeout_and_invalid_config_do_not_publish(self):
        mod=self.module();base,source,config,python,out=self.files()
        def timeout(argv,**kwargs):raise subprocess.TimeoutExpired(argv,210,output='部分输出')
        with patch.object(mod,'subprocess',SimpleNamespace(run=timeout,TimeoutExpired=subprocess.TimeoutExpired)):
            with self.assertRaises(AudioError) as err:mod.separate_reference(source,config,python,out,seconds=.5)
        self.assertEqual(err.exception.code,'separation_backend_timeout');self.assertFalse(out.exists())
        config.write_text('[]')
        with self.assertRaises(AudioError):mod.separate_reference(source,config,python,base/'另一输出',seconds=.5)

    def test_exit_zero_decode_error_with_exact_frames_stops_before_backend(self):
        mod=self.module();base,source,config,python,out=self.files()
        wave=np.zeros((22050,2),dtype='<f4')
        decoded=SimpleNamespace(returncode=0,stdout=wave.tobytes(),stderr=b'Error while decoding stream #0:0: Invalid data')
        with patch.object(mod,'_run',return_value=decoded),patch.object(mod,'subprocess') as backend:
            backend.run.return_value=SimpleNamespace(returncode=2,stdout='',stderr='不应执行')
            with self.assertRaises(AudioError) as err:mod.separate_reference(source,config,python,out,seconds=.5)
        self.assertEqual(err.exception.code,'separation_decode');backend.run.assert_not_called()
        self.assertFalse(out.exists());self.assertEqual(len(list(base.glob('.separate-*/failure.json'))),1)

    def test_private_empty_pycache_prefix_avoids_same_timestamp_stale_bytecode(self):
        mod=self.module();base,source,config,python,out=self.files()
        module=base/'locked_toy.py';module.write_text("VALUE='old'\n");stamp=module.stat()
        py_compile.compile(str(module),doraise=True)
        module.write_text("VALUE='new'\n");os.utime(module,ns=(stamp.st_atime_ns,stamp.st_mtime_ns))
        code='import sys;sys.path.insert(0,'+repr(str(base))+');import locked_toy;print(locked_toy.VALUE)'
        real_run=subprocess.run
        ordinary=real_run([sys.executable,'-B','-c',code],capture_output=True,text=True,timeout=20)
        self.assertEqual(ordinary.stdout.strip(),'old','正控必须真实复现-B仍读取陈旧pyc')
        observed=[];caches=[]
        def backend(argv,**kwargs):
            at=next(i for i,v in enumerate(argv) if v.endswith('demucs_local.py'))
            cache=[Path(v.split('=',1)[1]) for v in argv[:at] if v.startswith('pycache_prefix=')]
            before=list(cache[0].iterdir()) if cache else None
            p=real_run([sys.executable,*argv[1:at],'-c',code],capture_output=True,text=True,timeout=20)
            observed.append((p.returncode,p.stdout.strip()));caches.append((cache,before,list(cache[0].iterdir()) if cache else None))
            return SimpleNamespace(returncode=2,stdout='',stderr='完成解释器隔离诊断，不加载模型')
        with patch.object(mod,'subprocess',SimpleNamespace(run=backend,TimeoutExpired=subprocess.TimeoutExpired)):
            with self.assertRaises(AudioError):mod.separate_reference(source,config,python,out,seconds=.5)
        self.assertEqual(observed,[(0,'new')]);self.assertEqual(len(caches[0][0]),1)
        self.assertEqual(caches[0][1],[]);self.assertEqual(caches[0][2],[])

    def test_success_is_exact_float_window_and_source_mutation_blocks_publish(self):
        mod=self.module()
        for mutate in [False,True,'after_validation','output_after_validation','report_after_validation']:
            with self.subTest(mutate=mutate):
                base,source,config,python,out=self.files()
                def simulated_backend(argv,**kwargs):
                    prepared=Path(argv[argv.index('--input')+1]);backend=Path(argv[argv.index('--out')+1]);backend.mkdir()
                    x,rate=sf.read(prepared,dtype='float32',always_2d=True)
                    self.assertEqual((rate,x.shape),(44100,(22050,2)))
                    outputs={}
                    for name in (*mod.SOURCES,'reconstruction','residual'):
                        a=x/6 if name in mod.SOURCES else x if name=='reconstruction' else np.zeros_like(x)
                        file=backend/(name+'.wav');sf.write(file,a,rate,subtype='FLOAT')
                        outputs[name]={'file':file.name,'sha256':hashlib.sha256(file.read_bytes()).hexdigest()}
                    report={'status':'separation_estimates_only','ready_for_composition':False,'labels_are_physical_identities':False,
                            'source':{'sha256':hashlib.sha256(prepared.read_bytes()).hexdigest(),'frames':22050,'sample_rate_hz':44100,'channels':2},
                            'runtime':{'checkpoint_sha256':'b'*64,'device':'cpu','dtype':'float32','torch_threads':2,'network_blocked':True,
                                       'strict_state':True,'missing_keys':[],'unexpected_keys':[], 'peak_rss_bytes':1000,
                                       'model_key_count':525,'checkpoint_key_count':525},'outputs':outputs}
                    (backend/'report.json').write_text(json.dumps(report))
                    if mutate is True:sf.write(source,x*.5,rate,subtype='FLOAT')
                    return SimpleNamespace(returncode=0,stdout='模拟契约输出，非真实模型',stderr='')
                original_validate=mod.validate_separation
                def validated(*args):
                    r=original_validate(*args)
                    if mutate=='after_validation':sf.write(source,np.zeros((22050,2)),44100,subtype='FLOAT')
                    if mutate=='output_after_validation':sf.write(Path(args[1])/'piano.wav',np.zeros((22050,2)),44100,subtype='FLOAT')
                    if mutate=='report_after_validation':
                        report_file=Path(args[1])/'report.json';report_file.write_text(report_file.read_text()+'\n')
                    return r
                with patch.object(mod,'subprocess',SimpleNamespace(run=simulated_backend,TimeoutExpired=subprocess.TimeoutExpired)),patch.object(mod,'validate_separation',side_effect=validated):
                    if mutate:
                        with self.assertRaises(AudioError) as error:mod.separate_reference(source,config,python,out,seconds=.5)
                        self.assertEqual(error.exception.code,'separation_outputs_changed' if mutate in ('output_after_validation','report_after_validation') else 'separation_source_changed');self.assertFalse(out.exists())
                    else:
                        result=mod.separate_reference(source,config,python,out,seconds=.5)
                        self.assertFalse(result['ready_for_composition']);self.assertTrue((out/'analysis.json').is_file())
                        delivered=json.loads((out/'analysis.json').read_text())
                        self.assertEqual(delivered.get('raw_report_sha256'),hashlib.sha256((out/'backend/report.json').read_bytes()).hexdigest())

    def test_registration_rejects_wrong_weights_before_runtime_probe(self):
        mod=self.module('register_separation_backend');base,source,config,python,out=self.files()
        with self.assertRaises(ValueError):mod.register_backend(source,python,base/'注册.json')
        self.assertFalse((base/'注册.json').exists())

    def test_import_has_no_torch_demucs_or_registration_side_effect(self):
        self.module()
        code='import sys;sys.path.insert(0,'+repr(str(ROOT/'skills/code-band-bgm/scripts'))+');import separate_analysis,register_separation_backend;assert "torch" not in sys.modules;assert "demucs" not in sys.modules;print("无可选模型导入")'
        p=subprocess.run([sys.executable,'-B','-c',code],capture_output=True,text=True,timeout=20)
        self.assertEqual(p.returncode,0,p.stderr)

if __name__=='__main__':unittest.main()
