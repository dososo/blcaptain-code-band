"""参考语义接口的真实失败回归；不加载可选模型。"""
from pathlib import Path
import importlib.util,json,sys,tempfile,unittest,subprocess
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'skills/code-band-bgm/scripts'))
from audio_tools import AudioError
class ReviewTests(unittest.TestCase):
 def api(self):
  self.assertIsNotNone(importlib.util.find_spec('semantic_review'),'缺少稳定语义审查接口')
  import semantic_review
  return semantic_review
 def answer(self):return {'声音大类':'音乐','主要有调前景的发声特征':'键盘般的击发','节奏与低音的发声特征':'电子打击','可区分的层及其配合':'未知','两个竞争解释':['单一复合音色','两层叠加'],'未知':['具体乐器']}
 def test_exact_fenced_json_is_unwrapped_without_repair(self):
  m=self.api();s=json.dumps(self.answer(),ensure_ascii=False)
  self.assertEqual(m.parse_answer('```json\n'+s+'\n```','roles',100,768),self.answer())
 def test_truncated_json_is_never_usable_even_when_parseable(self):
  m=self.api()
  with self.assertRaises(AudioError):m.parse_answer(json.dumps(self.answer()),'roles',768,768)
 def test_reference_nulls_and_wrong_types_are_rejected(self):
  m=self.api()
  for field,value in [('声音大类',None),('未知','没有'),('两个竞争解释',['a','b','c']),('节奏与低音的发声特征','')]:
   a=self.answer();a[field]=value
   with self.subTest(field=field),self.assertRaises(AudioError):m.parse_answer(json.dumps(a),'roles',100,768)
 def test_unrelated_text_multiple_blocks_and_empty_answer_are_rejected(self):
  m=self.api()
  for s in ['', '说明'+json.dumps(self.answer()),'```json\n{}\n```\n```json\n{}\n```','{}']:
   with self.subTest(s=s),self.assertRaises(AudioError):m.parse_answer(s,'roles',100,768)
 def test_unknown_reference_is_valid_but_never_observed_or_render_ready(self):
  m=self.api();a={k:'未知'for k in list(self.answer())[:4]};a.update(两个竞争解释=[],未知=['不能区分'])
  parsed=m.parse_answer(json.dumps(a),'roles',100,768)
  claims=m.candidate_claims(parsed,60,8,'a'*64)
  self.assertTrue(all(c['status']in['unknown','candidate']for c in claims))
  self.assertFalse(any(c.get('render_ready',False)for c in claims))
  self.assertTrue(all(c['source_sha256']=='a'*64 and c['absolute_window_seconds']==[60,68]for c in claims))
 def test_control_values_are_strict_and_not_rewritten(self):
  m=self.api();a={'声音大类':'音乐','发声特征':'击发后衰减','中间较高短音与首个低音':'重叠','观察':'短音在底音中进入'}
  self.assertEqual(m.parse_answer(json.dumps(a),'control',100,768),a)
  a['发声特征']='真实原钢琴'
  with self.assertRaises(AudioError):m.parse_answer(json.dumps(a),'control',100,768)
 def test_extra_unvalidated_keys_cannot_become_claims(self):
  m=self.api();a=self.answer();a['推荐预设']={'bank':0,'program':5}
  with self.assertRaises(AudioError):m.parse_answer(json.dumps(a),'roles',100,768)
 def test_unknown_list_is_never_a_positive_candidate(self):
  m=self.api();claims=m.candidate_claims(self.answer(),60,8,'a'*64)
  unknown=next(c for c in claims if c['claim_id']=='未知')
  self.assertEqual(unknown['status'],'unknown');self.assertIsNone(unknown['value'])
 def test_import_keeps_mlx_optional(self):
  self.api();self.assertNotIn('mlx',sys.modules)
 def test_existing_output_is_untouched_before_dependencies(self):
  m=self.api();p=Path(tempfile.mkdtemp(prefix='band-review-test-'));out=p/'用户目录';out.mkdir();(out/'原文').write_text('保留')
  with self.assertRaises(AudioError):m.review_reference('缺输入','缺配置','缺解释器',out)
  self.assertEqual((out/'原文').read_text(),'保留')
 def worker(self):
  return {'execution_status':'completed','input_sha256':'a'*64,'frames':8000,'sample_rate_hz':16000,'channels':1,'load':{'strict':True,'model_keys':1433,'checkpoint_keys':1433,'missing_keys':[],'unexpected_keys':[],'shape_errors':[]},'model_weight_sha256':'b'*64,'final_text':json.dumps(self.answer()),'generation_tokens':100,'max_tokens':768,'trace':{'mel_shape':[1,128,50],'valid_mel_frames':[50],'audio_token_lens':[7],'audio_mask_count':7,'encoder_output_shape':[7,1280]}}
 def test_failed_or_inconsistent_worker_never_publishes_old_answer(self):
  m=self.api()
  for bad in [dict(self.worker(),execution_status='失败'),dict(self.worker(),error='失败')]:
   with self.assertRaises(AudioError):m.validate_worker(bad,8000,'a'*64,'b'*64,'roles',768)
  bad=self.worker();bad['load']['missing_keys']=['缺键']
  with self.assertRaises(AudioError):m.validate_worker(bad,8000,'a'*64,'b'*64,'roles',768)
 def test_malformed_subreports_raise_stable_audio_error(self):
  m=self.api()
  for key,value in [('load',[]),('trace','错误类型')]:
   bad=self.worker();bad[key]=value
   with self.subTest(key=key),self.assertRaises(AudioError):m.validate_worker(bad,8000,'a'*64,'b'*64,'roles',768)
  bad=self.worker();bad['trace'].update(mel_shape=[1,128,1],valid_mel_frames=[1],audio_token_lens=[1],audio_mask_count=1,encoder_output_shape=[1,1280])
  with self.assertRaises(AudioError):m.validate_worker(bad,8000,'a'*64,'b'*64,'roles',768)
 def test_worker_source_model_strict_and_actual_trace_must_agree(self):
  m=self.api();v=self.worker();self.assertEqual(m.validate_worker(v,8000,'a'*64,'b'*64,'roles',768),self.answer())
  for field,value in [('input_sha256','c'*64),('frames',7999),('channels',2),('model_weight_sha256','c'*64)]:
   bad=self.worker();bad[field]=value
   with self.subTest(field=field),self.assertRaises(AudioError):m.validate_worker(bad,8000,'a'*64,'b'*64,'roles',768)
  for field,value in [('strict',False),('model_keys',1432),('checkpoint_keys',0)]:
   bad=self.worker();bad['load'][field]=value
   with self.subTest(field=field),self.assertRaises(AudioError):m.validate_worker(bad,8000,'a'*64,'b'*64,'roles',768)
  for field,value in [('audio_mask_count',6),('encoder_output_shape',[6,1280]),('valid_mel_frames',[0])]:
   bad=self.worker();bad['trace'][field]=value
   with self.subTest(field=field),self.assertRaises(AudioError):m.validate_worker(bad,8000,'a'*64,'b'*64,'roles',768)
 def test_invalid_time_windows_fail_without_publishing(self):
  m=self.api();p=Path(tempfile.mkdtemp(prefix='band-review-window-'))
  for start,seconds in [(float('nan'),8),(-1,8),(0,31),(0,.1)]:
   with self.subTest(start=start,seconds=seconds),self.assertRaises(AudioError):m.review_reference('缺输入','缺配置','缺解释器',p/'结果',start,seconds)
  self.assertFalse((p/'结果').exists())
 def test_cli_exposes_candidate_review_and_fails_missing_assets(self):
  band=ROOT/'skills/code-band-bgm/scripts/band.py'
  help_result=subprocess.run([sys.executable,str(band),'review-reference','--help'],capture_output=True,text=True)
  self.assertEqual(help_result.returncode,0,help_result.stderr);self.assertIn('--model-config',help_result.stdout)
  p=Path(tempfile.mkdtemp(prefix='band-review-cli-'))
  result=subprocess.run([sys.executable,str(band),'review-reference','--input','缺音频','--model-config','缺配置','--backend-python',sys.executable,'--out',str(p/'未发布')],capture_output=True,text=True)
  self.assertNotEqual(result.returncode,0);self.assertEqual(json.loads(result.stdout)['status'],'失败');self.assertFalse((p/'未发布').exists())
 def test_backend_failure_and_timeout_preserve_evidence_without_output(self):
  m=self.api();import numpy as np,soundfile as sf
  real_run=subprocess.run
  for timeout in [False,True]:
   p=Path(tempfile.mkdtemp(prefix='band-review-failure-'));audio=p/'原始.wav';sf.write(audio,np.full(8000,.01),16000,subtype='FLOAT')
   cfg=p/'配置.json';cfg.write_text(json.dumps({'backend':'moss_music_mlx','model_assets':{'model.safetensors':'a'*64}}))
   def failed(argv,**kwargs):
    if '-B'in argv:
     if timeout:raise subprocess.TimeoutExpired(argv,240,output='已加载，未完成'.encode())
     return subprocess.CompletedProcess(argv,2,'原始诊断','模型加载失败')
    return real_run(argv,**kwargs)
   with patch.object(m.subprocess,'run',side_effect=failed),self.assertRaises(AudioError):m.review_reference(audio,cfg,sys.executable,p/'未发布',seconds=.5)
   self.assertFalse((p/'未发布').exists());stages=list(p.glob('.review-*'));self.assertEqual(len(stages),1)
   self.assertTrue((stages[0]/'失败记录.json').is_file());self.assertTrue((stages[0]/'input.wav').is_file())
   self.assertTrue((stages[0]/'backend.log').is_file())
