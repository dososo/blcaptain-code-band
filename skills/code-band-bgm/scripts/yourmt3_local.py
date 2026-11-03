"""本地YourMT3分析适配器：固定外部源码/权重，CPU、严格加载，只输出研究候选。"""
from pathlib import Path
from collections import Counter
import dataclasses,hashlib,json,os,sys,time,signal,socket,threading
try:
 import resource
except ImportError:
 resource=None

def peak_rss_bytes():
 if resource is None:raise ValueError('本地YourMT3适配器尚未验证当前平台的资源计量')
 value=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
 return int(value if sys.platform=='darwin'else value*1024)

def sha256(path):
 h=hashlib.sha256()
 with Path(path).open('rb')as f:
  while block:=f.read(1024*1024):h.update(block)
 return h.hexdigest()
def plain(value):
 if dataclasses.is_dataclass(value):return plain(dataclasses.asdict(value))
 if isinstance(value,dict):return {str(k):plain(v)for k,v in value.items()}
 if isinstance(value,(tuple,list,set)):return [plain(v)for v in value]
 return value.item()if hasattr(value,'item')else value
def write(path,value):Path(path).write_text(json.dumps(plain(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def load(config_path):
 """配置路径相对配置文件；返回可供多clip复用的同一模型会话。"""
 return Session(config_path)

class Session:
 def __init__(self,config_path):
  self.config_path=Path(config_path).resolve();self.config=json.loads(self.config_path.read_text());cfg=self.config;base=self.config_path.parent
  resolve=lambda key:(base/cfg[key]).resolve()if not Path(cfg[key]).is_absolute()else Path(cfg[key])
  self.source_root=resolve('source_root');self.dependencies=resolve('target_dependencies');self.checkpoint=resolve('checkpoint')
  assert self.source_root.is_dir()and self.dependencies.is_dir()and self.checkpoint.is_file()
  if sha256(self.checkpoint)!=cfg['checkpoint_sha256']:raise ValueError('权重SHA不匹配，拒绝加载')
  if any(p.is_symlink()for root in(self.source_root,self.dependencies)for p in root.rglob('*')):raise ValueError('源码/依赖内部含未验证符号链接')
  actual_sources={str(p.relative_to(self.source_root))for p in self.source_root.rglob('*.py')}
  if not actual_sources or actual_sources!=set(cfg['source_sha256']):raise ValueError('来源代码缺失或多出未登记Python文件')
  for name,expected in cfg['source_sha256'].items():
   if sha256(self.source_root/name)!=expected:raise ValueError('固定来源代码SHA不匹配: '+name)
  locked=cfg.get('target_dependency_sha256',{})
  actual_dependencies={str(p.relative_to(self.dependencies))for p in self.dependencies.rglob('*')if p.is_file()and'__pycache__'not in p.parts and p.suffix!='.pyc'}
  if not locked or actual_dependencies!=set(locked):raise ValueError('隔离依赖缺失或多出未登记文件')
  for name,expected in locked.items():
   if sha256(self.dependencies/name)!=expected:raise ValueError('隔离依赖SHA不匹配: '+name)
  sys.path[:0]=[str(self.dependencies),str(self.source_root)]
  os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',WANDB_MODE='disabled',CUDA_VISIBLE_DEVICES='',TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1')
  def deny(*a,**k):raise RuntimeError('本地模型会话禁止网络连接')
  socket.socket.connect=deny;socket.socket.connect_ex=deny;socket.create_connection=deny
  import numpy as np,torch,soundfile as sf,transformers,tokenizers
  from mt3_infer.models.yourmt3.inference_loader import load_model_for_inference
  from mt3_infer.models.yourmt3.utils.audio import slice_padded_array
  from mt3_infer.models.yourmt3.utils.event2note import merge_zipped_note_events_and_ties_to_notes
  from mt3_infer.models.yourmt3.utils.note2event import mix_notes
  if transformers.__version__!='4.45.1'or tokenizers.__version__!='0.20.3':raise ValueError('未验证的推理依赖版本')
  torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.set_default_dtype(torch.float32);torch.manual_seed(0)
  self.np,self.torch,self.sf=np,torch,sf;self.slice,self.merge,self.mix=slice_padded_array,merge_zipped_note_events_and_ties_to_notes,mix_notes
  self.strict=[];original=torch.nn.Module.load_state_dict
  def observed(model,state_dict,strict=True,assign=False):
   if type(model).__name__=='YourMT3':
    if strict is not True:raise ValueError('禁止非严格模型权重加载')
    row={'strict':True,'model_key_count':len(model.state_dict()),'checkpoint_key_count':len(state_dict),'missing_keys':sorted(set(model.state_dict())-set(state_dict)),'unexpected_keys':sorted(set(state_dict)-set(model.state_dict()))};self.strict.append(row)
   return original(model,state_dict,strict=strict,assign=assign)
  torch.nn.Module.load_state_dict=observed;started=time.monotonic()
  try:self.model=load_model_for_inference(str(self.checkpoint),device='cpu',task_name='mc13_full_plus_256')
  finally:torch.nn.Module.load_state_dict=original
  self.model.float().cpu().eval()
  if len(self.strict)!=1 or self.strict[0]['missing_keys']or self.strict[0]['unexpected_keys']:raise ValueError('严格权重键核验失败')
  assert all(p.device.type=='cpu'and(not p.is_floating_point()or p.dtype==torch.float32)for p in self.model.parameters())
  self.runtime={'python':sys.version,'torch':torch.__version__,'transformers':transformers.__version__,'tokenizers':tokenizers.__version__,'device':'cpu','dtype':'float32','batch_size':1,'torch_threads':torch.get_num_threads(),'torch_interop_threads':torch.get_num_interop_threads(),'load_seconds':time.monotonic()-started,'strict_weights':self.strict,'checkpoint_sha256':cfg['checkpoint_sha256'],'source_commit':cfg['source_commit']}
  self.model_load_count=1
 def analyze(self,input_wav,output_directory,prefix_seconds=0.0):
  """只接16k单声道有限音频，保留分块原始解码与删失标记，不静默裁切。"""
  np,torch,sf=self.np,self.torch,self.sf;path=Path(input_wav).resolve();out=Path(output_directory).resolve();out.mkdir(exist_ok=False)
  if prefix_seconds not in(0.0,.25):raise ValueError('本轮仅验证0或0.25秒前置上下文')
  wave,sr=sf.read(path,dtype='float32',always_2d=True)
  if sr!=16000 or wave.shape[1]!=1 or not 0<len(wave)<=16000*30 or not np.isfinite(wave).all():raise ValueError('须为0–30秒、16k单声道有限输入')
  raw=np.concatenate([np.zeros(round(prefix_seconds*sr),dtype='float32'),wave[:,0]])
  pieces=self.slice(raw[None,:],self.model.audio_cfg['input_frames'],self.model.audio_cfg['input_frames']);tensor=torch.from_numpy(pieces.astype('float32')).unsqueeze(1);started=time.monotonic()
  def over_time(signum,frame):raise TimeoutError('单clip超过180秒')
  previous=signal.signal(signal.SIGALRM,over_time);signal.alarm(180)
  try:
   with torch.inference_mode():preds,_=self.model.inference_file(bsz=1,audio_segments=tensor)
   assert preds and all(p.ndim==3 and p.shape[1]==13 and np.issubdtype(p.dtype,np.integer)for p in preds)
   np.savez_compressed(out/'raw_tokens.npz',**{'batch_'+str(i):p for i,p in enumerate(preds)})
   starts=[self.model.audio_cfg['input_frames']*i/16000 for i in range(tensor.shape[0])];channels=[];allnotes=[];errors=Counter();left=[]
   for ch in range(13):
    zipped,events,err=self.model.task_manager.detokenize_list_batches([x[:,ch,:]for x in preds],starts,return_events=True);notes,merge_errors=self.merge(zipped);allnotes.append(notes);errors+=err;errors+=merge_errors
    if zipped:
     for n in zipped[0][1]:left.append({'channel':ch,'program':n.program,'pitch':n.pitch,'is_drum':n.is_drum,'onset':None,'reason':'first_chunk_tie_without_observed_onset'})
    channels.append({'channel':ch,'events':events,'zipped_note_events_and_tie':zipped,'notes':notes,'decode_errors':dict(err),'merge_errors':dict(merge_errors)})
   notes=plain(self.mix(allnotes));boundaries=[self.model.audio_cfg['input_frames']*(i+1)/16000 for i in range(tensor.shape[0])];accepted=[];excluded=[]
   for note in notes:
    if not all(np.isfinite(note[k])for k in ['onset','offset'])or note['offset']<=note['onset']:raise ValueError('模型音符时间异常')
    note['possible_right_censored_at_chunk_boundary']=any(abs(note['offset']-b)<1e-6 for b in boundaries)
    note['onset']-=prefix_seconds;note['offset']-=prefix_seconds
    if not 0<=note['onset']<note['offset']<=len(wave)/sr:excluded.append({**note,'reason':'outside_observed_input_window_not_clipped'})
    else:accepted.append(note)
   write(out/'raw_decode.json',channels)
   result={'status':'analysis_candidates_only','source':{'file':str(path),'sha256':sha256(path),'frames':len(wave),'sample_rate_hz':sr,'channels':1},'runtime':self.runtime,'model_load_count':self.model_load_count,'padding':{'prefix_seconds':prefix_seconds,'waveform_gain':1.0,'time_origin_subtracted':prefix_seconds,'chunk_count':len(preds),'chunk_end_seconds_in_original_origin':[x-prefix_seconds for x in boundaries]},'notes':accepted,'excluded_candidates':excluded,'left_censored_candidates':left,'decode_errors':dict(errors),'program_label_map':self.model.midi_output_inverse_vocab,'raw_tokens':'raw_tokens.npz','raw_decode':'raw_decode.json','elapsed_seconds':time.monotonic()-started,'peak_rss_bytes':peak_rss_bytes(),'limits':['program是模型/词表候选，非真实preset身份','velocity为模型活动标记，非原录音演奏力度','左边缘tie不补造onset；块末offset可能右删失，不当实际音长','窗外候选保留并排除，不静默裁切','不可将参考符号候选送入原创主题；不是配器/和弦准确率签收']};write(out/'analysis.json',result);return result
  finally:signal.alarm(0);signal.signal(signal.SIGALRM,previous)

if __name__=='__main__':
 import argparse
 parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--input',required=True,action='append');parser.add_argument('--out',required=True);parser.add_argument('--prefix-seconds',type=float,default=0.0);args=parser.parse_args();root=Path(args.out).resolve();root.mkdir(exist_ok=False)
 if resource is None or not hasattr(signal,'SIGALRM'):
  write(root/'failure.json',{'type':'unsupported_platform','message':'当前本地模型适配器需要POSIX资源与计时接口；本轮仅实测macOS'})
  sys.exit(2)
 def rss_watch():
  while True:
   peak=peak_rss_bytes()
   if peak>6*1024**3:write(root/'resource_stop.json',{'status':'rss_limit','peak_rss_bytes':peak,'limit':6*1024**3});os._exit(73)
   time.sleep(.25)
 threading.Thread(target=rss_watch,daemon=True).start()
 try:
  session=load(args.config);write(root/'runtime.json',session.runtime)
  for i,path in enumerate(args.input):
   r=session.analyze(path,root/(str(i)+'_'+Path(path).stem),args.prefix_seconds);print(json.dumps({'input':str(path),'notes':len(r['notes']),'excluded':len(r['excluded_candidates']),'seconds':r['elapsed_seconds']},ensure_ascii=False),flush=True)
 except BaseException as e:
  write(root/'failure.json',{'type':type(e).__name__,'message':str(e)});raise
