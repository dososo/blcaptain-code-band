"""本地音乐模型审查：执行、结构与候选主张分开，不能自动签收参考。"""
from pathlib import Path
import hashlib,json,math,os,re,subprocess,tempfile
import numpy as np
import soundfile as sf
from audio_tools import AudioError,INPUT_FLAGS,_local,_probe,_run,_hash_file

ROLE_KEYS={'声音大类','主要有调前景的发声特征','节奏与低音的发声特征','可区分的层及其配合','两个竞争解释','未知'}
CONTROL_VALUES={'声音大类':['音乐','噪声','静音','讲话','未知'],'发声特征':['击发后衰减','持续延续','混合','未知'],'中间较高短音与首个低音':['重叠','不重叠','不适用','未知']}
def require(condition,message):
    if not condition:raise AudioError('semantic_review_invalid',message)

def parse_answer(text,task,generated_tokens,max_tokens):
    require(type(generated_tokens)is int and type(max_tokens)is int and 0<generated_tokens<max_tokens,'空输出或token截断，不可作为语义答案')
    require(isinstance(text,str)and bool(text.strip()),'缺少最终回答')
    text=text.strip()
    if text.startswith('```'):
        block=re.fullmatch(r'```(?:json)?\s*\n(.*?)\n```',text,re.S)
        require(block is not None,'只接受一个完整JSON围栏，不提取混合正文')
        text=block.group(1)
    try:value=json.loads(text,parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))
    except (ValueError,TypeError):raise AudioError('semantic_review_invalid','最终回答不是有限JSON对象')from None
    required=ROLE_KEYS if task=='roles'else set(CONTROL_VALUES)|{'观察'}
    require(task in('roles','control')and isinstance(value,dict)and required==set(value),'语义回答字段须与任务精确一致；额外内容只保留在原文')
    strings=required-{'两个竞争解释','未知'}
    require(all(isinstance(value[k],str)and value[k].strip()for k in strings),'语义字段须为非空字符串')
    if task=='roles':
        require(all(isinstance(value[k],list)and all(isinstance(x,str)and x.strip()for x in value[k])for k in ['两个竞争解释','未知']),'解释/未知须为字符串数组')
        require(len(value['两个竞争解释'])<=2,'竞争解释最多两条；没有解释可保留空数组')
    else:
        require(all(value[k]in options for k,options in CONTROL_VALUES.items()),'控制答案不在预先声明的枚举内')
    return value

def candidate_claims(answer,start,seconds,source_sha256):
    claims=[]
    for key,value in answer.items():
        unknown=key=='未知'or value in(None,'未知')or value==[]
        claims.append({'claim_id':key,'value':None if unknown else value,'status':'unknown'if unknown else'candidate','source_sha256':source_sha256,'relative_window_seconds':[0,seconds],'absolute_window_seconds':[start,start+seconds],'evidence_json_pointer':'/answer/'+key,'evidence_scope':'模型整窗候选，不是逐项时间对齐或真实乐器识别','render_ready':False})
    return claims

def validate_worker(report,frames,input_sha,model_sha,task,max_tokens):
    require(isinstance(report,dict)and report.get('execution_status')=='completed'and not report.get('error'),'模型未完成，禁止复用残留答案')
    require(isinstance(report,dict)and report.get('input_sha256')==input_sha and report.get('frames')==frames and report.get('sample_rate_hz')==16000 and report.get('channels')==1,'后端输入SHA/帧数/规格不匹配')
    require(report.get('model_weight_sha256')==model_sha and report.get('max_tokens')==max_tokens,'后端模型或token设置不匹配')
    load=report.get('load',{});require(isinstance(load,dict),'严格加载子报告须为对象');count=load.get('model_keys')
    require(load.get('strict')is True and type(count)is int and count>0 and load.get('checkpoint_keys')==count and all(load.get(k)==[]for k in('missing_keys','unexpected_keys','shape_errors')),'模型必须严格加载且没有缺键/多键/形状错误')
    trace=report.get('trace',{});require(isinstance(trace,dict),'特征trace须为对象');shape=trace.get('mel_shape',[]);lengths=trace.get('valid_mel_frames',[]);tokens=trace.get('audio_token_lens',[])
    mel_frames=frames//160;expected_tokens=(mel_frames+7)//8
    require(shape==[1,128,mel_frames]and lengths==[mel_frames],'实际特征帧与16k输入/hop160不一致')
    require(tokens==[expected_tokens]and expected_tokens==trace.get('audio_mask_count')and trace.get('encoder_output_shape')==[expected_tokens,1280],'实际音频token/编码器输出与三层卷积下采样不一致')
    return parse_answer(report.get('final_text'),task,report.get('generation_tokens'),max_tokens)

def review_reference(input_path,model_config,backend_python,out,start=0,seconds=8,task='roles',max_tokens=768):
    output=Path(out).resolve()
    require(not output.exists(),'输出已存在，禁止覆盖')
    require(all(type(v)in(int,float)and math.isfinite(v)for v in(start,seconds))and start>=0 and .5<=seconds<=30,'窗口须为有限非负起点及0.5至30秒')
    require(task in('roles','control')and type(max_tokens)is int and 256<=max_tokens<=1024,'任务或token预算不支持')
    config=Path(model_config).resolve();cfg=json.loads(config.read_text())
    require(isinstance(cfg,dict)and cfg.get('backend')=='moss_music_mlx'and isinstance(cfg.get('model_assets'),dict),'缺显式本地MOSS后端及资产配置')
    model_sha=cfg['model_assets'].get('model.safetensors');require(isinstance(model_sha,str)and re.fullmatch('[a-f0-9]{64}',model_sha),'缺期望模型SHA256')
    source=_local(input_path);source_sha=_hash_file(source);config_sha=_hash_file(config);meta=_probe(source);require(start+seconds<=meta['container_duration_seconds']+1/16000,'分析窗口超出真实音频')
    args=['ffmpeg','-v','error','-nostdin',*INPUT_FLAGS,'-i',str(source),'-ss',str(start),'-t',str(seconds),'-map','0:a:0','-vn','-ar','16000','-ac','1','-f','f32le','pipe:1']
    data=np.frombuffer(_run(args).stdout,dtype='<f4');frames=round(seconds*16000)
    require(len(data)==frames and np.isfinite(data).all(),'输入解码帧数或有限值错误')
    if task=='roles':require(float(np.max(np.abs(data)))>1e-7,'静音不进入参考角色推断；静音控制请显式使用control任务')
    output.parent.mkdir(parents=True,exist_ok=True);stage=Path(tempfile.mkdtemp(prefix='.review-',dir=output.parent));prepared=stage/'input.wav';sf.write(prepared,data,16000,subtype='FLOAT')
    worker=Path(__file__).with_name('moss_review_backend.py');argv=[os.path.abspath(backend_python),'-B',str(worker),'--config',str(config),'--input',str(prepared),'--task',task,'--tokens',str(max_tokens),'--out',str(stage/'model_result.json')]
    try:
        try:process=subprocess.run(argv,capture_output=True,text=True,timeout=240,check=False)
        except subprocess.TimeoutExpired as exc:
            text=lambda value:value.decode(errors='replace')if isinstance(value,bytes)else(value or'')
            (stage/'backend.log').write_text(text(exc.stdout)+text(exc.stderr))
            raise AudioError('semantic_review_timeout','整个模型子进程超过240秒（含加载、预处理和生成）')from exc
        (stage/'backend.log').write_text(process.stdout+process.stderr)
        require(process.returncode==0,'隔离音乐模型执行失败，保留日志、不发布假结果')
        require(_hash_file(source)==source_sha and _hash_file(config)==config_sha,'执行期间源音频或后端配置发生变化，不能混淆证据')
        raw=json.loads((stage/'model_result.json').read_text());answer=validate_worker(raw,frames,_hash_file(prepared),model_sha,task,max_tokens)
        report={'status':'通过','status_scope':'local_music_model_execution_and_structure','semantic_validation':'未经核准；候选/未知不能自动决定配器或放行新曲','ready_for_composition':False,'source':{'file_name':source.name,'file_sha256':source_sha},'window':{'start_seconds':start,'seconds':seconds,'frames':frames,'sample_rate_hz':16000,'channels':1,'prepared_sha256':_hash_file(prepared)},'answer':answer,'claims':candidate_claims(answer,start,seconds,source_sha),'backend_config_sha256':config_sha,'worker_sha256':_hash_file(worker),'runtime':raw.get('runtime',{}),'load':raw['load'],'trace':raw['trace'],'commands':{'decode':args,'backend':argv},'limits':['当前后端仅实测Apple Silicon','没有人工听辨，不保证原乐器/和弦/关系正确','代码围栏仅无损解包，不补造答案；原文与截断记录保留']}
        (stage/'review.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
        require(not output.exists(),'发布前输出目录已出现，禁止覆盖');stage.rename(output)
        return {'status':'通过','status_scope':report['status_scope'],'out':str(output),'review':str(output/'review.json'),'ready_for_composition':False,'semantic_validation':report['semantic_validation']}
    except BaseException as exc:
        (stage/'失败记录.json').write_text(json.dumps({'status':'失败','reason':str(exc),'published':False},ensure_ascii=False,indent=2)+'\n')
        if isinstance(exc,AudioError):exc.details['failure_evidence']=str(stage)
        raise
