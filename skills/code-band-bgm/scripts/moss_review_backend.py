"""可选MOSS本地后端：封存资产、隔离执行、保留原文，不核准配器。"""
from pathlib import Path
import argparse,hashlib,importlib.metadata as md,json,os,platform,re,socket,struct,sys,time

VERSIONS=('mlx','mlx-audio','transformers','numpy','soundfile')
CONTROL='请仅依据音频判断，不依据文件名、不猜音符名、调性或BPM。只返回一个JSON对象，不加Markdown。字段："声音大类"取"音乐"/"噪声"/"静音"/"讲话"/"未知"；"发声特征"取"击发后衰减"/"持续延续"/"混合"/"未知"；"中间较高短音与首个低音"取"重叠"/"不重叠"/"不适用"/"未知"；"观察"为最多两句直接听到的现象，不推测物理乐器名称。该短音或初音不能辨认时请填未知。'
ROLES='除‘两个竞争解释’和‘未知’为字符串数组外，其余字段必须为非空字符串。竞争解释最多两条，无法判断可只给‘未知’。请只依据这段音频观察，不猜标题作者、不指定GM预设、不编造音符或精确和弦。仅返回JSON对象："声音大类"、"主要有调前景的发声特征"、"节奏与低音的发声特征"、"可区分的层及其配合"、"两个竞争解释"、"未知"。每个观察限一两句；时间仅在能辨认时写大致相对区间，不把输入标记当定位证据。无法判断的项明确未知。不要创作或推荐配器。'

def sha(path):
    result=hashlib.sha256()
    with Path(path).open('rb')as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):result.update(block)
    return result.hexdigest()

def model_assets(root):
    root=Path(root)
    if not root.is_dir():raise ValueError('本地模型目录不存在')
    files=list(root.rglob('*'))
    if any(p.is_symlink()or p.suffix in('.py','.pyc')for p in files):raise ValueError('模型目录含未准入符号链接或可执行源码')
    assets={str(p.relative_to(root)):sha(p)for p in sorted(files)if p.is_file()}
    if not {'model.safetensors','config.json'}<=set(assets)or not('tokenizer.json'in assets or{'vocab.json','merges.txt'}<=set(assets)):raise ValueError('模型/配置/标准分词器文件不完整')
    return assets

def verify_assets(root,expected):
    if not isinstance(expected,dict)or not expected or any(Path(k).is_absolute()or'..'in Path(k).parts or not re.fullmatch('[a-f0-9]{64}',v)for k,v in expected.items()):raise ValueError('资产摘要或相对路径无效')
    if model_assets(root)!=expected:raise ValueError('模型文件集合或内容已改变，拒绝加载')

def runtime_lock():
    versions={name:md.version(name)for name in VERSIONS};sources={}
    for name in('mlx-audio',):
        dist=md.distribution(name)
        sources[name]={str(p):sha(dist.locate_file(p))for p in dist.files if str(p).endswith('.py')}
        if not sources[name]:raise ValueError('缺少可核本地实现源码')
    return {'versions':versions,'source_sha256':sources,'scope':'MLX-Audio全部已装Python源码（含内置LM实现）及五项依赖版本；非全系统二进制可复现保证'}

def register(model_dir,expected_sha256,out):
    output=Path(out).absolute()
    if output.exists():raise ValueError('配置已存在，禁止覆盖')
    root=Path(model_dir).resolve();assets=model_assets(root)
    if assets['model.safetensors']!=expected_sha256:raise ValueError('权重与显式期望SHA256不一致')
    config=json.loads((root/'config.json').read_text());tokenizer=json.loads((root/'tokenizer_config.json').read_text())
    if config.get('model_type')!='moss_music'or tokenizer.get('auto_map'):raise ValueError('仅准入已核MOSS结构及无动态源码的标准分词器')
    value={'backend':'moss_music_mlx','model_dir':os.path.relpath(root,output.parent),'model_assets':assets,'runtime_lock':runtime_lock(),'license_scope':'操作者另核模型许可与来源；本配置仅锁定已取得文件，不下载、不推断授权'}
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x')as stream:json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
    return value

def run(config_path,input_path,task,tokens,out):
    output=Path(out)
    if output.exists():raise ValueError('后端输出已存在，禁止覆盖')
    if sys.platform!='darwin'or platform.machine()!='arm64':raise ValueError('当前MOSS MLX后端仅实测Apple Silicon，不自动改用另一模型')
    if task not in('control','roles')or not 256<=tokens<=1024:raise ValueError('任务或预算无效')
    config_path=Path(config_path).resolve();cfg=json.loads(config_path.read_text());root=(config_path.parent/cfg['model_dir']).resolve()
    if cfg.get('backend')!='moss_music_mlx':raise ValueError('后端不匹配')
    verify_assets(root,cfg['model_assets'])
    if runtime_lock()!=cfg['runtime_lock']:raise ValueError('推理实现源码或依赖版本已改变，须重新审查并登记')
    tokenizer=json.loads((root/'tokenizer_config.json').read_text())
    if tokenizer.get('auto_map'):raise ValueError('禁止模型目录动态分词器代码')
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',TOKENIZERS_PARALLELISM='false',HF_HUB_DISABLE_IMPLICIT_TOKEN='1')
    def deny(*a,**k):raise RuntimeError('本地音乐审查禁止网络连接')
    socket.socket.connect=deny;socket.socket.connect_ex=deny;socket.create_connection=deny
    import numpy as np
    import soundfile as sf
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlx_audio.stt.utils import load_model
    wave,sr=sf.read(input_path,dtype='float32',always_2d=True)
    if sr!=16000 or wave.shape[1]!=1 or not 8000<=len(wave)<=480000 or not np.isfinite(wave).all():raise ValueError('需0.5至30秒16k单声道有限音频')
    mx.set_memory_limit(14*1024**3);mx.random.seed(20261031)
    started=time.monotonic();model=load_model(root,strict=True,lazy=False);model.eval();mx.eval(model.parameters())
    with(root/'model.safetensors').open('rb')as stream:
        size=struct.unpack('<Q',stream.read(8))[0]
        if size>10*1024**2:raise ValueError('权重头过大')
        header=json.loads(stream.read(size))
    header.pop('__metadata__',None)
    views={k:np.broadcast_to(np.array(0,dtype='uint8'),v['shape'])for k,v in header.items()}
    shapes={k:list(v.shape)for k,v in model.sanitize(views).items()};params=dict(tree_flatten(model.parameters()))
    load={'strict':True,'checkpoint_keys':len(header),'model_keys':len(params),'missing_keys':sorted(set(params)-set(shapes)),'unexpected_keys':sorted(set(shapes)-set(params)),'shape_errors':[k for k in set(params)&set(shapes)if list(params[k].shape)!=shapes[k]],'load_seconds':time.monotonic()-started}
    if load['checkpoint_keys']!=load['model_keys']or any(load[k]for k in('missing_keys','unexpected_keys','shape_errors')):raise ValueError('严格权重键/形状复核失败')
    trace={}
    class Capture:
        def __init__(self,delegate):self.delegate=delegate
        def __getattr__(self,key):return getattr(self.delegate,key)
        def __call__(self,**kwargs):
            data=self.delegate(**kwargs)
            trace.update(mel_shape=list(data.audio_data.shape),valid_mel_frames=np.asarray(data.audio_data_seqlens).tolist(),audio_token_lens=data.token_lens,audio_mask_count=int(mx.sum(data.audio_input_mask).item()),mel_sha256=hashlib.sha256(np.asarray(data.audio_data).tobytes()).hexdigest())
            return data
    model._processor=Capture(model._ensure_processor());original=model._encode_audio_chunks
    def encode(data,lengths):
        encoded=original(data,lengths);trace.update(encoder_input_dtype=str(data.dtype),encoder_output_shape=list(encoded[0].shape));return encoded
    model._encode_audio_chunks=encode;prompt=CONTROL if task=='control'else ROLES
    result=model.generate(audio=wave[:,0],prompt=prompt,max_tokens=tokens,temperature=0,strip_thinking=False,enable_time_marker=True)
    report={'execution_status':'completed','input_sha256':sha(input_path),'frames':len(wave),'sample_rate_hz':sr,'channels':1,'model_weight_sha256':cfg['model_assets']['model.safetensors'],'load':load,'trace':trace,'raw_text':result.text,'final_text':model._strip_thinking(result.text).strip(),'generation_tokens':result.generation_tokens,'max_tokens':tokens,'prompt':prompt,'elapsed_seconds':time.monotonic()-started,'runtime':{'versions':cfg['runtime_lock']['versions'],'mlx_peak_memory_bytes':mx.get_peak_memory(),'mlx_memory_limit_bytes':14*1024**3,'seed':20261031,'temperature':0,'network':'离线环境变量与Python socket拒绝；非操作系统网络沙箱','trace_scope':'本次generate内部真实processor/encoder调用；不是另一次预计算'}}
    with output.open('x')as stream:json.dump(report,stream,ensure_ascii=False,indent=2,allow_nan=False)
    return {'status':'模型已实际执行，语义未经核准','output':str(output)}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config');parser.add_argument('--input');parser.add_argument('--task',choices=['control','roles'],default='roles');parser.add_argument('--tokens',type=int,default=768);parser.add_argument('--out',required=True);parser.add_argument('--register-model');parser.add_argument('--expected-sha256');args=parser.parse_args()
    try:
        if args.register_model:
            value=register(args.register_model,args.expected_sha256,args.out);print(json.dumps({'status':'本地资产已封存，未运行模型','assets':len(value['model_assets']),'out':args.out},ensure_ascii=False))
        else:print(json.dumps(run(args.config,args.input,args.task,args.tokens,args.out),ensure_ascii=False))
    except Exception as exc:
        print(json.dumps({'status':'失败','type':type(exc).__name__,'message':str(exc)},ensure_ascii=False));sys.exit(2)
