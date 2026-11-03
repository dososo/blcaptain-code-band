"""已给候选基频的局部多正弦联合投影；不自动识别乐器或未知混响。"""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy import linalg, signal

PARAMETERS = {
    'analysis_rate': 16000, 'max_harmonics': 32, 'max_frequency_hz': 7200,
    'merge_hz': 10.0, 'svd_cutoff': 1e-6, 'maximum_condition': 10000,
    'window': 'tukey_0.25', 'early_after_onset_seconds': [0.008, 0.035],
    'body_after_onset_seconds': [0.12, 0.24],
    'scope': '显式F0候选的联合投影；合并近共频基，输出归属歧义。早窗不自动等干声。',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_audio(path):
    import soundfile as sf
    from math import gcd
    x, rate = sf.read(path, always_2d=True)
    if not np.isfinite(x).all() or x.shape[1] != 2 or not np.any(x):
        raise ValueError('需要有限非静音双声道音频')
    target = PARAMETERS['analysis_rate']
    g = gcd(rate, target)
    return signal.resample_poly(x, target//g, rate//g), target


def basis_frequencies(f0s, rate):
    if not f0s or any(not np.isfinite(f) or f < 65 or f > 1500 for f in f0s):
        raise ValueError('候选基频须为65至1500Hz有限数')
    candidates = sorted((f * h, k, h) for k, f in enumerate(f0s)
                        for h in range(1, PARAMETERS['max_harmonics'] + 1)
                        if f*h < min(PARAMETERS['max_frequency_hz'], rate * .45))
    clusters = []
    for freq, k, h in candidates:
        if clusters and freq - clusters[-1][0][0] <= PARAMETERS['merge_hz']:
            clusters[-1].append((freq, k, h))
        else:
            clusters.append([(freq, k, h)])
    return clusters


def project(audio, rate, start, end, f0s):
    """共享频率设计同时拟合左右声道；不以mono相消幅度估计音体。"""
    if not 0 <= start < end <= len(audio)/rate or end-start < .02-1e-12 or end-start > .5+1e-12:
        raise ValueError('原时间窗须在输入内且长20至500ms，不能靠取整修复越界')
    a, b = round(start*rate), round(end*rate)
    if not 0 <= a < b <= len(audio) or b-a < round(.02*rate):
        raise ValueError('观察窗须在输入内且至少20ms')
    x = np.asarray(audio[a:b], dtype=np.float64)
    if not np.isfinite(x).all() or x.ndim != 2 or x.shape[1] != 2:
        raise ValueError('局部输入须有限双声道')
    clusters = basis_frequencies(f0s, rate)
    freqs = np.array([np.mean([v[0] for v in group]) for group in clusters])
    t = np.arange(len(x))/rate
    phase = 2*np.pi*t[:,None]*freqs
    basis = np.concatenate([np.cos(phase), np.sin(phase), np.ones((len(x),1))], axis=1)
    weight = np.sqrt(signal.windows.tukey(len(x), alpha=.25))
    coeff, _, rank, singular = linalg.lstsq(basis*weight[:,None], x*weight[:,None],
                                           cond=PARAMETERS['svd_cutoff'], lapack_driver='gelsd')
    condition = float(singular[0]/max(singular[-1],1e-30))
    count = len(freqs)
    amplitudes = np.sqrt(np.mean(coeff[:count]**2 + coeff[count:2*count]**2, axis=1))
    profiles = np.full((len(f0s), PARAMETERS['max_harmonics']), np.nan)
    uncertainties = []
    target_signal = np.zeros_like(x)
    for i, cluster in enumerate(clusters):
        owners = {p[1] for p in cluster}
        if len(owners) == 1:
            for _, k, h in cluster:
                profiles[k,h-1] = amplitudes[i]
            if owners == {0}:
                target_signal += basis[:,i,None]*coeff[i] + basis[:,count+i,None]*coeff[count+i]
        else:
            uncertainties.append({'frequency_hz':float(freqs[i]),
                                  'owners':[{'group':k,'harmonic':h} for _,k,h in cluster]})
    recon = np.einsum('ij,jk->ik', basis, coeff, optimize=False)
    power = float(np.sum((x*weight[:,None])**2))
    error = float(np.sum(((x-recon)*weight[:,None])**2)/max(power,1e-30))
    target = profiles[0]
    valid = np.isfinite(target)
    if not valid[0] or target[0] < 1e-10:
        raise ValueError('目标基频缺失或与竞争解释共频，不能归一为已识音体')
    ratios = target/target[0]
    mid = (x[:,0]+x[:,1])*.5
    side = (x[:,0]-x[:,1])*.5
    info = {
        'window_seconds':[start,end], 'f0_candidates_hz':list(f0s),
        'analysis_sample_rate_hz':rate, 'analysis_frame_interval':[a,b],
        'target_amplitudes':[float(v) if np.isfinite(v) else None for v in target],
        'target_ratios':[float(v) if np.isfinite(v) else None for v in ratios],
        'shared_partials':uncertainties, 'rank':int(rank), 'columns':basis.shape[1],
        'condition_number':condition, 'weighted_residual_power_ratio':error,
        'target_projection_power_ratio':float(np.sum((target_signal*weight[:,None])**2)/max(power,1e-30)),
        'side_mid_db':float(10*np.log10(max(np.sum(side**2),1e-30)/max(np.sum(mid**2),1e-30))),
        'numerically_identifiable':rank==basis.shape[1] and condition<=PARAMETERS['maximum_condition'],
        'physical_source_identified':False,
    }
    return info, target_signal


def early_body(path, onset, f0s):
    audio, rate = read_audio(path)
    result = {'source':str(Path(path).resolve()),'source_sha256':sha(path),'parameters':PARAMETERS}
    for phase in ('early','body'):
        a,b = PARAMETERS[phase+'_after_onset_seconds']
        result[phase], _ = project(audio, rate, onset+a, onset+b, f0s)
    return result


def inspect_harmonics(request_file):
    """检查显式候选和观察窗；输入与输出均可追踪，不自动签原器或干声。"""
    request_path = Path(request_file).resolve()
    if request_path.stat().st_size > 1024**2:
        raise ValueError('观察请求超过1MiB')
    request_bytes = request_path.read_bytes()
    request_sha = hashlib.sha256(request_bytes).hexdigest()
    request = json.loads(request_bytes)
    fields = {'schema_version','audio_file','audio_sha256','observations'}
    if not isinstance(request, dict) or set(request) != fields or request['schema_version'] != '1.0.0':
        raise ValueError('不支持的谐波观察请求或字段')
    source = (request_path.parent / request['audio_file']).resolve()
    if not source.is_file() or sha(source) != request['audio_sha256']:
        raise ValueError('观察音频不存在或SHA不符')
    import soundfile as sf
    info = sf.info(source)
    if info.channels != 2 or not 0 < info.duration <= 30 or not 8000 <= info.samplerate <= 192000:
        raise ValueError('观察输入限30秒内双声道8至192kHz音频')
    rows = request['observations']
    if not isinstance(rows,list) or not 1 <= len(rows) <= 32:
        raise ValueError('需要1至32个显式观察窗')
    audio, rate = read_audio(source)
    output, ids = [], set()
    for row in rows:
        if set(row) != {'observation_id','start_seconds','end_seconds','f0_candidates_hz'}:
            raise ValueError('观察窗字段缺失或未知')
        key, f0s = row['observation_id'], row['f0_candidates_hz']
        if not isinstance(key,str) or not key or key in ids:
            raise ValueError('观察ID须非空且唯一')
        if not isinstance(f0s,list) or not 1 <= len(f0s) <= 6 or len(set(f0s)) != len(f0s):
            raise ValueError('需要1至6个不重复候选基频；第一项是目标')
        start, end = row['start_seconds'], row['end_seconds']
        if any(type(v) not in (float,int) or not np.isfinite(v) for v in [start,end,*f0s]):
            raise ValueError('时间和候选基频必须为有限数')
        if not 0 <= start < end <= info.duration or end-start < .02-1e-12 or end-start > .5+1e-12:
            raise ValueError('原时间窗须在源音频内且长20至500ms')
        result, _ = project(audio,rate,start,end,f0s)
        output.append({'observation_id':key, **result})
        ids.add(key)
    if sha(source) != request['audio_sha256']:
        raise ValueError('观察期间源音频改变')
    if sha(request_path) != request_sha:
        raise ValueError('观察期间请求JSON改变')
    return {'status':'通过','status_scope':'local_harmonic_observation',
            'request_sha256':request_sha,'source_sha256':sha(source),
            'parameters':PARAMETERS,'observations':output,
            'physical_identity_confirmed':False,'dry_source_recovered':False,
            'note':'联合投影只分已给候选频率；近共频归属、未知音符和效果仍未确定。数值可辨识不等物理源可辨识。'}
