"""可选ACE-Step固定版本的自动掩码兼容补丁；不导入或改写上游。"""
from contextlib import contextmanager
import functools
import hashlib
import inspect
from pathlib import Path

SUPPORTED_SOURCE_SHA256 = '5d9c1fdb249931794af76db83166347be0e8a081f958338c9a7963131e87a365'
METHOD = '_build_chunk_masks_and_src_latents'


@contextmanager
def corrected_auto_mask(handler):
    """仅对显式启用的实例修正已知版本；退出时恢复原属性。"""
    original = getattr(handler, METHOD)
    if not hasattr(original, '__func__'):
        raise ValueError('ACE掩码实例方法已被覆盖；拒绝嵌套或未知补丁')
    source = inspect.getsourcefile(original.__func__)
    if source is None or hashlib.sha256(Path(source).read_bytes()).hexdigest() != SUPPORTED_SOURCE_SHA256:
        raise ValueError('ACE掩码实现版本不匹配；拒绝静默应用补丁')
    signature = inspect.signature(original)
    if 'chunk_mask_modes' not in signature.parameters:
        raise ValueError('ACE掩码接口缺少模式参数')
    previous = handler.__dict__.get(METHOD)
    existed = METHOD in handler.__dict__

    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        modes = bound.arguments['chunk_mask_modes']
        output = original(*args, **kwargs)
        if not modes or not any(mode == 'auto' for mode in modes):
            return output
        if not isinstance(output, tuple) or len(output) != 5 or output[0].shape[0] != len(modes):
            raise ValueError('ACE掩码返回结构不匹配')
        mask = output[0].float().clone()
        for row, mode in enumerate(modes):
            if mode == 'auto':
                mask[row] = 2.0
        return (mask, *output[1:])

    setattr(handler, METHOD, wrapped)
    try:
        yield
    finally:
        if existed:
            setattr(handler, METHOD, previous)
        else:
            delattr(handler, METHOD)
