import keras
from distutils.util import strtobool
import os
from keras import ops
KERNEL_TYPE = os.environ.get("KERNEL_TYPE", "cuda")
USE_KERNEL = False
if keras.config.backend() == "torch":
    import torch

    if torch.cuda.is_available() and not KERNEL_TYPE.lower()=="native":
        from ops.torch_op import generalized_delta_rule

        USE_KERNEL = True
    elif  KERNEL_TYPE.lower()=="triton":
        from ops.native_keras_op import generalized_delta_rule
    else:
        CHUNK_LEN = 16
        from torch.utils.cpp_extension import load
        def generalized_delta_rule(
            r: torch.Tensor,
            w: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            a: torch.Tensor,
            b: torch.Tensor,
            initial_state: torch.Tensor = None,
            output_final_state: bool = True,
            head_first: bool = False,
            use_chunk: bool = True,
        ):
            
            DTYPE = r.dtype

            HEAD_SIZE = ops.shape(r)[-1]
            flags = ['-res-usage', f'-D_C_={HEAD_SIZE}', f"-D_CHUNK_LEN_={CHUNK_LEN}", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization"]
            load(name="wind_backstepping", sources=[f'cuda/wkv7_cuda.cu', 'cuda/wkv7_op.cpp'], is_python_module=False, verbose=True, extra_cuda_cflags=flags)

            class WindBackstepping(torch.autograd.Function):
                @staticmethod
                def forward(ctx, w,q,k,v,z,b):
                    B,T,H,C = w.shape 
                    DTYPE = q.dtype
                    q = ops.cast(q,"bfloat16")
                    k = ops.cast(k,"bfloat16")
                    v = ops.cast(v,"bfloat16")
                    a = ops.cast(a,"bfloat16")
                    b = ops.cast(b,"bfloat16")
                    w = ops.cast(w,"bfloat16")
                    assert T%CHUNK_LEN == 0
                    assert all(i.is_contiguous() for i in [w,q,k,v,z,b])
                    y = torch.empty_like(v)
                    s = torch.empty(B,H,T//CHUNK_LEN,C,C, dtype=torch.float32,device=w.device)
                    sa = torch.empty(B,T,H,C, dtype=torch.float32,device=w.device)
                    torch.ops.wind_backstepping.forward(w,q,k,v,z,b, y,s,sa)
                    ctx.save_for_backward(w,q,k,v,z,b,s,sa)
                    return ops.cast(y,DTYPE)
                @staticmethod
                def backward(ctx, dy):
                    DTYPE = dy.dtype
                    dy = ops.cast(dy,torch.bfloat16)
                    assert all(i.dtype==torch.bfloat16 for i in [dy])
                    assert all(i.is_contiguous() for i in [dy])
                    w,q,k,v,z,b,s,sa = ctx.saved_tensors
                    dw,dq,dk,dv,dz,db = [torch.empty_like(x) for x in [w,q,k,v,z,b]]
                    torch.ops.wind_backstepping.backward(w,q,k,v,z,b, dy,s,sa, dw,dq,dk,dv,dz,db)
                    return  [
                            ops.cast(dw,DTYPE),
                            ops.cast(dq,DTYPE),
                            ops.cast(dk,DTYPE),
                            ops.cast(dv,DTYPE),
                            ops.cast(dz,DTYPE),
                            ops.cast(db,DTYPE)
                            ]

            def RUN_CUDA_RWKV7g(q,w,k,v,a,b):
                B,T,HC = q.shape
                q,w,k,v,a,b = [i.view(B,T,HC//64,64) for i in [q,w,k,v,a,b]]
                return WindBackstepping.apply(w,q,k,v,a,b).view(B,T,HC)
            return RUN_CUDA_RWKV7g(r,w,k,v,a,b)
elif keras.config.backend() == "jax":
    from ops.get_jax_devices_info import is_nvidia
    from jax.lib import xla_bridge
    import jax
    import os

    os.environ["TRITON_LOG_LEVEL"] = "ERROR"  # 只显示错误级别的日志
    os.environ["TRITON_DISABLE_AUTOTUNE"] = "1"  # 禁用自动调优日志
    if is_nvidia and xla_bridge.get_backend().platform == "gpu" and not  KERNEL_TYPE.lower()=="native":
        from ops.jax_op import generalized_delta_rule

        USE_KERNEL = True
    else:
        from ops.native_keras_op import generalized_delta_rule

        generalized_delta_rule = jax.checkpoint(generalized_delta_rule)

else:
    from ops.native_keras_op import generalized_delta_rule
