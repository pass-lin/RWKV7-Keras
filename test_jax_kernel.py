import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["KERAS_BACKEND"] = "jax"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TRITON_PRINT_AUTOTUNING"] = "0"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"
os.environ["JAX_LOG_COMPILE"] = "False"
import logging

# 设置日志级别为ERROR，屏蔽INFO和WARNING
logging.basicConfig(level=logging.ERROR)
import numpy as np
from keras import ops
import torch
import keras
import torch.nn.functional as F

def test_output(output_jax, output_torch):
    for i in range(len(output_jax)):
        if output_jax[i] == None and output_torch[i] == None:
            print(f"第{i + 1}个输出函数的校验结果是:{flag}")
            continue
        out_jax = ops.convert_to_numpy(output_jax[i])
        out_torch = output_torch[i].cpu().numpy()
        flag = np.allclose(out_jax, out_torch, atol=1e-3, rtol=1e-2)
        print(f"第{i + 1}个输出函数的校验结果是:{flag}")

def normalize(
    z,
    p=2,
    dim=-1,
    eps: float = 1e-12,
):
    # F.normalize like api
    denom = ops.norm(z, ord=p, axis=dim, keepdims=True)
    denom = ops.maximum(denom, 1e-12)
    return z / denom
T = 128
B = 1
H = 6
K = 128
inputs = [np.random.randn(B, T, H, K) for _ in range(30)]
dht = np.random.randn(B, T, H, T)
d0 = np.random.randn(B, H, T, T)
h = np.random.randn(B, 8, H, T, T)
from ops.jax_kernel.chunk import chunk_dplr_fwd
inputs = [np.random.randn(B, T, H, K) for _ in range(30)]
jax_inputs = [ops.convert_to_tensor(t, dtype="float32") for t in inputs]
torch_inputs = [torch.from_numpy(t).float().cuda() for t in inputs]
output_jax = chunk_dplr_fwd(
    q=jax_inputs[0],
    k=jax_inputs[1],
    v=jax_inputs[2],
    a=-normalize(jax_inputs[3]),
    b=normalize(jax_inputs[3]),
    gk=-ops.exp( -ops.softplus(jax_inputs[5])),
    scale=1,
    initial_state=None,
    output_final_state=True,
    chunk_size=32,
    head_first=False,
)

from ops.torch_kernel.chunk import chunk_dplr_fwd as torch_chunk_dplr_fwd
def normalize(
    z,
    p=2,
    dim=-1,
    eps: float = 1e-12,
):
    # F.normalize like api
    denom = ops.norm(z, ord=p, axis=dim, keepdims=True)
    denom = ops.maximum(denom, 1e-12)
    return z / denom
output_torch = torch_chunk_dplr_fwd(
    q=torch_inputs[0],
    k=torch_inputs[1],
    v=torch_inputs[2],
    a=-F.normalize(torch_inputs[3], dim=-1, p=2.0),
    b=F.normalize(torch_inputs[3], dim=-1, p=2.0),
    gk=-torch.exp( -F.softplus(torch_inputs[5])),
    scale=1,
    initial_state=None,
    output_final_state=True,
    chunk_size=32,
    head_first=False,
)

print("校验chunk_dplr_fwd函数")
test_output(output_jax, output_torch)