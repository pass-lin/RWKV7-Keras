import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["KERAS_BACKEND"] = "jax"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TRITON_PRINT_AUTOTUNING"] = "0"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["JAX_LOG_COMPILE"] = "False"
import logging

# 设置日志级别为ERROR，屏蔽INFO和WARNING
logging.basicConfig(level=logging.ERROR)
import numpy as np
from keras import ops
import torch
import torch.nn.functional as F


def test_output(output_jax, output_torch):
    for i in range(len(output_jax)):
        if output_jax[i] == None and output_torch[i] == None:
            continue
        out_jax = ops.convert_to_numpy(ops.cast(output_jax[i], "float32"))
        out_torch = output_torch[i].float().cpu().numpy()
        flag = np.allclose(out_jax, out_torch, rtol=1e-2, atol=0.03)
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
from ops.jax_kernel.chunk import chunk_dplr_bwd,CHUNKSIZE
from ops.torch_kernel.chunk import chunk_dplr_bwd as torch_chunk_dplr_bwd
inputs = [np.random.randn(B, T, H, K) for _ in range(30)]
jax_inputs = [ops.convert_to_tensor(t, dtype="float32") for t in inputs]
torch_inputs = [torch.from_numpy(t).float().cuda() for t in inputs]
output_jax = chunk_dplr_bwd(
    q=jax_inputs[0],
    k=jax_inputs[1],
    v=jax_inputs[2],
    a=-normalize(jax_inputs[3]),
    b=normalize(jax_inputs[3]),
    gk=-ops.exp(-ops.softplus(jax_inputs[5])),
    scale=1,
    do = ops.convert_to_tensor(d0, dtype=jax_inputs[0].dtype),
    dht=ops.convert_to_tensor(dht, dtype=jax_inputs[0].dtype),
    initial_state = None,
    head_first=False,
)
output_torch = torch_chunk_dplr_bwd(
    q=torch_inputs[0],
    k=torch_inputs[1],
    v=torch_inputs[2],
    a=-F.normalize(torch_inputs[3], dim=-1, p=2.0),
    b=F.normalize(torch_inputs[3], dim=-1, p=2.0),
    gk=-torch.exp(-F.softplus(torch_inputs[5])),
    scale=1,
    do = torch.from_numpy(d0).to(torch_inputs[0]),
    dht= torch.from_numpy(dht).to(torch_inputs[0]),
    BT=CHUNKSIZE,
    initial_state = None,
    head_first=False,
)

print("校验chunk_dplr_fwd_bwd函数")
test_output(output_jax, output_torch)

jax_inputs = [ops.convert_to_tensor(t, dtype="bfloat16") for t in inputs]
torch_inputs = [torch.from_numpy(t).bfloat16().cuda() for t in inputs]

from ops.jax_op import chunk_dplr


output_jax = chunk_dplr(
    r=jax_inputs[0],
    k=jax_inputs[1],
    v=jax_inputs[2],
    a=-normalize(jax_inputs[3]),
    b=normalize(jax_inputs[3]),
    gk=-ops.exp(-ops.softplus(jax_inputs[5])),
    scale=1,
    initial_state=None,
    output_final_state=True,
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
    gk=-torch.exp(-F.softplus(torch_inputs[5])),
    scale=1,
    initial_state=None,
    output_final_state=True,
    head_first=False,
)

print("校验chunk_dplr_fwd_fwd函数")
test_output(output_jax, output_torch)

from src.layer import GroupNorm

ln_jax = GroupNorm(groups=H, epsilon=64e-5, dtype=output_jax[0].dtype)
ln_jax_out = ln_jax(ops.reshape(output_jax[0], (B * T, -1)))
if output_torch[0].type == torch.bfloat16:
    ln_torch = torch.nn.GroupNorm(H, K * H, eps=64e-5).cuda().bfloat16()
else:
    ln_torch = torch.nn.GroupNorm(H, K * H, eps=64e-5).cuda()
ln_torch_out = ln_torch(output_torch[0].view(B * T, -1).float())

ln_jax_out = ops.convert_to_numpy(ops.cast(ln_jax_out, "float32"))
ln_torch_out = ln_torch_out.float().detach().cpu().numpy()
flag = np.allclose(ln_jax_out, ln_torch_out, atol=0.03, rtol=1e-2)
print(f"ln函数的校验结果是:{flag}")


import jax
grad = jax.grad(chunk_dplr)(
    r=jax_inputs[0],
    k=jax_inputs[1],
    v=jax_inputs[2],
    a=-normalize(jax_inputs[3]),
    b=normalize(jax_inputs[3]),
    gk=-ops.exp(-ops.softplus(jax_inputs[5])),
    scale=1,
    initial_state=None,
    output_final_state=True,
    head_first=False,
)