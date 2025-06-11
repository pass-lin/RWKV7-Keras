import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["KERAS_BACKEND"] = "jax"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TRITON_PRINT_AUTOTUNING"] = "-1"
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
    tol = 1e-3
    for i in range(len(output_jax)):
        if output_jax[i] == None or output_torch[i] == None:
            continue
        out_jax = ops.convert_to_numpy(ops.cast(output_jax[i], "float32"))
        out_torch = output_torch[i].float().cpu().numpy()
        flag = np.allclose(out_jax, out_torch, rtol=max(tol, 1e-5), atol=tol)
        if np.sum(out_jax - out_torch) == 0:
            print(f"第{i + 1}个输出结果完全一致")
        else:
            print(f"第{i + 1}个输出函数的校验结果是:{flag}")
        if np.sum(np.isnan(out_jax)):
            print("存在NAN值")
        else:
            print("不存在NAN值")


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
np.random.seed(0)
inputs = [np.random.randn(B, T, H, K) for _ in range(30)]
dht = np.random.randn(B, T, H, T)
d0 = np.random.randn(B, H, T, T)
h = np.random.randn(B, 8, H, T, T)
from ops.jax_op import chunk_dplr_bwd, CHUNKSIZE
from ops.torch_op import chunk_dplr_bwd as torch_chunk_dplr_bwd


jax_inputs = [ops.convert_to_tensor(t, dtype="bfloat16") for t in inputs]
torch_inputs = [torch.from_numpy(t).bfloat16().cuda() for t in inputs]

from ops.jax_op import chunk_dplr_fwd as chunk_dplr

from ops.torch_op import chunk_dplr_fwd as torch_chunk_dplr_fwd


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


a = np.array(-normalize(jax_inputs[3], dim=-1, p=2.0), "float32")
b = np.array(normalize(jax_inputs[3], dim=-1, p=2.0), "float32")
gk = np.array(-ops.exp(-ops.softplus(jax_inputs[5])), "float32")
output_jax = chunk_dplr(
    q=jax_inputs[0],
    k=jax_inputs[1],
    v=jax_inputs[2],
    a=ops.convert_to_tensor(a, jax_inputs[2].dtype),
    b=ops.convert_to_tensor(b, jax_inputs[2].dtype),
    gk=ops.convert_to_tensor(gk, jax_inputs[2].dtype),
    scale=1,
    initial_state=None,
    output_final_state=True,
    chunk_size=CHUNKSIZE,
)

output_torch = torch_chunk_dplr_fwd(
    q=torch_inputs[0],
    k=torch_inputs[1],
    v=torch_inputs[2],
    a=torch.from_numpy(a).bfloat16().cuda(),
    b=torch.from_numpy(b).bfloat16().cuda(),
    gk=torch.from_numpy(gk).bfloat16().cuda(),
    scale=1,
    initial_state=None,
    output_final_state=True,
)

print("校验chunk_dplr_fwd_fwd函数")
test_output(output_jax, output_torch)


output_jax = chunk_dplr_bwd(
    q=jax_inputs[0],
    k=jax_inputs[1],
    v=jax_inputs[2],
    a=ops.convert_to_tensor(a, jax_inputs[2].dtype),
    b=ops.convert_to_tensor(b, jax_inputs[2].dtype),
    gk=ops.convert_to_tensor(gk, jax_inputs[2].dtype),
    scale=1,
    do=ops.convert_to_tensor(d0, dtype=jax_inputs[0].dtype),
    dht=ops.convert_to_tensor(dht, dtype=jax_inputs[0].dtype),
    initial_state=None,
)
output_torch = torch_chunk_dplr_bwd(
    q=torch_inputs[0],
    k=torch_inputs[1],
    v=torch_inputs[2],
    a=torch.from_numpy(a).bfloat16().cuda(),
    b=torch.from_numpy(b).bfloat16().cuda(),
    gk=torch.from_numpy(gk).bfloat16().cuda(),
    scale=1,
    do=torch.from_numpy(d0).to(torch_inputs[0]),
    dht=torch.from_numpy(dht).to(torch_inputs[0]),
    BT=CHUNKSIZE,
    initial_state=None,
    DTYPE=torch.from_numpy(d0).to(torch_inputs[0]).dtype,
)

print("校验chunk_dplr_fwd_bwd函数")
test_output(output_jax, output_torch)
