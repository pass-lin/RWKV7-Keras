import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["KERAS_BACKEND"] = "jax"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TRITON_PRINT_AUTOTUNING"] = "-1"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["JAX_LOG_COMPILE"] = "False"


# use kernel by torch
import torch
import jax.numpy as jnp

"""


@triton.autotune(
    configs=[
        triton.Config({"BS": BS}, num_warps=num_warps, num_stages=num_stages)
        for BS in [16, 32, 64]
        for num_warps in [4, 8, 16]
        for num_stages in [2, 3, 4]
    ],
    key=["S", "BT"],
    use_cuda_graph=use_cuda_graph,
)
@triton.jit(do_not_specialize=["T"])
def chunk_rwkv6_fwd_cumsum_kernel(
    s,
    T,
    oi,
    oe,
    H: tl.constexpr,
    S: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
):
    cu_seqlens = None
    chunk_indices = None
    i_s, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    if False:
        i_n, i_t = (
            tl.load(chunk_indices + i_t * 2).to(tl.int32),
            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32),
        )
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    o_i = tl.arange(0, BT)
    m_i = tl.where(o_i[:, None] >= o_i[None, :], 1.0, 0.0).to(tl.float32)
    m_e = tl.where(o_i[:, None] > o_i[None, :], 1.0, 0.0).to(tl.float32)

    p_s = tl.make_block_ptr(
        s + (bos * H + i_h) * S,
        (T, S),
        (H * S, 1),
        (i_t * BT, i_s * BS),
        (BT, BS),
        (1, 0),
    )
    p_oi = tl.make_block_ptr(
        oi + (bos * H + i_h) * S,
        (T, S),
        (H * S, 1),
        (i_t * BT, i_s * BS),
        (BT, BS),
        (1, 0),
    )
    p_oe = tl.make_block_ptr(
        oe + (bos * H + i_h) * S,
        (T, S),
        (H * S, 1),
        (i_t * BT, i_s * BS),
        (BT, BS),
        (1, 0),
    )
    # [BT, BS]
    b_s = tl.load(p_s, boundary_check=(0, 1)).to(tl.float32)
    b_oi = tl.dot(m_i, b_s)
    b_oe = tl.dot(m_e, b_s)
    tl.store(
        p_oi,
        b_oi.to(p_oi.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )
    tl.store(
        p_oe,
        b_oe.to(p_oe.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )


#use kernel by jax

def chunk_rwkv6_fwd_cumsum_jax(
    g: jax.Array,
    chunk_size: int = 16,
) -> jax.Array:
    B, T, H, S = g.shape
    BT = chunk_size
    NT = triton.cdiv(T, BT) 

    out_shapes = [
        jax.ShapeDtypeStruct(g.shape, "float32"),
        jax.ShapeDtypeStruct(g.shape, "float32"),
    ]

    def grid(meta):
        return (triton.cdiv(meta["S"], meta["BS"]), NT, B * H)

    gi, ge = jt.triton_call(
        g,
        T,
        H=H,
        S=S,
        BT=BT,
        grid=grid,
        kernel=chunk_rwkv6_fwd_cumsum_kernel,
        out_shape=out_shapes,
    )

    return gi, ge




def chunk_rwkv6_fwd_cumsum_torch(
    g: torch.Tensor,
    chunk_size: int = 16,
) -> torch.Tensor:
    B, T, H, S = g.shape
    BT = chunk_size
    NT = triton.cdiv(T, BT)

    gi, ge = (
        torch.empty_like(g, dtype=torch.float),
        torch.empty_like(g, dtype=torch.float),
    )

    def grid(meta):
        return (triton.cdiv(meta["S"], meta["BS"]), NT, B * H)

    # keep cummulative normalizer in fp32
    chunk_rwkv6_fwd_cumsum_kernel[grid](
        g,
        T,
        gi,
        ge,
        H=H,
        S=S,
        BT=BT,
    )
    return gi, ge
"""
from ops.jax_op import chunk_rwkv6_fwd_cumsum as chunk_rwkv6_fwd_cumsum_jax
from ops.torch_op import chunk_rwkv6_fwd_cumsum as chunk_rwkv6_fwd_cumsum_torch
import numpy as np

T = 128
B = 1
H = 6
K = 128
x = np.random.randn(B, T, H, K)
from keras import ops

x = -ops.exp(-ops.softplus(x))
x = ops.convert_to_numpy(x)


def test_output(output_jax, output_torch):
    tol = 5e-3
    for i in range(len(output_jax)):
        if output_jax[i] == None and output_torch[i] == None:
            continue
        out_jax = np.array(jnp.astype(output_jax[i], "float32"))
        out_torch = output_torch[i].float().cpu().numpy()
        flag = np.allclose(out_jax, out_torch, rtol=max(tol, 1e-5), atol=tol)

        print(f"The verification result of the {i + 1}th output function is: {flag}")


output_jax = chunk_rwkv6_fwd_cumsum_jax(jnp.array(x, "bfloat16"), 16)
output_torch = chunk_rwkv6_fwd_cumsum_torch(torch.from_numpy(x).bfloat16().cuda(), 16)
test_output(output_jax, output_torch)
