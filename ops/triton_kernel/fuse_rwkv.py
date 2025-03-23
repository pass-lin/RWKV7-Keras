import triton
import triton.language as tl

from ops.get_devices_info import use_cuda_graph
from ops.triton_kernel.math import exp


@triton.heuristics({"USE_OFFSETS": lambda args: args["offsets"] is not None})
@triton.autotune(
    configs=[
        triton.Config({"BV": BV}, num_warps=num_warps, num_stages=num_stages)
        for BV in [32, 64]
        for num_warps in [2, 4, 8, 16]
        for num_stages in [2, 3, 4]
    ],
    key=["BK"],
    use_cuda_graph=use_cuda_graph,
)
@triton.jit(do_not_specialize=["T"])
def fused_rwkv7_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    w_ptr,
    a_ptr,
    b_ptr,
    state_ptr,
    output_ptr,
    state_output_ptr,
    K: tl.constexpr,
    V: tl.constexpr,
    L,
    H: tl.constexpr,
    offsets,
    scale: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    USE_OFFSETS: tl.constexpr,
    HEAD_FIRST: tl.constexpr,
):
    # indices
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H

    if USE_OFFSETS:
        bos, eos = (
            tl.load(offsets + i_n).to(tl.int64),
            tl.load(offsets + i_n + 1).to(tl.int64),
        )
        L = eos - bos
    else:
        bos, eos = i_n * L, i_n * L + L

    if HEAD_FIRST:
        p_q = q_ptr + i_nh * L * K + tl.arange(0, BK)
        p_k = k_ptr + i_nh * L * K + tl.arange(0, BK)
        p_w = w_ptr + i_nh * L * K + tl.arange(0, BK)
        p_a = a_ptr + i_nh * L * K + tl.arange(0, BK)
        p_b = b_ptr + i_nh * L * K + tl.arange(0, BK)
        p_o = output_ptr + i_nh * L * V + i_v * BV + tl.arange(0, BV)
        p_v = v_ptr + i_nh * L * V + i_v * BV + tl.arange(0, BV)
    else:
        p_q = q_ptr + (bos * H + i_h) * K + tl.arange(0, BK)
        p_k = k_ptr + (bos * H + i_h) * K + tl.arange(0, BK)
        p_w = w_ptr + (bos * H + i_h) * K + tl.arange(0, BK)
        p_a = a_ptr + (bos * H + i_h) * K + tl.arange(0, BK)
        p_b = b_ptr + (bos * H + i_h) * K + tl.arange(0, BK)
        p_v = v_ptr + (bos * H + i_h) * V + i_v * BV + tl.arange(0, BV)
        p_o = output_ptr + (bos * H + i_h) * V + i_v * BV + tl.arange(0, BV)

    mask_k = tl.arange(0, BK) < K
    mask_v = (i_v * BV + tl.arange(0, BV)) < V
    mask_h = mask_k[None, :] & mask_v[:, None]

    b_h = tl.zeros([BV, BK], dtype=tl.float32)

    if USE_INITIAL_STATE:
        p_h0 = (
            state_ptr
            + i_nh * K * V
            + (tl.arange(0, BK)[None, :]) * V
            + ((i_v * BV + tl.arange(0, BV))[:, None])
        )
        b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    for _ in range(0, L):
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_w = tl.load(p_w, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32) * scale
        b_a = tl.load(p_a, mask=mask_k, other=0).to(tl.float32)
        b_b = tl.load(p_b, mask=mask_k, other=0).to(tl.float32)
        # to store
        tmp = tl.sum(b_h * b_a[None, :], axis=1)
        b_h = exp(-exp(b_w))[None, :] * b_h + (
            tmp[:, None] * b_b[None, :] + b_k[None, :] * b_v[:, None]
        )
        _o = b_h * b_q[None, :]
        _o = tl.sum(_o, axis=1)
        tl.store(p_o, _o.to(p_o.dtype.element_ty), mask=mask_v)
        p_q += K if HEAD_FIRST else K * H
        p_k += K if HEAD_FIRST else K * H
        p_w += K if HEAD_FIRST else K * H
        p_o += V if HEAD_FIRST else V * H
        p_v += V if HEAD_FIRST else V * H
        p_a += K if HEAD_FIRST else K * H
        p_b += K if HEAD_FIRST else K * H

    if STORE_FINAL_STATE:
        p_ht = (
            state_output_ptr
            + i_nh * K * V
            + (tl.arange(0, BK)[None, :]) * V
            + ((i_v * BV + tl.arange(0, BV))[:, None])
        )
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)
