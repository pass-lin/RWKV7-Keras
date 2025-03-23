# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import triton
import triton.language as tl

from ops.get_devices_info import use_cuda_graph
from ops.triton_kernel.math import exp


@triton.heuristics(
    {
        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
        "STORE_FINAL_STATE": lambda args: args["ht"] is not None,
        "USE_OFFSETS": lambda args: args["offsets"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config({"BV": BV}, num_warps=num_warps, num_stages=num_stages)
        for BV in [16, 32, 64]
        for num_warps in [2, 4, 8, 16]
        for num_stages in [2, 3, 4]
    ],
    key=["BK"],
    use_cuda_graph=use_cuda_graph,
)
@triton.jit(do_not_specialize=["T"])
def fused_recurrent_dplr_delta_rule_fwd_kernel(
    q,
    k,
    v,
    a,
    b,
    gk,
    o,
    h0,
    ht,
    offsets,
    scale,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    REVERSE: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    USE_OFFSETS: tl.constexpr,
    HEAD_FIRST: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    i_n, i_h = i_nh // H, i_nh % H

    if USE_OFFSETS:
        bos, eos = (
            tl.load(offsets + i_n).to(tl.int64),
            tl.load(offsets + i_n + 1).to(tl.int64),
        )
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    if HEAD_FIRST:
        p_q = q + i_nh * T * K + ((T - 1) * K if REVERSE else 0) + o_k
        p_k = k + i_nh * T * K + ((T - 1) * K if REVERSE else 0) + o_k
        p_a = a + i_nh * T * K + ((T - 1) * K if REVERSE else 0) + o_k
        p_b = b + i_nh * T * K + ((T - 1) * K if REVERSE else 0) + o_k
        p_gk = gk + i_nh * T * K + ((T - 1) * K if REVERSE else 0) + o_k
        p_v = v + i_nh * T * V + ((T - 1) * V if REVERSE else 0) + o_v
        p_o = o + i_nh * T * V + ((T - 1) * V if REVERSE else 0) + o_v

    else:
        p_q = q + (bos + ((T - 1) if REVERSE else 0)) * H * K + i_h * K + o_k
        p_k = k + (bos + ((T - 1) if REVERSE else 0)) * H * K + i_h * K + o_k
        p_a = a + (bos + ((T - 1) if REVERSE else 0)) * H * K + i_h * K + o_k
        p_b = b + (bos + ((T - 1) if REVERSE else 0)) * H * K + i_h * K + o_k
        p_gk = gk + (bos + ((T - 1) if REVERSE else 0)) * H * K + i_h * K + o_k
        p_v = v + (bos + ((T - 1) if REVERSE else 0)) * H * V + i_h * V + o_v
        p_o = o + (bos + ((T - 1) if REVERSE else 0)) * H * V + i_h * V + o_v

    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[None, :] & mask_v[:, None]
    b_h = tl.zeros([BV, BK], dtype=tl.float32)

    if USE_INITIAL_STATE:
        p_h0 = h0 + i_nh * K * V + o_k[None, :] * V + o_v[:, None]
        b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    for _ in range(0, T):
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32) * scale
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_a = tl.load(p_a, mask=mask_k, other=0).to(tl.float32)
        b_b = tl.load(p_b, mask=mask_k, other=0).to(tl.float32)
        b_gk = tl.load(p_gk, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)

        tmp = tl.sum(b_h * b_a[None, :], axis=1)
        b_h = exp(b_gk)[None, :] * b_h + (
            tmp[:, None] * b_b[None, :] + b_k[None, :] * b_v[:, None]
        )
        b_o = tl.sum(b_h * b_q[None, :], axis=1)

        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)
        p_q += (-1 if REVERSE else 1) * (1 if HEAD_FIRST else H) * K
        p_k += (-1 if REVERSE else 1) * (1 if HEAD_FIRST else H) * K
        p_a += (-1 if REVERSE else 1) * (1 if HEAD_FIRST else H) * K
        p_b += (-1 if REVERSE else 1) * (1 if HEAD_FIRST else H) * K
        p_gk += (-1 if REVERSE else 1) * (1 if HEAD_FIRST else H) * K
        p_v += (-1 if REVERSE else 1) * (1 if HEAD_FIRST else H) * V
        p_o += (-1 if REVERSE else 1) * (1 if HEAD_FIRST else H) * V

    if STORE_FINAL_STATE:
        p_ht = ht + i_nh * K * V + o_k[None, :] * V + o_v[:, None]
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)
