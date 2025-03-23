# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import triton
import triton.language as tl


@triton.heuristics({"USE_OFFSETS": lambda args: args["offsets"] is not None})
@triton.autotune(
    configs=[
        triton.Config({"BS": BS}, num_warps=num_warps, num_stages=num_stages)
        for BS in [16, 32, 64]
        for num_warps in [4, 8, 16, 32]
        for num_stages in [2, 3, 4]
    ],
    key=["S", "BT"],
    use_cuda_graph=0,
)
@triton.jit(do_not_specialize=["T"])
def chunk_rwkv6_fwd_cumsum_kernel(
    s,
    oi,
    oe,
    offsets,
    indices,
    T,
    H: tl.constexpr,
    S: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    HEAD_FIRST: tl.constexpr,
    USE_OFFSETS: tl.constexpr,
):
    i_s, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    if USE_OFFSETS:
        i_n, i_t = (
            tl.load(indices + i_t * 2).to(tl.int32),
            tl.load(indices + i_t * 2 + 1).to(tl.int32),
        )
        bos, eos = (
            tl.load(offsets + i_n).to(tl.int32),
            tl.load(offsets + i_n + 1).to(tl.int32),
        )
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    o_i = tl.arange(0, BT)
    m_i = tl.where(o_i[:, None] >= o_i[None, :], 1.0, 0.0).to(tl.float32)
    m_e = tl.where(o_i[:, None] > o_i[None, :], 1.0, 0.0).to(tl.float32)

    if HEAD_FIRST:
        p_s = tl.make_block_ptr(
            s + i_bh * T * S,
            (T, S),
            (S, 1),
            (i_t * BT, i_s * BS),
            (BT, BS),
            (1, 0),
        )
        p_oi = tl.make_block_ptr(
            oi + i_bh * T * S,
            (T, S),
            (S, 1),
            (i_t * BT, i_s * BS),
            (BT, BS),
            (1, 0),
        )
        p_oe = tl.make_block_ptr(
            oe + i_bh * T * S,
            (T, S),
            (S, 1),
            (i_t * BT, i_s * BS),
            (BT, BS),
            (1, 0),
        )
    else:
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
