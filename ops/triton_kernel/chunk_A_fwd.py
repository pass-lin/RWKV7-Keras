# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang


import triton
import triton.language as tl

from ops.get_torch_devices_info import use_cuda_graph
from ops.triton_kernel.math import exp


@triton.heuristics({"USE_OFFSETS": lambda args: args["offsets"] is not None})
@triton.autotune(
    configs=[
        triton.Config({"BK": BK}, num_warps=num_warps, num_stages=num_stages)
        for BK in [32, 64]
        for num_warps in [2, 4, 8, 16]
        for num_stages in [2, 3, 4]
    ],
    key=["BC", "K"],
    use_cuda_graph=use_cuda_graph,
)
@triton.jit(do_not_specialize=["T"])
def chunk_dplr_fwd_A_kernel_intra_sub_inter(
    q,
    k,
    a,
    b,
    gi,  # cumsum
    ge,  # before cumsum
    Aqk,
    Aqb,
    Aab,
    Aak,
    offsets,
    indices,
    scale: tl.constexpr,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    NC: tl.constexpr,
    USE_OFFSETS: tl.constexpr,
    HEAD_FIRST: tl.constexpr,
):
    i_t, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    i_i, i_j = i_c // NC, i_c % NC
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

    if i_t * BT + i_i * BC >= T:
        return
    if i_i <= i_j:
        return

    b_Aqk = tl.zeros([BC, BC], dtype=tl.float32)
    b_Aqb = tl.zeros([BC, BC], dtype=tl.float32)
    b_Aab = tl.zeros([BC, BC], dtype=tl.float32)
    b_Aak = tl.zeros([BC, BC], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K

        if HEAD_FIRST:
            p_q = tl.make_block_ptr(
                q + i_bh * T * K,
                (T, K),
                (K, 1),
                (i_t * BT + i_i * BC, i_k * BK),
                (BC, BK),
                (1, 0),
            )
            p_a = tl.make_block_ptr(
                a + i_bh * T * K,
                (T, K),
                (K, 1),
                (i_t * BT + i_i * BC, i_k * BK),
                (BC, BK),
                (1, 0),
            )
            p_gq_i = tl.make_block_ptr(
                gi + i_bh * T * K,
                (T, K),
                (K, 1),
                (i_t * BT + i_i * BC, i_k * BK),
                (BC, BK),
                (1, 0),
            )
            p_gq_e = tl.make_block_ptr(
                ge + i_bh * T * K,
                (T, K),
                (K, 1),
                (i_t * BT + i_i * BC, i_k * BK),
                (BC, BK),
                (1, 0),
            )
            p_k = tl.make_block_ptr(
                k + i_bh * T * K,
                (K, T),
                (1, K),
                (i_k * BK, i_t * BT + i_j * BC),
                (BK, BC),
                (0, 1),
            )
            p_b = tl.make_block_ptr(
                b + i_bh * T * K,
                (K, T),
                (1, K),
                (i_k * BK, i_t * BT + i_j * BC),
                (BK, BC),
                (0, 1),
            )
            p_gk = tl.make_block_ptr(
                gi + i_bh * T * K,
                (K, T),
                (1, K),
                (i_k * BK, i_t * BT + i_j * BC),
                (BK, BC),
                (0, 1),
            )
            p_gn = tl.max_contiguous(
                tl.multiple_of(gi + (i_bh * T + i_t * BT + i_i * BC - 1) * K + o_k, BK),
                BK,
            )
        else:
            p_q = tl.make_block_ptr(
                q + (bos * H + i_h) * K,
                (T, K),
                (H * K, 1),
                (i_t * BT + i_i * BC, i_k * BK),
                (BC, BK),
                (1, 0),
            )
            p_a = tl.make_block_ptr(
                a + (bos * H + i_h) * K,
                (T, K),
                (H * K, 1),
                (i_t * BT + i_i * BC, i_k * BK),
                (BC, BK),
                (1, 0),
            )
            p_gq_i = tl.make_block_ptr(
                gi + (bos * H + i_h) * K,
                (T, K),
                (H * K, 1),
                (i_t * BT + i_i * BC, i_k * BK),
                (BC, BK),
                (1, 0),
            )
            p_gq_e = tl.make_block_ptr(
                ge + (bos * H + i_h) * K,
                (T, K),
                (H * K, 1),
                (i_t * BT + i_i * BC, i_k * BK),
                (BC, BK),
                (1, 0),
            )
            p_k = tl.make_block_ptr(
                k + (bos * H + i_h) * K,
                (K, T),
                (1, H * K),
                (i_k * BK, i_t * BT + i_j * BC),
                (BK, BC),
                (0, 1),
            )
            p_b = tl.make_block_ptr(
                b + (bos * H + i_h) * K,
                (K, T),
                (1, H * K),
                (i_k * BK, i_t * BT + i_j * BC),
                (BK, BC),
                (0, 1),
            )
            p_gk = tl.make_block_ptr(
                gi + (bos * H + i_h) * K,
                (K, T),
                (1, H * K),
                (i_k * BK, i_t * BT + i_j * BC),
                (BK, BC),
                (0, 1),
            )
            p_gn = gi + (bos + i_t * BT + i_i * BC - 1) * H * K + i_h * K + o_k
        # [BK,]
        b_gn = tl.load(p_gn, mask=m_k, other=0).to(tl.float32)
        # [BC, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_a = tl.load(p_a, boundary_check=(0, 1))
        b_gq_i = tl.load(p_gq_i, boundary_check=(0, 1))
        b_gq_e = tl.load(p_gq_e, boundary_check=(0, 1))
        b_ag = b_a * exp(b_gq_e - b_gn[None, :])
        b_qg = b_q * exp(b_gq_i - b_gn[None, :]) * scale
        # [BK, BC]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_b = tl.load(p_b, boundary_check=(0, 1))
        b_gk = tl.load(p_gk, boundary_check=(0, 1)).to(tl.float32)
        tmp = exp(b_gn[:, None] - b_gk)
        b_kg = b_k * tmp
        b_bg = b_b * tmp
        # [BC, BC] using tf32 to improve precision here.
        b_Aab += tl.dot(b_ag, b_bg)
        b_Aak += tl.dot(b_ag, b_kg)
        b_Aqk += tl.dot(b_qg, b_kg)
        b_Aqb += tl.dot(b_qg, b_bg)

    if HEAD_FIRST:
        p_Aqk = tl.make_block_ptr(
            Aqk + i_bh * T * BT,
            (T, BT),
            (BT, 1),
            (i_t * BT + i_i * BC, i_j * BC),
            (BC, BC),
            (1, 0),
        )
        p_Aqb = tl.make_block_ptr(
            Aqb + i_bh * T * BT,
            (T, BT),
            (BT, 1),
            (i_t * BT + i_i * BC, i_j * BC),
            (BC, BC),
            (1, 0),
        )
        p_Aab = tl.make_block_ptr(
            Aab + i_bh * T * BT,
            (T, BT),
            (BT, 1),
            (i_t * BT + i_i * BC, i_j * BC),
            (BC, BC),
            (1, 0),
        )
        p_Aak = tl.make_block_ptr(
            Aak + i_bh * T * BT,
            (T, BT),
            (BT, 1),
            (i_t * BT + i_i * BC, i_j * BC),
            (BC, BC),
            (1, 0),
        )
    else:
        p_Aqk = tl.make_block_ptr(
            Aqk + (bos * H + i_h) * BT,
            (T, BT),
            (H * BT, 1),
            (i_t * BT + i_i * BC, i_j * BC),
            (BC, BC),
            (1, 0),
        )
        p_Aqb = tl.make_block_ptr(
            Aqb + (bos * H + i_h) * BT,
            (T, BT),
            (H * BT, 1),
            (i_t * BT + i_i * BC, i_j * BC),
            (BC, BC),
            (1, 0),
        )
        p_Aab = tl.make_block_ptr(
            Aab + (bos * H + i_h) * BT,
            (T, BT),
            (H * BT, 1),
            (i_t * BT + i_i * BC, i_j * BC),
            (BC, BC),
            (1, 0),
        )
        p_Aak = tl.make_block_ptr(
            Aak + (bos * H + i_h) * BT,
            (T, BT),
            (H * BT, 1),
            (i_t * BT + i_i * BC, i_j * BC),
            (BC, BC),
            (1, 0),
        )
    tl.store(
        p_Aqk,
        b_Aqk.to(Aqk.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )
    tl.store(
        p_Aqb,
        b_Aqb.to(Aqb.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )
    tl.store(
        p_Aab,
        b_Aab.to(Aab.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )
    tl.store(
        p_Aak,
        b_Aak.to(Aak.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )


@triton.heuristics({"USE_OFFSETS": lambda args: args["offsets"] is not None})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4, 8, 16, 32]
        for num_stages in [2, 3, 4]
    ],
    key=["BK", "BT"],
    use_cuda_graph=use_cuda_graph,
)
@triton.jit(do_not_specialize=["T"])
def chunk_dplr_fwd_A_kernel_intra_sub_intra(
    q,
    k,
    a,
    b,
    gi,
    ge,
    Aqk,
    Aqb,
    Aab,
    Aak,
    qg,
    kg,
    ag,
    bg,
    offsets,
    indices,
    scale: tl.constexpr,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    NC: tl.constexpr,
    USE_OFFSETS: tl.constexpr,
    HEAD_FIRST: tl.constexpr,
):
    i_t, i_i, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    i_j = i_i
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

    if i_t * BT + i_i * BC >= T:
        return

    o_i = tl.arange(0, BC)
    o_k = tl.arange(0, BK)
    m_k = o_k < K
    m_A = (i_t * BT + i_i * BC + tl.arange(0, BC)) < T
    last_idx = min((i_t + 1) * BT, T) - 1
    if HEAD_FIRST:
        o_A = i_bh * T * BT + (i_t * BT + i_i * BC + tl.arange(0, BC)) * BT + i_j * BC
        p_q = tl.make_block_ptr(
            q + i_bh * T * K,
            (T, K),
            (K, 1),
            (i_t * BT + i_i * BC, 0),
            (BC, BK),
            (1, 0),
        )
        p_k = tl.make_block_ptr(
            k + i_bh * T * K,
            (T, K),
            (K, 1),
            (i_t * BT + i_i * BC, 0),
            (BC, BK),
            (1, 0),
        )
        p_a = tl.make_block_ptr(
            a + i_bh * T * K,
            (T, K),
            (K, 1),
            (i_t * BT + i_i * BC, 0),
            (BC, BK),
            (1, 0),
        )
        p_b = tl.make_block_ptr(
            b + i_bh * T * K,
            (T, K),
            (K, 1),
            (i_t * BT + i_i * BC, 0),
            (BC, BK),
            (1, 0),
        )
        p_gi = tl.make_block_ptr(
            gi + i_bh * T * K,
            (T, K),
            (K, 1),
            (i_t * BT + i_i * BC, 0),
            (BC, BK),
            (1, 0),
        )
        p_ge = tl.make_block_ptr(
            ge + i_bh * T * K,
            (T, K),
            (K, 1),
            (i_t * BT + i_i * BC, 0),
            (BC, BK),
            (1, 0),
        )
        p_g_last = gi + i_bh * T * K + last_idx * K + tl.arange(0, BK)
        b_g_last = tl.load(p_g_last, mask=m_k, other=0)

        p_qg = tl.make_block_ptr(
            qg + i_bh * T * K,
            (T, K),
            (K, 1),
            (i_t * BT + i_i * BC, 0),
            (BC, BK),
            (1, 0),
        )
        p_kg = tl.make_block_ptr(
            kg + i_bh * T * K,
            (T, K),
            (K, 1),
            (i_t * BT + i_i * BC, 0),
            (BC, BK),
            (1, 0),
        )
        p_ag = tl.make_block_ptr(
            ag + i_bh * T * K,
            (T, K),
            (K, 1),
            (i_t * BT + i_i * BC, 0),
            (BC, BK),
            (1, 0),
        )
        p_bg = tl.make_block_ptr(
            bg + i_bh * T * K,
            (T, K),
            (K, 1),
            (i_t * BT + i_i * BC, 0),
            (BC, BK),
            (1, 0),
        )
    else:
        o_A = (
            (bos + i_t * BT + i_i * BC + tl.arange(0, BC)) * H * BT
            + i_h * BT
            + i_j * BC
        )
        p_q = tl.make_block_ptr(
            q + (bos * H + i_h) * K,
            (T, K),
            (H * K, 1),
            (i_t * BT + i_i * BC, 0),
            (BC, BK),
            (1, 0),
        )
        p_k = tl.make_block_ptr(
            k + (bos * H + i_h) * K,
            (T, K),
            (H * K, 1),
            (i_t * BT + i_i * BC, 0),
            (BC, BK),
            (1, 0),
        )
        p_a = tl.make_block_ptr(
            a + (bos * H + i_h) * K,
            (T, K),
            (H * K, 1),
            (i_t * BT + i_i * BC, 0),
            (BC, BK),
            (1, 0),
        )
        p_b = tl.make_block_ptr(
            b + (bos * H + i_h) * K,
            (T, K),
            (H * K, 1),
            (i_t * BT + i_i * BC, 0),
            (BC, BK),
            (1, 0),
        )
        p_gi = tl.make_block_ptr(
            gi + (bos * H + i_h) * K,
            (T, K),
            (H * K, 1),
            (i_t * BT + i_i * BC, 0),
            (BC, BK),
            (1, 0),
        )
        p_ge = tl.make_block_ptr(
            ge + (bos * H + i_h) * K,
            (T, K),
            (H * K, 1),
            (i_t * BT + i_i * BC, 0),
            (BC, BK),
            (1, 0),
        )
        p_g_last = gi + (bos * H + i_h) * K + last_idx * H * K + tl.arange(0, BK)
        b_g_last = tl.load(p_g_last, mask=m_k, other=0)
        p_qg = tl.make_block_ptr(
            qg + (bos * H + i_h) * K,
            (T, K),
            (H * K, 1),
            (i_t * BT + i_i * BC, 0),
            (BC, BK),
            (1, 0),
        )
        p_kg = tl.make_block_ptr(
            kg + (bos * H + i_h) * K,
            (T, K),
            (H * K, 1),
            (i_t * BT + i_i * BC, 0),
            (BC, BK),
            (1, 0),
        )
        p_ag = tl.make_block_ptr(
            ag + (bos * H + i_h) * K,
            (T, K),
            (H * K, 1),
            (i_t * BT + i_i * BC, 0),
            (BC, BK),
            (1, 0),
        )
        p_bg = tl.make_block_ptr(
            bg + (bos * H + i_h) * K,
            (T, K),
            (H * K, 1),
            (i_t * BT + i_i * BC, 0),
            (BC, BK),
            (1, 0),
        )

    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_q = b_q * scale
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_a = tl.load(p_a, boundary_check=(0, 1))
    b_b = tl.load(p_b, boundary_check=(0, 1))
    b_gi = tl.load(p_gi, boundary_check=(0, 1)).to(tl.float32)
    b_ge = tl.load(p_ge, boundary_check=(0, 1)).to(tl.float32)

    # deal with decay term.
    g_exp = exp(b_gi)
    g_exp_inv = exp(-b_gi + b_g_last[None, :])
    b_qg = b_q * g_exp
    b_kg = b_k * g_exp_inv
    b_bg = b_b * g_exp_inv
    b_ag = b_a * exp(b_ge)
    tl.store(
        p_qg,
        b_qg.to(p_qg.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )
    tl.store(
        p_bg,
        b_bg.to(p_bg.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )
    tl.store(
        p_ag,
        b_ag.to(p_ag.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )
    tl.store(
        p_kg,
        b_kg.to(p_kg.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )
    b_qg, b_kg, b_ag, b_bg = None, None, None, None
    tl.debug_barrier()

    b_q = b_q.to(b_k.dtype)
    # inner attn
    for j in range(0, min(BC, T - i_t * BT - i_i * BC)):
        # a trick to index the j-th row of b_k, b_g, b_b
        mask = tl.arange(0, BC) == j
        b_k_j = tl.sum(tl.where(mask[:, None], b_k, 0), 0)
        b_gk_j = tl.sum(tl.where(mask[:, None], b_gi, 0), 0)
        b_b_j = tl.sum(tl.where(mask[:, None], b_b, 0), 0)
        tmp = exp(b_gi - b_gk_j[None, :])
        b_A_qk = tl.sum(b_q * b_k_j[None, :] * tmp, 1)
        b_A_qk = tl.where(o_i >= j, b_A_qk, 0.0)
        b_A_qb = tl.sum(b_q * b_b_j[None] * tmp, 1)
        b_A_qb = tl.where(o_i >= j, b_A_qb, 0.0)
        tmp2 = exp(b_ge - b_gk_j[None, :])
        b_A_ak = tl.sum(b_a * b_k_j[None, :] * tmp2, 1)
        b_A_ak = tl.where(o_i > j, b_A_ak, 0.0)
        b_A_ab = tl.sum(b_a * b_b_j[None, :] * tmp2, 1)
        b_A_ab = tl.where(o_i > j, b_A_ab, 0.0)
        tl.store(
            Aqk + o_A + j,
            b_A_qk.to(dtype=Aqk.dtype.element_ty, fp_downcast_rounding="rtne"),
            mask=m_A,
        )
        tl.store(
            Aqb + o_A + j,
            b_A_qb.to(dtype=Aqb.dtype.element_ty, fp_downcast_rounding="rtne"),
            mask=m_A,
        )
        tl.store(
            Aab + o_A + j,
            b_A_ab.to(dtype=Aqb.dtype.element_ty, fp_downcast_rounding="rtne"),
            mask=m_A,
        )
        tl.store(
            Aak + o_A + j,
            b_A_ak.to(dtype=Aqk.dtype.element_ty, fp_downcast_rounding="rtne"),
            mask=m_A,
        )
