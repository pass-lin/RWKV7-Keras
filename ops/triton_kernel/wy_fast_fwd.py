# -*- coding: utf-8 -*-
# Copyright (c) 2024, Songlin Yang, Yu Zhang


import triton
import triton.language as tl

from ops.get_devices_info import use_cuda_graph


@triton.heuristics({"USE_OFFSETS": lambda args: args["offsets"] is not None})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps) for num_warps in [1, 2, 4, 8, 16]
    ],
    key=["BT"],
    use_cuda_graph=use_cuda_graph,
)
@triton.jit(do_not_specialize=["T"])
def fwd_prepare_wy_repr_kernel_chunk32(
    A_ab,
    A_ab_inv,
    offsets,
    indices,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,  # placeholder, do not delete
    USE_OFFSETS: tl.constexpr,
    HEAD_FIRST: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
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
    if HEAD_FIRST:
        p_Aab = tl.make_block_ptr(
            A_ab + i_bh * T * BT,
            (T, BT),
            (BT, 1),
            (i_t * BT, 0),
            (BT, BT),
            (1, 0),
        )
        p_Aab_inv = tl.make_block_ptr(
            A_ab_inv + i_bh * T * BT,
            (T, BT),
            (BT, 1),
            (i_t * BT, 0),
            (BT, BT),
            (1, 0),
        )
    else:
        p_Aab = tl.make_block_ptr(
            A_ab + (bos * H + i_h) * BT,
            (T, BT),
            (H * BT, 1),
            (i_t * BT, 0),
            (BT, BT),
            (1, 0),
        )
        p_Aab_inv = tl.make_block_ptr(
            A_ab_inv + (bos * H + i_h) * BT,
            (T, BT),
            (H * BT, 1),
            (i_t * BT, 0),
            (BT, BT),
            (1, 0),
        )
    b_A_ab = tl.load(p_Aab, boundary_check=(0, 1))
    b_A_ab = tl.where(
        tl.arange(0, BT)[:, None] > tl.arange(0, BT)[None, :], b_A_ab, 0
    )
    for i in range(1, BT):
        mask = tl.arange(0, BT) == i
        b_a = tl.sum(tl.where(mask[:, None], b_A_ab, 0), 0)
        b_a = b_a + tl.sum(b_a[:, None] * b_A_ab, 0) * (tl.arange(0, BT) < i)
        b_A_ab = tl.where(mask[:, None], b_a, b_A_ab)
    b_A_ab += tl.arange(0, BT)[:, None] == tl.arange(0, BT)[None, :]
    tl.store(
        p_Aab_inv, b_A_ab.to(p_Aab_inv.dtype.element_ty), boundary_check=(0, 1)
    )


@triton.heuristics({"USE_OFFSETS": lambda args: args["offsets"] is not None})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=["BC"],
    use_cuda_graph=use_cuda_graph,
)
@triton.jit(do_not_specialize=["T"])
def fwd_prepare_wy_repr_kernel_chunk64(
    A_ab,
    A_ab_inv,
    offsets,
    indices,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    USE_OFFSETS: tl.constexpr,
    HEAD_FIRST: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
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

    if HEAD_FIRST:
        p_A1 = tl.make_block_ptr(
            A_ab + i_bh * T * BT,
            (T, BT),
            (BT, 1),
            (i_t * BT, 0),
            (BC, BC),
            (1, 0),
        )
        p_A2 = tl.make_block_ptr(
            A_ab + i_bh * T * BT,
            (T, BT),
            (BT, 1),
            (i_t * BT + BC, BC),
            (BC, BC),
            (1, 0),
        )
        p_A3 = tl.make_block_ptr(
            A_ab + i_bh * T * BT,
            (T, BT),
            (BT, 1),
            (i_t * BT + BC, 0),
            (BC, BC),
            (1, 0),
        )
        p_A_inv1 = tl.make_block_ptr(
            A_ab_inv + i_bh * T * BT,
            (T, BT),
            (BT, 1),
            (i_t * BT, 0),
            (BC, BC),
            (1, 0),
        )
        p_A_inv2 = tl.make_block_ptr(
            A_ab_inv + i_bh * T * BT,
            (T, BT),
            (BT, 1),
            (i_t * BT + BC, BC),
            (BC, BC),
            (1, 0),
        )
        p_A_inv3 = tl.make_block_ptr(
            A_ab_inv + i_bh * T * BT,
            (T, BT),
            (BT, 1),
            (i_t * BT + BC, 0),
            (BC, BC),
            (1, 0),
        )
        p_A_inv4 = tl.make_block_ptr(
            A_ab_inv + i_bh * T * BT,
            (T, BT),
            (BT, 1),
            (i_t * BT, BC),
            (BC, BC),
            (1, 0),
        )
    else:
        p_A1 = tl.make_block_ptr(
            A_ab + (bos * H + i_h) * BT,
            (T, BT),
            (H * BT, 1),
            (i_t * BT, 0),
            (BC, BC),
            (1, 0),
        )
        p_A2 = tl.make_block_ptr(
            A_ab + (bos * H + i_h) * BT,
            (T, BT),
            (H * BT, 1),
            (i_t * BT + BC, BC),
            (BC, BC),
            (1, 0),
        )
        p_A3 = tl.make_block_ptr(
            A_ab + (bos * H + i_h) * BT,
            (T, BT),
            (H * BT, 1),
            (i_t * BT + BC, 0),
            (BC, BC),
            (1, 0),
        )
        p_A_inv1 = tl.make_block_ptr(
            A_ab_inv + (bos * H + i_h) * BT,
            (T, BT),
            (H * BT, 1),
            (i_t * BT, 0),
            (BC, BC),
            (1, 0),
        )
        p_A_inv2 = tl.make_block_ptr(
            A_ab_inv + (bos * H + i_h) * BT,
            (T, BT),
            (H * BT, 1),
            (i_t * BT + BC, BC),
            (BC, BC),
            (1, 0),
        )
        p_A_inv3 = tl.make_block_ptr(
            A_ab_inv + (bos * H + i_h) * BT,
            (T, BT),
            (H * BT, 1),
            (i_t * BT + BC, 0),
            (BC, BC),
            (1, 0),
        )
        p_A_inv4 = tl.make_block_ptr(
            A_ab_inv + (bos * H + i_h) * BT,
            (T, BT),
            (H * BT, 1),
            (i_t * BT, BC),
            (BC, BC),
            (1, 0),
        )

    b_A = tl.load(p_A1, boundary_check=(0, 1))
    b_A2 = tl.load(p_A2, boundary_check=(0, 1))
    b_A3 = tl.load(p_A3, boundary_check=(0, 1))
    b_A = tl.where(
        tl.arange(0, BC)[:, None] > tl.arange(0, BC)[None, :], b_A, 0
    )
    b_A2 = tl.where(
        tl.arange(0, BC)[:, None] > tl.arange(0, BC)[None, :], b_A2, 0
    )

    for i in range(1, BC):
        mask = tl.arange(0, BC) == i
        b_a = tl.sum(tl.where(mask[:, None], b_A, 0), 0)
        b_a2 = tl.sum(tl.where(mask[:, None], b_A2, 0), 0)
        b_a = b_a + tl.sum(b_a[:, None] * b_A, 0) * (tl.arange(0, BC) < i)
        b_a2 = b_a2 + tl.sum(b_a2[:, None] * b_A2, 0) * (tl.arange(0, BC) < i)
        b_A = tl.where(mask[:, None], b_a, b_A)
        b_A2 = tl.where(mask[:, None], b_a2, b_A2)

    # blockwise computation of lower triangular matrix's inverse
    # i.e., [A11, 0; A21, A22]^-1 = [A11^-1, 0; -A22^-1 A21 A11^-1, A22^-1]
    b_A += tl.arange(0, BC)[:, None] == tl.arange(0, BC)[None, :]
    b_A2 += tl.arange(0, BC)[:, None] == tl.arange(0, BC)[None, :]
    b_A3 = tl.dot(tl.dot(b_A2, b_A3), b_A)
    tl.debug_barrier()
    tl.store(
        p_A_inv1,
        b_A.to(p_A_inv1.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )
    tl.store(
        p_A_inv2,
        b_A2.to(p_A_inv2.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )
    tl.store(
        p_A_inv3,
        b_A3.to(p_A_inv3.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )
    # causal mask
    tl.store(
        p_A_inv4,
        tl.zeros([BC, BC], dtype=tl.float32).to(p_A_inv4.dtype.element_ty),
        boundary_check=(0, 1),
    )


@triton.heuristics({"USE_OFFSETS": lambda args: args["offsets"] is not None})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4, 8, 16, 32]
        for num_stages in [2, 3, 4]
    ],
    key=["BT", "BK", "BV"],
    use_cuda_graph=use_cuda_graph,
)
@triton.jit(do_not_specialize=["T"])
def fwd_wu_kernel(
    u,
    w,
    ag,
    v,
    A_ab_inv,
    A_ak,
    offsets,
    indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_OFFSETS: tl.constexpr,
    HEAD_FIRST: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
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

    if HEAD_FIRST:
        p_A_ab_inv = tl.make_block_ptr(
            A_ab_inv + i_bh * T * BT,
            (T, BT),
            (BT, 1),
            (i_t * BT, 0),
            (BT, BT),
            (1, 0),
        )
        p_A_ak = tl.make_block_ptr(
            A_ak + i_bh * T * BT,
            (T, BT),
            (BT, 1),
            (i_t * BT, 0),
            (BT, BT),
            (1, 0),
        )
    else:
        p_A_ab_inv = tl.make_block_ptr(
            A_ab_inv + (bos * H + i_h) * BT,
            (T, BT),
            (H * BT, 1),
            (i_t * BT, 0),
            (BT, BT),
            (1, 0),
        )
        p_A_ak = tl.make_block_ptr(
            A_ak + (bos * H + i_h) * BT,
            (T, BT),
            (H * BT, 1),
            (i_t * BT, 0),
            (BT, BT),
            (1, 0),
        )
    b_Aab_inv = tl.load(p_A_ab_inv, boundary_check=(0, 1))
    b_Aak = tl.load(p_A_ak, boundary_check=(0, 1))
    o_s = tl.arange(0, BT)
    b_Aab_inv = tl.where(o_s[:, None] >= o_s[None, :], b_Aab_inv, 0)
    b_Aak = tl.where(o_s[:, None] > o_s[None, :], b_Aak, 0)
    # let's use tf32 here
    b_Aak = tl.dot(b_Aab_inv, b_Aak)
    # (SY 01/04) should be bf16 or tf32? To verify.
    b_Aak = b_Aak.to(v.dtype.element_ty, fp_downcast_rounding="rtne")
    b_Aab_inv = b_Aab_inv.to(ag.dtype.element_ty, fp_downcast_rounding="rtne")

    for i_k in range(tl.cdiv(K, BK)):
        if HEAD_FIRST:
            p_ag = tl.make_block_ptr(
                ag + i_bh * T * K,
                (T, K),
                (K, 1),
                (i_t * BT, i_k * BK),
                (BT, BK),
                (1, 0),
            )
            p_w = tl.make_block_ptr(
                w + i_bh * T * K,
                (T, K),
                (K, 1),
                (i_t * BT, i_k * BK),
                (BT, BK),
                (1, 0),
            )
        else:
            p_ag = tl.make_block_ptr(
                ag + (bos * H + i_h) * K,
                (T, K),
                (H * K, 1),
                (i_t * BT, i_k * BK),
                (BT, BK),
                (1, 0),
            )
            p_w = tl.make_block_ptr(
                w + (bos * H + i_h) * K,
                (T, K),
                (H * K, 1),
                (i_t * BT, i_k * BK),
                (BT, BK),
                (1, 0),
            )
        b_ag = tl.load(p_ag, boundary_check=(0, 1))
        b_w = tl.dot(b_Aab_inv, b_ag)  # both bf16 or fp16
        tl.store(
            p_w,
            b_w.to(p_w.dtype.element_ty, fp_downcast_rounding="rtne"),
            boundary_check=(0, 1),
        )

    for i_v in range(tl.cdiv(V, BV)):
        if HEAD_FIRST:
            p_v = tl.make_block_ptr(
                v + i_bh * T * V,
                (T, V),
                (V, 1),
                (i_t * BT, i_v * BV),
                (BT, BV),
                (1, 0),
            )
            p_u = tl.make_block_ptr(
                u + i_bh * T * V,
                (T, V),
                (V, 1),
                (i_t * BT, i_v * BV),
                (BT, BV),
                (1, 0),
            )
        else:
            p_v = tl.make_block_ptr(
                v + (bos * H + i_h) * V,
                (T, V),
                (H * V, 1),
                (i_t * BT, i_v * BV),
                (BT, BV),
                (1, 0),
            )
            p_u = tl.make_block_ptr(
                u + (bos * H + i_h) * V,
                (T, V),
                (H * V, 1),
                (i_t * BT, i_v * BV),
                (BT, BV),
                (1, 0),
            )
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_u = tl.dot(b_Aak, b_v)  # both bf16 or fp16
        tl.store(
            p_u,
            b_u.to(p_u.dtype.element_ty, fp_downcast_rounding="rtne"),
            boundary_check=(0, 1),
        )
