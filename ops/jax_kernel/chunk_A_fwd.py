# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang


import jax
import jax_triton as jt
import triton
from ops.triton_kernel.chunk_A_fwd import *


def chunk_fwd_intra_dplr_fn(
    q: jax.Array,
    k: jax.Array,
    a: jax.Array,
    b: jax.Array,
    gi: jax.Array,
    ge: jax.Array,
    scale: float = 1.0,
    chunk_size: int = 64,
    offsets=None,
    indices=None,
    head_first: bool = True,
):
    if head_first:
        B, H, T, K = k.shape
    else:
        B, T, H, K = k.shape
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    NT = triton.cdiv(T, BT) if offsets is None else len(indices)
    BC = min(16, BT)
    NC = triton.cdiv(BT, BC)

    shape = [B, *((H, T) if head_first else (T, H)), BT]
    grid = (int(NT), int(NC * NC), int(B * H))

    out_shapes = [
        jax.ShapeDtypeStruct(shape, q.dtype),
        jax.ShapeDtypeStruct(shape, q.dtype),
        jax.ShapeDtypeStruct(shape, "float32"),
        jax.ShapeDtypeStruct(shape, "float32"),
    ]
    Aqk, Aqb, Aab, Aak = jt.triton_call(
        q,
        k,
        a,
        b,
        gi,  # cumsum
        ge,  # before cumsum
        offsets=offsets,  # 明确命名参数
        indices=indices,
        scale=scale,
        T=T,
        H=H,
        K=K,
        BT=BT,
        BC=BC,
        NC=NC,
        USE_OFFSETS=offsets is not None,
        HEAD_FIRST=head_first,
        kernel=chunk_dplr_fwd_A_kernel_intra_sub_inter.fn,
        out_shape=out_shapes,
        grid=grid,
    )
    BK = triton.next_power_of_2(K)
    grid = (NT, NC, B * H)

    out_shapes = [
        jax.ShapeDtypeStruct(q.shape, q.dtype),
        jax.ShapeDtypeStruct(k.shape, q.dtype),
        jax.ShapeDtypeStruct(a.shape, q.dtype),
        jax.ShapeDtypeStruct(b.shape, q.dtype),
    ]
    qg, kg, ag, bg = jt.triton_call(
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
        offsets=offsets,  # 明确命名参数
        indices=indices,
        scale=scale,
        T=T,
        H=H,
        K=K,
        BT=BT,
        BC=BC,
        BK=BK,
        NC=NC,
        USE_OFFSETS=offsets is not None,
        HEAD_FIRST=head_first,
        kernel=chunk_dplr_fwd_A_kernel_intra_sub_intra.fn,
        out_shape=out_shapes,
        grid=grid,
    )

    return Aab, Aqk, Aak, Aqb, qg, kg, ag, bg
