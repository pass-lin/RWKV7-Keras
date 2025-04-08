# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from typing import Optional

import jax
import jax.numpy as jnp
import jax_triton as jt
import triton

from ops.jax_kernel.chunk_A_fwd import chunk_fwd_intra_dplr_fn
from ops.jax_kernel.chunk_h_fwd import chunk_dplr_fwd_h
from ops.jax_kernel.chunk_o_fwd import chunk_dplr_fwd_o
from ops.jax_kernel.wy_fast_fwd import fwd_prepare_wy_repr
from ops.triton_kernel.chunk import chunk_rwkv6_fwd_cumsum_kernel


def chunk_rwkv6_fwd_cumsum(
    g: jax.Array,
    chunk_size: int,
    offsets = None,
    indices = None,
    head_first: bool = True,
) -> jax.Array:
    if head_first:
        B, H, T, S = g.shape
    else:
        B, T, H, S = g.shape
    BT = chunk_size
    NT = triton.cdiv(T, BT) if offsets is None else len(indices)

    gi, ge = (
        jnp.empty_like(g, dtype="float32"),
        jnp.empty_like(g, dtype="float32"),
    )
    out_shapes = [
        jax.ShapeDtypeStruct([], gi.dtype),
        jax.ShapeDtypeStruct([], ge.dtype),
    ]

    def grid(meta):
        return (triton.cdiv(meta["S"], meta["BS"]), NT, B * H)
    jt.triton_call(
        g,
        gi,
        ge,
        offsets, 
        indices,
        T=T,
        H=H,
        S=S, 
        BT=BT, 
        HEAD_FIRST=head_first,
        USE_OFFSETS=offsets is not None,
        grid=grid,
        kernel=chunk_rwkv6_fwd_cumsum_kernel.fn,
        out_shape=out_shapes,
    )

    return gi, ge


def chunk_dplr_fwd(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    a: jax.Array,
    b: jax.Array,
    gk: jax.Array,
    scale: float,
    initial_state: jax.Array,
    output_final_state: bool,
    offsets = None,
    indices = None,
    head_first: bool = True,
    chunk_size: int = 64,
):
    T = q.shape[2] if head_first else q.shape[1]
    BT = min(chunk_size, max(triton.next_power_of_2(T), 16))
    gi, ge = chunk_rwkv6_fwd_cumsum(
        gk, BT, offsets=offsets, indices=indices, head_first=head_first
    )

    A_ab, A_qk, A_ak, A_qb, qg, kg, ag, bg = chunk_fwd_intra_dplr_fn(
        q=q,
        k=k,
        a=a,
        b=b,
        gi=gi,
        ge=ge,
        scale=scale,
        offsets=offsets,
        indices=indices,
        chunk_size=BT,
        head_first=head_first,
    )
    
    del ge

    # A_ab, A_ak, gi, ge "float32"32
    # A_qk, A_qb, qg, kg, ag, bg, dtype=q.dtype, eg: bf16
    w, u, _ = fwd_prepare_wy_repr(
        ag=ag,
        A_ab=A_ab,
        A_ak=A_ak,
        v=v,
        offsets=offsets,
        indices=indices,
        head_first=head_first,
        chunk_size=BT,
    )
    del A_ab, A_ak
    
    h, v_new, final_state = chunk_dplr_fwd_h(
        kg=kg,
        bg=bg,
        v=v,
        w=w,
        u=u,
        gk=gi,
        initial_state=initial_state,
        output_final_state=output_final_state,
        offsets=offsets,
        head_first=head_first,
        chunk_size=BT,
    )
    del u, kg, bg, gi

    o = chunk_dplr_fwd_o(
        qg=qg,
        v=v,
        v_new=v_new,
        A_qk=A_qk,
        A_qb=A_qb,
        h=h,
        offsets=offsets,
        indices=indices,
        head_first=head_first,
        chunk_size=BT,
    )
    del v_new, h, A_qk, A_qb

    return o, final_state
