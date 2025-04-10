# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang


import jax
import jax.numpy as jnp
import jax_triton as jt
import triton

from ops.jax_kernel.chunk_A_fwd import chunk_fwd_intra_dplr_fn
from ops.jax_kernel.chunk_h_fwd import chunk_dplr_fwd_h
from ops.jax_kernel.chunk_o_fwd import chunk_dplr_fwd_o
from ops.jax_kernel.wy_fast_fwd import fwd_prepare_wy_repr
from ops.jax_kernel.chunk_A_bwd import chunk_dplr_bwd_dqk_intra
from ops.jax_kernel.chunk_h_bwd import chunk_dplr_bwd_dhu
from ops.jax_kernel.chunk_o_bwd import (
    chunk_dplr_bwd_dAu,
    chunk_dplr_bwd_o,
    chunk_dplr_bwd_dv,
)
from ops.jax_kernel.wy_fast_bwd import chunk_dplr_bwd_wy
from ops.triton_kernel.chunk import chunk_rwkv6_fwd_cumsum_kernel

CHUNKSIZE = 16


def chunk_rwkv6_fwd_cumsum(
    g: jax.Array,
    chunk_size: int,
    offsets=None,
    indices=None,
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
    offsets=None,
    indices=None,
    head_first: bool = True,
    chunk_size: int = CHUNKSIZE,
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


def chunk_dplr_bwd(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    a: jax.Array,
    b: jax.Array,
    gk: jax.Array,
    initial_state,
    head_first,
    scale,
    do: jax.Array,
    dht: jax.Array,
    offsets=None,
    indices=None,
):
    DTYPE = do.dtype
    BT = CHUNKSIZE
    # ******* start recomputing everything, otherwise i believe the gpu memory will be exhausted *******
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
    w, u, A_ab_inv = fwd_prepare_wy_repr(
        ag=ag,
        A_ab=A_ab,
        A_ak=A_ak,
        v=v,
        offsets=offsets,
        indices=indices,
        head_first=head_first,
        chunk_size=BT,
    )
    del A_ab
    h, v_new, _ = chunk_dplr_fwd_h(
        kg=kg,
        bg=bg,
        v=v,
        w=w,
        u=u,
        gk=gi,
        initial_state=initial_state,
        offsets=offsets,
        head_first=head_first,
        chunk_size=BT,
    )
    del u
    # ******* end of recomputation *******
    # A_ak, A_ab_inv, gi, ge torch.float32
    # A_qk, A_qb, qg, kg, ag, bg, v_new dtype=q.dtype, eg: bf16

    dv_new_intra, dA_qk, dA_qb = chunk_dplr_bwd_dAu(
        v=v,
        v_new=v_new,
        do=do,
        A_qb=A_qb,
        scale=scale,
        offsets=offsets,
        indices=indices,
        head_first=head_first,
        chunk_size=BT,
    )

    dh, dh0, dv_new = chunk_dplr_bwd_dhu(
        qg=qg,
        bg=bg,
        w=w,
        gk=gi,
        h0=initial_state,
        dht=dht,
        do=do,
        dv=dv_new_intra,
        offsets=offsets,
        head_first=head_first,
        chunk_size=BT,
    )

    dv = chunk_dplr_bwd_dv(
        A_qk=A_qk,
        kg=kg,
        do=do,
        dh=dh,
        offsets=offsets,
        indices=indices,
        head_first=head_first,
        chunk_size=BT,
    )

    del A_qk

    dqg, dkg, dw, dbg, dgk_last = chunk_dplr_bwd_o(
        k=kg,
        b=bg,
        v=v,
        v_new=v_new,
        do=do,
        h=h,
        dh=dh,
        dv=dv_new,
        w=w,
        gk=gi,
        offsets=offsets,
        indices=indices,
        chunk_size=BT,
        scale=scale,
        head_first=head_first,
    )

    del v_new

    dA_ab, dA_ak, dv, dag = chunk_dplr_bwd_wy(
        A_ab_inv=A_ab_inv,
        A_ak=A_ak,
        v=v,
        ag=ag,
        dw=dw,
        du=dv_new,
        dv0=dv,
        offsets=offsets,
        indices=indices,
        head_first=head_first,
        chunk_size=BT,
    )

    del A_ak

    dq, dk, da, db, dgk = chunk_dplr_bwd_dqk_intra(
        q=q,
        k=k,
        a=a,
        b=b,
        gi=gi,
        ge=ge,
        dAqk=dA_qk,
        dAqb=dA_qb,
        dAak=dA_ak,
        dAab=dA_ab,
        dgk_last=dgk_last,
        dqg=dqg,
        dkg=dkg,
        dag=dag,
        dbg=dbg,
        chunk_size=BT,
        scale=scale,
        head_first=head_first,
        offsets=offsets,
        indices=indices,
    )

    return (
        jnp.asarray(dq, dtype=DTYPE),
        jnp.asarray(dk, dtype=DTYPE),
        jnp.asarray(dv, dtype=DTYPE),
        jnp.asarray(da, dtype=DTYPE),
        jnp.asarray(db, dtype=DTYPE),
        jnp.asarray(dgk, dtype=DTYPE),
        None,
        dh0,
        None,
        None,
        None,
    )
