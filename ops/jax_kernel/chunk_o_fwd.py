# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang


import jax
import jax_triton as jt
import triton

from ops.triton_kernel.chunk_o_fwd import chunk_dplr_fwd_kernel_o


def chunk_dplr_fwd_o(
    qg: jax.Array,
    v: jax.Array,
    v_new: jax.Array,
    A_qk: jax.Array,
    A_qb: jax.Array,
    h: jax.Array,
    offsets=None,
    indices=None,
    head_first: bool = True,
    chunk_size: int = 64,
) -> jax.Array:
    if head_first:
        B, H, T, K, V = *qg.shape, v.shape[-1]
    else:
        B, T, H, K, V = *qg.shape, v.shape[-1]
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    NT = triton.cdiv(T, BT) if offsets is None else len(indices)

    def grid(meta):
        return (triton.cdiv(V, meta["BV"]), NT, B * H)

    o = jt.triton_call(
        qg,
        v,
        v_new,
        A_qk,
        A_qb,
        h,
        offsets=offsets,
        indices=indices,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        HEAD_FIRST=head_first,
        USE_OFFSETS=offsets is not None,
        grid=grid,
        out_shape=jax.ShapeDtypeStruct(v.shape, v.dtype),
        kernel=chunk_dplr_fwd_kernel_o.fn,
    )
    return o
