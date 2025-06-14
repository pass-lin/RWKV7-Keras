# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from typing import Optional
import jax_triton as jt
import jax
import triton
from ops.triton_kernel.chunk_A_bwd import *
from ops.triton_kernel.utils import is_gather_supported
from ops.get_torch_devices_info import check_shared_mem, prepare_chunk_indices


def chunk_dplr_bwd_dqk_intra(
    q: jax.Array,
    k: jax.Array,
    a: jax.Array,
    b: jax.Array,
    gi: jax.Array,
    ge: jax.Array,
    dAqk: jax.Array,
    dAqb: jax.Array,
    dAak: jax.Array,
    dAab: jax.Array,
    dqg: jax.Array,
    dkg: jax.Array,
    dag: jax.Array,
    dbg: jax.Array,
    dgk_last: jax.Array,
    scale: float = 1.0,
    cu_seqlens=None,
    chunk_size: int = 64,
):
    B, T, H, K = q.shape
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    BK = (
        min(64, triton.next_power_of_2(K))
        if check_shared_mem()
        else min(32, triton.next_power_of_2(K))
    )

    chunk_indices = (
        prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    )
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    NK = triton.cdiv(K, BK)
    out_shapes = [
        jax.ShapeDtypeStruct(q.shape, q.dtype),
        jax.ShapeDtypeStruct(k.shape, k.dtype),
        jax.ShapeDtypeStruct(a.shape, a.dtype),
        jax.ShapeDtypeStruct(b.shape, b.dtype),
        jax.ShapeDtypeStruct(gi.shape, "float32"),
        jax.ShapeDtypeStruct(gi.shape, "float32"),
    ]

    grid = (NK, NT, B * H)
    dq, dk, da, db, dgk, dgk_offset = jt.triton_call(
        q,
        k,
        a,
        b,
        gi,
        ge,
        dAqk,
        dAqb,
        dAak,
        dAab,
        dqg,
        dkg,
        dag,
        dbg,
        T,
        scale=scale,
        H=H,
        K=K,
        BT=BT,
        BC=BT,
        BK=BK,
        GATHER_SUPPORTED=is_gather_supported,
        kernel=chunk_dplr_bwd_kernel_intra,
        out_shape=out_shapes,
        grid=grid,
    )

    def grid(meta):
        return (NT, triton.cdiv(K, meta["BK"]), B * H)

    dgk_output = jt.triton_call(
        dgk,
        dgk_offset,
        dgk_last,
        T,
        H=H,
        K=K,
        BT=BT,
        kernel=chunk_dplr_bwd_dgk_kernel,
        out_shape=jax.ShapeDtypeStruct(dgk.shape, dgk.dtype),
        grid=grid,
    )
    return dq, dk, da, db, dgk_output
