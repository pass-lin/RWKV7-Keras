# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from typing import Optional, Tuple

import jax_triton as jt
import jax
import triton

from ops.get_torch_devices_info import prepare_chunk_indices, check_shared_mem
from ops.triton_kernel.chunk_o_bwd import *


def chunk_dplr_bwd_dv(
    A_qk: jax.Array,
    kg: jax.Array,
    do: jax.Array,
    dh: jax.Array,
    cu_seqlens=None,
    chunk_size: int = 64,
) -> jax.Array:
    B, T, H, K, V = *kg.shape, do.shape[-1]
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))

    chunk_indices = (
        prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    )
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    def grid(meta):
        return (triton.cdiv(V, meta["BV"]), NT, B * H)

    dv = jt.triton_call(
        A_qk,
        kg,
        do,
        dh,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        IS_VARLEN=cu_seqlens is not None,
        kernel=chunk_dplr_bwd_kernel_dv.fn,
        out_shape=jax.ShapeDtypeStruct(do.shape, do.dtype),
        grid=grid,
    )
    return dv


def chunk_dplr_bwd_o(
    k: jax.Array,
    b: jax.Array,
    v: jax.Array,
    v_new: jax.Array,
    gk: jax.Array,
    do: jax.Array,
    h: jax.Array,
    dh: jax.Array,
    dv: jax.Array,
    w: jax.Array,
    cu_seqlens=None,
    chunk_size: int = 64,
    scale: float = 1.0,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    B, T, H, K, V = *w.shape, v.shape[-1]

    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    chunk_indices = (
        prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    )
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    BK = (
        min(triton.next_power_of_2(K), 64)
        if check_shared_mem()
        else min(triton.next_power_of_2(K), 32)
    )
    BV = (
        min(triton.next_power_of_2(V), 64)
        if check_shared_mem()
        else min(triton.next_power_of_2(K), 32)
    )
    NK = triton.cdiv(K, BK)
    grid = (NK, NT, B * H)

    out_shapes = [
        jax.ShapeDtypeStruct(k.shape, k.dtype),
        jax.ShapeDtypeStruct(k.shape, k.dtype),
        jax.ShapeDtypeStruct(w.shape, w.dtype),
        jax.ShapeDtypeStruct(b.shape, b.dtype),
        jax.ShapeDtypeStruct([B, NT, H, K], "float32"),
    ]
    dq, dk, dw, db, dgk_last = jt.triton_call(
        v,
        v_new,
        h,
        do,
        dh,
        w,
        dv,
        gk,
        k,
        b,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
        IS_VARLEN=cu_seqlens is not None,
        kernel=chunk_dplr_bwd_o_kernel.fn,
        out_shape=out_shapes,
        grid=grid,
    )
    return dq, dk, dw, db, dgk_last


def chunk_dplr_bwd_dAu(
    v: jax.Array,
    v_new: jax.Array,
    do: jax.Array,
    A_qb: jax.Array,
    scale: float,
    cu_seqlens=None,
    chunk_size: int = 64,
) -> jax.Array:
    B, T, H, V = v.shape
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    chunk_indices = (
        prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    )
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    if check_shared_mem("ampere"):  # A100
        BV = min(triton.next_power_of_2(V), 128)
    elif check_shared_mem("ada"):  # 4090
        BV = min(triton.next_power_of_2(V), 64)
    else:
        BV = min(triton.next_power_of_2(V), 32)

    grid = (NT, B * H)
    dA_qk = torch.empty(B, T, H, BT, dtype=torch.float, device=v.device)
    dA_qb = torch.empty(B, T, H, BT, dtype=torch.float, device=v.device)
    dv_new = torch.empty_like(v_new)
    chunk_dplr_bwd_kernel_dAu[grid](
        v=v,
        do=do,
        v_new=v_new,
        A_qb=A_qb,
        dA_qk=dA_qk,
        dA_qb=dA_qb,
        dv_new=dv_new,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        T=T,
        H=H,
        V=V,
        BT=BT,
        BV=BV,
    )
    return dv_new, dA_qk, dA_qb
