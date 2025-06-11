# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from typing import Optional, Tuple

import jax_triton as jt
import jax
import triton


from ops.get_torch_devices_info import check_shared_mem, prepare_chunk_indices
from ops.triton_kernel.wy_fast_bwd import *


def chunk_dplr_bwd_wy(
    A_ab_inv: jax.Array,
    A_ak: jax.Array,
    v: jax.Array,
    ag: jax.Array,
    dw: jax.Array,
    du: jax.Array,
    dv0: jax.Array,
    cu_seqlens,
    chunk_size: int,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    B, T, H, K, V = *dw.shape, du.shape[-1]
    BT = min(chunk_size, max(triton.next_power_of_2(T), 16))

    chunk_indices = (
        prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    )
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    BK = min(triton.next_power_of_2(K), 64)
    BV = (
        min(triton.next_power_of_2(V), 64)
        if check_shared_mem()
        else min(triton.next_power_of_2(V), 32)
    )
    grid = (NT, B * H)
    out_shapes = [
        jax.ShapeDtypeStruct(A_ak.shape, "float32"),
        jax.ShapeDtypeStruct(A_ab_inv.shape, "float32"),
        jax.ShapeDtypeStruct(v.shape, v.dtype),
        jax.ShapeDtypeStruct(ag.shape, ag.dtype),
    ]
    dA_ak, dA_ab, dv, dag = jt.triton_call(
        A_ab_inv,
        A_ak,
        ag,
        v,
        dw,
        du,
        dv0,
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
        grid=grid,
        kernel=prepare_wy_repr_bwd_kernel.fn,
        out_shape=out_shapes,
    )
    return dA_ab, dA_ak, dv, dag
