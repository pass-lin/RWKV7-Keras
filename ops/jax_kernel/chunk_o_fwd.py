# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from typing import Optional

import jax_triton as jt
import jax
import triton

from ops.get_torch_devices_info import prepare_chunk_indices
from ops.triton_kernel.chunk_o_fwd import *


def chunk_dplr_fwd_o(
    qg: jax.Array,
    v: jax.Array,
    v_new: jax.Array,
    A_qk: jax.Array,
    A_qb: jax.Array,
    h: jax.Array,
    cu_seqlens=None,
    chunk_size: int = 64,
) -> jax.Array:
    B, T, H, K, V = *qg.shape, v.shape[-1]
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))

    chunk_indices = (
        prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    )
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    def grid(meta):
        return (triton.cdiv(V, meta["BV"]), NT, B * H)

    o = jt.triton_call(
        qg,
        v,
        v_new,
        A_qk,
        A_qb,
        h,
        T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        kernel=chunk_dplr_fwd_kernel_o,
        out_shape=jax.ShapeDtypeStruct(v.shape, v.dtype),
        grid=grid,
    )
    return o
