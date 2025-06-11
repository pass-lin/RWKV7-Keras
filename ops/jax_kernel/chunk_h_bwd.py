# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from typing import Optional, Tuple

import jax_triton as jt
import jax
import triton

from ops.get_jax_devices_info import check_shared_mem, prepare_chunk_indices
from ops.triton_kernel.chunk_h_bwd import *


def chunk_dplr_bwd_dhu(
    qg: jax.Array,
    bg: jax.Array,
    w: jax.Array,
    gk: jax.Array,
    h0: jax.Array,
    dht: Optional[jax.Array],
    do: jax.Array,
    dv: jax.Array,
    cu_seqlens=None,
    chunk_size: int = 64,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    B, T, H, K, V = *qg.shape, do.shape[-1]
    BT = min(chunk_size, max(triton.next_power_of_2(T), 16))
    BK = triton.next_power_of_2(K)
    assert BK <= 256, (
        "current kernel does not support head dimension being larger than 256."
    )
    # H100
    if check_shared_mem("hopper", jax.devices().index(qg.device)):
        BV = 64
        BC = 64 if K <= 128 else 32
    elif check_shared_mem("ampere", jax.devices().index(qg.device)):  # A100
        BV = 32
        BC = 32
    else:  # Etc: 4090
        BV = 16
        BC = 16

    chunk_indices = (
        prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    )
    # N: the actual number of sequences in the batch with either equal or variable lengths
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        raise (1)
        N, NT, chunk_offsets = (
            len(cu_seqlens) - 1,
            len(chunk_indices),
            prepare_chunk_offsets(cu_seqlens, BT),
        )

    BC = min(BT, BC)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
    assert NK == 1, (
        "NK > 1 is not supported because it involves time-consuming synchronization"
    )
    dh_shape = (B, NT, H, K, V)
    out_shapes = [
        jax.ShapeDtypeStruct(dh_shape, dv.dtype),
        jax.ShapeDtypeStruct(dv.shape, "float32"),
        jax.ShapeDtypeStruct(dv.shape, dv.dtype),
    ]

    grid = (NK, NV, N * H)
    dh, dh0, dv2 = jt.triton_call(
        qg,
        bg,
        w,
        gk,
        dht,
        do,
        dv,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BC=BC,
        BK=BK,
        BV=BV,
        IS_VARLEN=cu_seqlens is not None,
        kernel=chunk_dplr_bwd_kernel_dhu.fn,
        out_shape=out_shapes,
        grid=grid,
        USE_FINAL_STATE_GRADIENT=dht is not None,
        USE_INITIAL_STATE=h0 is not None,
    )
    return dh, dh0, dv2
