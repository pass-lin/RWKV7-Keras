# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
from typing import Optional

import jax
import jax.numpy as jnp
import jax_triton as jt
import triton
from ops.triton_kernel.chunk_h_bwd import chunk_dplr_bwd_kernel_dhu

from ops.get_jax_devices_info import is_triton_shared_mem_enough
from ops.jax_kernel.utils import prepare_chunk_offsets


def chunk_dplr_bwd_dhu(
    qg: jax.Array,
    bg: jax.Array,
    w: jax.Array,
    gk: jax.Array,
    h0: jax.Array,
    dht: Optional[jax.Array],
    do: jax.Array,
    dv: jax.Array,
    offsets=None,
    head_first: bool = True,
    chunk_size: int = 64,
):
    if head_first:
        B, H, T, K, V = *qg.shape, do.shape[-1]
    else:
        B, T, H, K, V = *qg.shape, do.shape[-1]
    BT = min(chunk_size, max(triton.next_power_of_2(T), 16))
    BK = triton.next_power_of_2(K)
    assert BK <= 256, (
        "current kernel does not support head dimension being larger than 256."
    )
    # H100
    if is_triton_shared_mem_enough(233472, 0):
        BV = 64
        BC = 64 if K <= 128 else 32
    elif is_triton_shared_mem_enough(131072, 0):  # A100
        BV = 32
        BC = 32
    else:  # Etc: 4090
        BV = 16
        BC = 16

    # N: the actual number of sequences in the batch with either equal or variable lengths
    if offsets is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        N = len(offsets) - 1
        chunk_offsets = prepare_chunk_offsets(offsets, BT)
        NT = chunk_offsets[-1]

    BC = min(BT, BC)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
    assert NK == 1, (
        "NK > 1 is not supported because it involves time-consuming synchronization"
    )

    if head_first:
        dh = jnp.empty([B, H, NT, K, V])
    else:
        dh = jnp.empty([B, NT, H, K, V])
    dh0 = jnp.empty_like(dv, dtype="float32")
    dv2 = jnp.empty_like(dv)

    grid = (int(NK), int(NV), int(N * H))
    out_shapes = [
        jax.ShapeDtypeStruct([], dv2.dtype),
        jax.ShapeDtypeStruct([], dv2.dtype),
        jax.ShapeDtypeStruct([], dv2.dtype),
    ]
    jt.triton_call(
        qg,
        bg,
        w,
        gk,
        dht,
        dh0,
        do,
        dh,
        dv,
        dv2,
        offsets,  # 明确命名参数
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BC=BC,
        BK=BK,
        BV=BV,
        HEAD_FIRST=head_first,
        USE_OFFSETS=offsets is not None,
        USE_FINAL_STATE_GRADIENT=dht is not None,
        USE_INITIAL_STATE=dh0 is not None,
        grid=grid,
        out_shape=out_shapes,
        kernel=chunk_dplr_bwd_kernel_dhu.fn,
    )
    return dh, dh0 if h0 != None else None, dv2
