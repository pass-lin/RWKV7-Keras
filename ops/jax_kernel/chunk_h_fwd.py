# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from typing import Optional
from typing import Tuple

import jax
import jax.numpy as jnp
import jax_triton as jt

import triton

from ops.get_jax_devices_info import is_triton_shared_mem_enough
from ops.jax_kernel.utils import prepare_chunk_offsets
from ops.triton_kernel.chunk_h_fwd import chunk_dplr_fwd_kernel_h


def chunk_dplr_fwd_h(
    kg: jax.Array,
    v: jax.Array,
    w: jax.Array,
    u: jax.Array,
    bg: jax.Array,
    gk: jax.Array,
    initial_state: Optional[jax.Array] = None,
    output_final_state: bool = False,
    offsets=None,
    head_first: bool = True,
    chunk_size: int = 64,
) -> Tuple[jax.Array, jax.Array]:
    if head_first:
        B, H, T, K, V = *kg.shape, u.shape[-1]
    else:
        B, T, H, K, V = *kg.shape, u.shape[-1]
    BT = min(chunk_size, max(triton.next_power_of_2(T), 16))
    # N: the actual number of sequences in the batch with either equal or variable lengths
    if offsets is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        N = len(offsets) - 1
        chunk_offsets = prepare_chunk_offsets(offsets, BT)
        NT = chunk_offsets[-1]
    BK = triton.next_power_of_2(K)
    assert BK <= 256, "current kernel does not support head dimension larger than 256."
    # H100 can have larger block size

    if is_triton_shared_mem_enough(233472, 0):
        BV = 64
        BC = 64 if K <= 128 else 32
    elif is_triton_shared_mem_enough(131072, 0):  # A100
        BV = 32
        BC = 32
    else:
        BV = 16
        BC = 16

    BC = min(BT, BC)
    NK = triton.cdiv(K, BK)
    NV = triton.cdiv(V, BV)
    assert NK == 1, (
        "NK > 1 is not supported because it involves time-consuming synchronization"
    )

    if head_first:
        h_shape = (B, H, NT, K, V)
    else:
        h_shape = (B, NT, H, K, V)
    grid = (NK, NV, N * H)
    out_shapes = [
        jax.ShapeDtypeStruct(h_shape, kg.dtype),
        jax.ShapeDtypeStruct([N, H, K, V], "float32"),
        jax.ShapeDtypeStruct(u.shape, u.dtype),
    ]
    if initial_state == None:
        initial_state = jnp.zeros([N, H, K, V], dtype="float32")
    h, final_state, v_new = jt.triton_call(
        kg,
        v,
        w,
        bg,
        u,
        gk,
        initial_state,
        offsets=offsets,
        chunk_offsets=None,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BC=BC,
        BK=BK,
        BV=BV,
        NT=NT,
        USE_OFFSETS=offsets is not None,
        HEAD_FIRST=head_first,
        USE_INITIAL_STATE=initial_state is not None,
        STORE_FINAL_STATE=True,
        grid=grid,
        out_shape=out_shapes,
        kernel=chunk_dplr_fwd_kernel_h.fn,
    )
    return h, v_new, final_state if output_final_state else None
