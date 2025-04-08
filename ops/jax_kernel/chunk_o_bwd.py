from typing import Tuple

import jax
import jax.numpy as jnp
import jax_triton as jt

import triton

from ops.get_jax_devices_info import is_triton_shared_mem_enough
from ops.triton_kernel.chunk_o_bwd import *


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
    offsets=None,
    indices=None,
    chunk_size: int = 64,
    scale: float = 1.0,
    head_first: bool = True,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    if head_first:
        B, H, T, K, V = *w.shape, v.shape[-1]
    else:
        B, T, H, K, V = *w.shape, v.shape[-1]

    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    if offsets is None:
        NT = triton.cdiv(T, BT)
    else:
        if indices is None:
            indices = jnp.concat(
                [
                    jnp.arange(n)
                    for n in triton.cdiv(offsets[1:] - offsets[:-1], BT).tolist()
                ]
            )
            indices = jnp.stack([jnp.cumsum(jnp.equal(indices, 0), 0) - 1, indices], 1)
        NT = len(indices)

    BK = (
        min(triton.next_power_of_2(K), 64)
        if device_capacity
        else min(triton.next_power_of_2(K), 32)
    )
    BV = (
        min(triton.next_power_of_2(V), 64)
        if device_capacity
        else min(triton.next_power_of_2(K), 32)
    )
    NK = triton.cdiv(K, BK)
    dq = jnp.empty_like(k)
    dk = jnp.empty_like(k)
    dw = jnp.empty_like(w)
    db = jnp.empty_like(b)
    grid = (NK, NT, B * H)

    dgk_last = (
        jnp.empty([B, H, NT, K], dtype="float32")
        if head_first
        else jnp.empty([B, NT, H, K], dtype="float32")
    )
    out_shapes = [
        jax.ShapeDtypeStruct([], v_new.dtype),
        jax.ShapeDtypeStruct([], v_new.dtype),
        jax.ShapeDtypeStruct([], v_new.dtype),
        jax.ShapeDtypeStruct([], v_new.dtype),
        jax.ShapeDtypeStruct([], v_new.dtype),
    ]
    jt.triton_call(
        v,
        v_new,
        h,
        do,
        dh,
        dk,
        db,
        w,
        dq,
        dv,
        dw,
        gk,
        dgk_last,
        k,
        b,
        offsets=offsets,
        indices=indices,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
        HEAD_FIRST=head_first,
        USE_OFFSETS=offsets is not None,
        grid=grid,
        out_shape=out_shapes,
        kernel=chunk_dplr_bwd_o_kernel.fn,
    )
    return dq, dk, dw, db, dgk_last


def chunk_dplr_bwd_dv(
    A_qk: jax.Array,
    kg: jax.Array,
    do: jax.Array,
    dh: jax.Array,
    offsets=None,
    indices=None,
    head_first: bool = True,
    chunk_size: int = 64,
) -> jax.Array:
    if head_first:
        B, H, T, K, V = *kg.shape, do.shape[-1]
    else:
        B, T, H, K, V = *kg.shape, do.shape[-1]
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    if offsets is None:
        NT = triton.cdiv(T, BT)
    else:
        if indices is None:
            indices = jnp.concat(
                [
                    jnp.arange(n)
                    for n in triton.cdiv(offsets[1:] - offsets[:-1], BT).tolist()
                ]
            )
            indices = jnp.stack([jnp.cumsum(jnp.equal(indices, 0), 0) - 1, indices], 1)
        NT = len(indices)

    dv = jnp.empty_like(do)

    def grid(meta):
        return (triton.cdiv(V, meta["BV"]), NT, B * H)

    jt.triton_call(
        A_qk,
        kg,
        do,
        dv,
        dh,
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
        out_shape=jax.ShapeDtypeStruct([], do.dtype),
        kernel=chunk_dplr_bwd_kernel_dv.fn,
    )

    return dv


def chunk_dplr_bwd_dAu(
    v: jax.Array,
    v_new: jax.Array,
    do: jax.Array,
    A_qb: jax.Array,
    scale: float,
    offsets=None,
    indices=None,
    head_first: bool = True,
    chunk_size: int = 64,
) -> jax.Array:
    if head_first:
        B, H, T, V = v.shape
    else:
        B, T, H, V = v.shape
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    if offsets is None:
        NT = triton.cdiv(T, BT)
    else:
        if indices is None:
            indices = jnp.concat(
                [
                    jnp.arange(n)
                    for n in triton.cdiv(offsets[1:] - offsets[:-1], BT).tolist()
                ]
            )
            indices = jnp.stack([jnp.cumsum(jnp.equal(indices, 0), 0) - 1, indices], 1)
        NT = len(indices)

    if is_triton_shared_mem_enough(131072):  # A100
        BV = min(triton.next_power_of_2(V), 128)
    elif is_triton_shared_mem_enough(101376):  # 4090
        BV = min(triton.next_power_of_2(V), 64)
    else:
        BV = min(triton.next_power_of_2(V), 32)

    grid = (NT, B * H)
    dA_qk = (
        jnp.empty([B, H, T, BT], dtype="float32")
        if head_first
        else jnp.empty([B, T, H, BT], dtype="float32")
    )
    dA_qb = (
        jnp.empty([B, H, T, BT], dtype="float32")
        if head_first
        else jnp.empty([B, T, H, BT], dtype="float32")
    )
    dv_new = jnp.empty_like(v_new)
    out_shapes = [
        jax.ShapeDtypeStruct([], v_new.dtype),
        jax.ShapeDtypeStruct([], v_new.dtype),
        jax.ShapeDtypeStruct([], v_new.dtype),
    ]
    jt.triton_call(
        v,
        do,
        v_new,
        A_qb,
        dA_qk,
        dA_qb,
        dv_new,
        offsets=offsets,
        indices=indices,
        scale=scale,
        T=T,
        H=H,
        V=V,
        BT=BT,
        BV=BV,
        HEAD_FIRST=head_first,
        USE_OFFSETS=offsets is not None,
        grid=grid,
        out_shape=out_shapes,
        kernel=chunk_dplr_bwd_kernel_dAu.fn,
    )
    return dv_new, dA_qk, dA_qb
