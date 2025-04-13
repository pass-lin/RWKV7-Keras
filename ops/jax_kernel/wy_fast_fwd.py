from typing import Tuple

import jax
import jax.numpy as jnp
import jax_triton as jt
import triton

from ops.triton_kernel.wy_fast_fwd import fwd_prepare_wy_repr_kernel_chunk32
from ops.triton_kernel.wy_fast_fwd import fwd_prepare_wy_repr_kernel_chunk64
from ops.triton_kernel.wy_fast_fwd import fwd_wu_kernel


def fwd_wu(
    ag: jax.Array,
    v: jax.Array,
    A_ak: jax.Array,
    A_ab_inv: jax.Array,
    offsets=None,
    indices=None,
    head_first: bool = True,
    chunk_size: int = 64,
) -> Tuple[jax.Array, jax.Array]:
    if head_first:
        B, H, T, K, V = *ag.shape, v.shape[-1]
    else:
        B, T, H, K, V = *ag.shape, v.shape[-1]
    BT = min(chunk_size, max(triton.next_power_of_2(T), 16))

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
    BK = min(triton.next_power_of_2(K), 64)
    BV = min(triton.next_power_of_2(V), 64)

    out_shapes = [
        jax.ShapeDtypeStruct(v.shape, v.dtype),
        jax.ShapeDtypeStruct(ag.shape, ag.dtype),
    ]
    w, u = jt.triton_call(
        ag,
        v,
        A_ab_inv,
        A_ak,
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
        grid=(int(NT), int(B * H)),
        kernel=fwd_wu_kernel.fn,
        out_shape=out_shapes,
    )

    return w, u


def fwd_prepare_wy_repr(
    ag: jax.Array,
    v: jax.Array,
    A_ak: jax.Array,
    A_ab: jax.Array,
    offsets=None,
    indices=None,
    head_first: bool = True,
    chunk_size: int = 64,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    if head_first:
        B, H, T, K = ag.shape
    else:
        B, T, H, K = ag.shape
    BT = min(chunk_size, max(triton.next_power_of_2(T), 16))

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
    BC = min(BT, 32)
    fwd_fn = (
        fwd_prepare_wy_repr_kernel_chunk64
        if BT == 64
        else fwd_prepare_wy_repr_kernel_chunk32
    )
    A_ab_inv = jt.triton_call(
        A_ab,
        offsets=offsets,
        indices=indices,
        T=T,
        H=H,
        BT=BT,
        BC=BC,
        HEAD_FIRST=head_first,
        USE_OFFSETS=offsets is not None,
        grid=(int(NT), int(B * H)),
        kernel=fwd_fn.fn,
        out_shape=jax.ShapeDtypeStruct(A_ab.shape, A_ab.dtype),
    )
    w, u = fwd_wu(
        ag=ag,
        v=v,
        A_ak=A_ak,
        A_ab_inv=A_ab_inv,
        offsets=offsets,
        indices=indices,
        head_first=head_first,
        chunk_size=BT,
    )
    return w, u, A_ab_inv
