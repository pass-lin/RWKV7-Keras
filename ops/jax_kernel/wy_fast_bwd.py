from typing import Tuple

import jax
import jax.numpy as jnp
import jax_triton as jt
import triton


from ops.get_torch_devices_info import device_capacity
from ops.triton_kernel.wy_fast_bwd import bwd_prepare_wy_repr_kernel


def chunk_dplr_bwd_wy(
    A_ab_inv: jax.Array,
    A_ak: jax.Array,
    v: jax.Array,
    ag: jax.Array,
    dw: jax.Array,
    du: jax.Array,
    dv0: jax.Array,
    offsets=None,
    indices=None,
    head_first: bool = True,
    chunk_size: int = 64,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    if head_first:
        B, H, T, K, V = *dw.shape, du.shape[-1]
    else:
        B, T, H, K, V = *dw.shape, du.shape[-1]
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
    BV = (
        min(triton.next_power_of_2(V), 64)
        if device_capacity
        else min(triton.next_power_of_2(V), 32)
    )

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
        grid=(NT, B * H),
        kernel=bwd_prepare_wy_repr_kernel.fn,
        out_shape=out_shapes,
    )
    return dA_ab, dA_ak, dv, dag
