from ops.triton_kernel.cumsum import *
import jax_triton as jt
import jax
import triton
from ops.get_torch_devices_info import prepare_chunk_indices


def chunk_rwkv6_fwd_cumsum(
    g: jax.Array,
    chunk_size: int,
    cu_seqlens=None,
) -> jax.Array:
    B, T, H, S = g.shape
    BT = chunk_size
    chunk_indices = (
        prepare_chunk_indices(cu_seqlens, chunk_size)
        if cu_seqlens is not None
        else None
    )
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    out_shapes = [
        jax.ShapeDtypeStruct(g.shape, "float32"),
        jax.ShapeDtypeStruct(g.shape, "float32"),
    ]

    def grid(meta):
        return (triton.cdiv(meta["S"], meta["BS"]), NT, B * H)

    gi, ge = jt.triton_call(
        g,
        T,
        H=H,
        S=S,
        BT=BT,
        grid=grid,
        kernel=chunk_rwkv6_fwd_cumsum_kernel,
        out_shape=out_shapes,
    )

    return gi, ge
