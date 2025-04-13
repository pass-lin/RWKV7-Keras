import jax
import jax_triton as jt
import triton
from ops.get_jax_devices_info import device_capacity
from ops.triton_kernel.chunk_A_bwd import *


# @partial(jax.jit, static_argnames=['head_first',"chunk_size","scale"])
def chunk_dplr_bwd_dqk_intra(
    q: jax.Array,
    k: jax.Array,
    a: jax.Array,
    b: jax.Array,
    gi: jax.Array,
    ge: jax.Array,
    dAqk: jax.Array,
    dAqb: jax.Array,
    dAak: jax.Array,
    dAab: jax.Array,
    dqg: jax.Array,
    dkg: jax.Array,
    dag: jax.Array,
    dbg: jax.Array,
    dgk_last: jax.Array,
    offsets=None,
    indices=None,
    head_first: bool = True,
    scale: float = 1.0,
    chunk_size: int = 64,
):
    if head_first:
        B, H, T, K = q.shape
    else:
        B, T, H, K = q.shape
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    BC = min(16, BT)
    BK = (
        min(64, triton.next_power_of_2(K))
        if device_capacity
        else min(32, triton.next_power_of_2(K))
    )
    NT = triton.cdiv(T, BT) if offsets is None else len(indices)
    NC = triton.cdiv(BT, BC)
    NK = triton.cdiv(K, BK)

    grid = (int(NK), int(NT * NC), int(B * H))
    # 定义输出形状结构
    out_shapes = [
        jax.ShapeDtypeStruct(q.shape, q.dtype),
        jax.ShapeDtypeStruct(k.shape, k.dtype),
        jax.ShapeDtypeStruct(a.shape, a.dtype),
        jax.ShapeDtypeStruct(b.shape, b.dtype),
        jax.ShapeDtypeStruct(gi.shape, "float32"),
        jax.ShapeDtypeStruct(gi.shape, "float32"),
    ]

    # 调用第一个内核
    dq, dk, da, db, dgk, dgk_offset = jt.triton_call(
        q,
        k,
        a,
        b,
        gi,
        ge,
        dAqk,
        dAqb,
        dAak,
        dAab,
        dqg,
        dkg,
        dag,
        dbg,
        offsets=offsets,
        indices=indices,
        scale=scale,
        T=T,
        H=H,
        K=K,
        BT=BT,
        BC=BC,
        BK=BK,
        NC=NC,
        USE_OFFSETS=offsets is not None,
        HEAD_FIRST=head_first,
        kernel=chunk_dplr_bwd_kernel_intra.fn,
        out_shape=out_shapes,
        grid=grid,
        num_warps=4,
        num_stages=3,
    )

    grid2 = (int(NT), int(triton.cdiv(K, BK)), int(B * H))

    dgk_output = jt.triton_call(
        dgk,
        dgk_offset,
        dgk_last,
        offsets=offsets,  # 明确命名参数
        indices=indices,
        T=T,  # 显式传递标量参数
        H=H,
        K=K,
        BT=BT,
        BK=BK,
        USE_OFFSETS=offsets is not None,
        HEAD_FIRST=head_first,
        kernel=chunk_dplr_bwd_dgk_kernel.fn,
        out_shape=jax.ShapeDtypeStruct(dgk.shape, dgk.dtype),  # 单输出结构
        grid=grid2,
        num_warps=4,
        num_stages=3,
    )

    return dq, dk, da, db, dgk_output
