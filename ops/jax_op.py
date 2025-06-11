import jax
import jax.numpy as jnp
from keras import ops
import triton
from einops import rearrange
from ops.jax_kernel.chunk_A_bwd import chunk_dplr_bwd_dqk_intra
from ops.jax_kernel.chunk_A_fwd import chunk_dplr_fwd_intra
from ops.jax_kernel.chunk_h_bwd import chunk_dplr_bwd_dhu
from ops.jax_kernel.chunk_h_fwd import chunk_dplr_fwd_h
from ops.jax_kernel.chunk_o_bwd import (
    chunk_dplr_bwd_dAu,
    chunk_dplr_bwd_dv,
    chunk_dplr_bwd_o,
)
from ops.jax_kernel.chunk_o_fwd import chunk_dplr_fwd_o
from ops.jax_kernel.wy_fast_bwd import chunk_dplr_bwd_wy
from ops.jax_kernel.wy_fast_fwd import prepare_wy_repr_fwd
from ops.jax_kernel.cumsum import chunk_rwkv6_fwd_cumsum
from ops.get_torch_devices_info import (
    autocast_custom_bwd,
    autocast_custom_fwd,
    input_guard,
)

CHUNKSIZE = 16


def chunk_dplr_fwd(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    a: jax.Array,
    b: jax.Array,
    gk: jax.Array,
    scale: float,
    initial_state: jax.Array,
    output_final_state: bool,
    chunk_size: int = 16,
):
    T = q.shape[1]
    BT = min(chunk_size, max(triton.next_power_of_2(T), 16))

    gi, ge = chunk_rwkv6_fwd_cumsum(gk, BT)
    A_ab, A_qk, A_ak, A_qb, qg, kg, ag, bg = chunk_dplr_fwd_intra(
        q=q,
        k=k,
        a=a,
        b=b,
        gi=gi,
        ge=ge,
        scale=scale,
        chunk_size=BT,
    )
    del ge

    # A_ab, A_ak, gi, ge torch.float32
    # A_qk, A_qb, qg, kg, ag, bg, dtype=q.dtype, eg: bf16
    w, u, _ = prepare_wy_repr_fwd(ag=ag, A_ab=A_ab, A_ak=A_ak, v=v, chunk_size=BT)

    del A_ab, A_ak
    h, v_new, final_state = chunk_dplr_fwd_h(
        kg=kg,
        bg=bg,
        v=v,
        w=w,
        u=u,
        gk=gi,
        initial_state=initial_state,
        output_final_state=output_final_state,
        chunk_size=BT,
    )

    del u, kg, bg, gi

    o = chunk_dplr_fwd_o(
        qg=qg,
        v=v,
        v_new=v_new,
        A_qk=A_qk,
        A_qb=A_qb,
        h=h,
        chunk_size=BT,
    )
    del v_new, h, A_qk, A_qb

    return o, final_state


def chunk_dplr_delta_rule_fwd(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    a: jax.Array,
    b: jax.Array,
    gk: jax.Array,
    scale=None,
    initial_state=None,
    output_final_state: bool = True,
):
    r"""
    Args:
        q (jax.Array):
            queries of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
        k (jax.Array):
            keys of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
        v (jax.Array):
            values of shape `[B, H, T, V]` if `head_first=True` else `[B, T, H, V]`.
        a (jax.Array):
            activations of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
        b (jax.Array):
            betas of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
        gk (jax.Array):
            gk of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`. decay term in log space!
        scale (Optional[int]):
            Scale factor for the RetNet attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[jax.Array]):
            Initial state of shape `[N, H, K, V]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N, H, K, V]`. Default: `False`.
        head_first (Optional[bool]):
            Whether the inputs are in the head-first format, which is not supported for variable-length inputs.
            Default: `False`.

    Returns:
        o (jax.Array):
            Outputs of shape `[B, H, T, V]` if `head_first=True` else `[B, T, H, V]`.
        final_state (jax.Array):
            Final state of shape `[N, H, K, V]` if `output_final_state=True` else `None`.
    """
    assert q.dtype == k.dtype == v.dtype
    # assert q.dtype != torch.float32, "ChunkDeltaRuleFunction does not support float32. Please use bfloat16."
    # gk = gk.float()

    scale = k.shape[-1] ** -0.5 if scale is None else scale
    chunk_size = CHUNKSIZE

    o, final_state = chunk_dplr_fwd(
        q=q,
        k=k,
        v=v,
        a=a,
        b=b,
        gk=gk,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        chunk_size=chunk_size,
    )
    return o, final_state


def cal_log_w(w: jax.Array) -> jax.Array:
    return -jnp.exp(w)


@jax.custom_vjp
def chunk_dplr(
    r: jax.Array,
    k: jax.Array,
    v: jax.Array,
    a: jax.Array,
    b: jax.Array,
    gk: jax.Array,
    initial_state: jax.Array = None,
):
    """
    Args:
        r (jax.Array):
            r of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
        k (jax.Array):
            k of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
        v (jax.Array):
            v of shape `[B, H, T, V]` if `head_first=True` else `[B, T, H, V]`.
        a (jax.Array):
            a of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
        b (jax.Array):
            b of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
        w (jax.Array):
            decay of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`, kernel
            will apply log_w = -torch.exp(w)
        log_w (jax.Array):
            log decay of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
        scale (float):
            scale of the attention.
        initial_state (Optional[jax.Array]):
            Initial state of shape `[N, H, K, V]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N, H, K, V]`. Default: `False`.
        head_first (bool):
            whether to use head first. Recommended to be False to avoid extra transposes.
    """

    return chunk_dplr_delta_rule_fwd(
        q=r,
        k=k,
        v=v,
        a=a,
        b=b,
        gk=gk,
        scale=1,
        initial_state=initial_state,
        output_final_state=True,
    )


def chunk_dplr_fwd_jax(
    r: jax.Array,
    k: jax.Array,
    v: jax.Array,
    a: jax.Array,
    b: jax.Array,
    gk: jax.Array,
    initial_state: jax.Array = None,
):
    o, state = chunk_dplr_delta_rule_fwd(
        q=r,
        k=k,
        v=v,
        a=a,
        b=b,
        gk=gk,
        scale=1,
        initial_state=initial_state,
        output_final_state=True,
    )
    cache = (r, k, v, a, b, gk, initial_state)
    return [o, state], cache


def chunk_dplr_bwd(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    a: jax.Array,
    b: jax.Array,
    gk: jax.Array,
    initial_state,
    scale,
    do: jax.Array,
    dht: jax.Array,
    chunk_size: int = CHUNKSIZE,
):
    DTYPE = do.dtype
    BT = chunk_size
    scale = scale
    if do != None:
        do = ops.cast(do, q.dtype)
    if dht != None:
        dht = ops.cast(dht, q.dtype)

    # ******* start recomputing everything, otherwise i believe the gpu memory will be exhausted *******
    gi, ge = chunk_rwkv6_fwd_cumsum(gk, BT)

    A_ab, A_qk, A_ak, A_qb, qg, kg, ag, bg = chunk_dplr_fwd_intra(
        q=q,
        k=k,
        a=a,
        b=b,
        gi=gi,
        ge=ge,
        scale=scale,
        chunk_size=BT,
    )
    w, u, A_ab_inv = prepare_wy_repr_fwd(
        ag=ag, A_ab=A_ab, A_ak=A_ak, v=v, chunk_size=BT
    )
    del A_ab
    h, v_new, _ = chunk_dplr_fwd_h(
        kg=kg,
        bg=bg,
        v=v,
        w=w,
        u=u,
        gk=gi,
        initial_state=initial_state,
        chunk_size=BT,
    )

    del u
    # ******* end of recomputation *******
    # A_ak, A_ab_inv, gi, ge torch.float32
    # A_qk, A_qb, qg, kg, ag, bg, v_new dtype=q.dtype, eg: bf16

    dv_new_intra, dA_qk, dA_qb = chunk_dplr_bwd_dAu(
        v=v,
        v_new=v_new,
        do=do,
        A_qb=A_qb,
        scale=scale,
        chunk_size=BT,
    )
    return v, v_new, do, A_qb, dv_new_intra, dA_qk, dA_qb
    dh, dh0, dv_new = chunk_dplr_bwd_dhu(
        qg=qg,
        bg=bg,
        w=w,
        gk=gi,
        h0=initial_state,
        dht=dht,
        do=do,
        dv=dv_new_intra,
        chunk_size=BT,
    )

    dv = chunk_dplr_bwd_dv(A_qk=A_qk, kg=kg, do=do, dh=dh, chunk_size=BT)
    del A_qk

    dqg, dkg, dw, dbg, dgk_last = chunk_dplr_bwd_o(
        k=kg,
        b=bg,
        v=v,
        v_new=v_new,
        do=do,
        h=h,
        dh=dh,
        dv=dv_new,
        w=w,
        gk=gi,
        chunk_size=BT,
        scale=scale,
    )
    del v_new

    dA_ab, dA_ak, dv, dag = chunk_dplr_bwd_wy(
        A_ab_inv=A_ab_inv,
        A_ak=A_ak,
        v=v,
        ag=ag,
        dw=dw,
        du=dv_new,
        dv0=dv,
        chunk_size=BT,
    )
    del A_ak

    dq, dk, da, db, dgk = chunk_dplr_bwd_dqk_intra(
        q=q,
        k=k,
        a=a,
        b=b,
        gi=gi,
        ge=ge,
        dAqk=dA_qk,
        dAqb=dA_qb,
        dAak=dA_ak,
        dAab=dA_ab,
        dgk_last=dgk_last,
        dqg=dqg,
        dkg=dkg,
        dag=dag,
        dbg=dbg,
        chunk_size=BT,
        scale=scale,
    )

    return (
        ops.cast(dq, DTYPE),
        ops.cast(dk, DTYPE),
        ops.cast(dv, DTYPE),
        ops.cast(da, DTYPE),
        ops.cast(db, DTYPE),
        ops.cast(dgk, DTYPE),
        ops.cast(dh0, DTYPE),
    )


def chunk_dplr_bwd_jax(res, g):
    q, k, v, a, b, gk, initial_state = res
    do, dht = g
    dq, dk, dv, da, db, dgk, dh0 = chunk_dplr_bwd(
        q,
        k,
        v,
        a,
        b,
        gk,
        initial_state,
        scale=1,
        do=do,
        dht=dht,
    )

    return (
        dq,
        dk,
        dv,
        da,
        db,
        dgk,
        dh0,
    )


chunk_dplr.defvjp(chunk_dplr_fwd_jax, chunk_dplr_bwd_jax)


def transpose_head(x, head_first):
    if head_first:
        return jnp.transpose(x, (0, 2, 1, 3))
    else:
        return x


# @partial(jax.jit, static_argnames=['initial_state',"output_final_state","head_first","use_chunk"])
def generalized_delta_rule(
    r: jax.Array,
    w: jax.Array,
    k: jax.Array,
    v: jax.Array,
    a: jax.Array,
    b: jax.Array,
    initial_state: jax.Array = None,
    output_final_state: bool = True,
    head_first: bool = False,
):
    """
    Args:
        r (jax.Array):
            r of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
        k (jax.Array):
            k of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
        v (jax.Array):
            v of shape `[B, H, T, V]` if `head_first=True` else `[B, T, H, V]`.
        a (jax.Array):
            a of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
        b (jax.Array):
            b of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
        w (jax.Array):
            decay of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`, kernel
            will apply log_w = -torch.exp(w)
        initial_state (Optional[jax.Array]):
            Initial state of shape `[N, H, K, V]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N, H, K, V]`. Default: `False`.
        head_first (bool):
            whether to use head first. Recommended to be False to avoid extra transposes.
    """
    r = transpose_head(r, head_first)
    k = transpose_head(k, head_first)
    v = transpose_head(v, head_first)
    a = transpose_head(a, head_first)
    b = transpose_head(b, head_first)

    if w is not None:
        log_w = cal_log_w(w)
    else:
        assert log_w is not None, "Either w or log_w must be provided!"
    log_w = transpose_head(log_w, head_first)
    o, final_state = chunk_dplr(
        r=r,
        k=k,
        v=v,
        a=a,
        b=b,
        gk=log_w,
        initial_state=initial_state,
    )
    if output_final_state:
        return o, final_state
    return o
