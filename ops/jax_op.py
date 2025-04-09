from ops.jax_kernel.chunk import *
def chunk_dplr_delta_rule(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    a: jax.Array,
    b: jax.Array,
    gk: jax.Array,
    scale: Optional[float] = None,
    initial_state: Optional[jax.Array] = None,
    output_final_state: bool = False,
    cu_seqlens = None,
    head_first: bool = False,
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
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.
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

    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                f"Please flatten variable-length inputs before processing."
            )
        if head_first:
            raise RuntimeError(
                "Sequences with variable lengths are not supported for head-first mode"
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}."
            )
    scale = k.shape[-1] ** -0.5 if scale is None else scale
    chunk_size = 16

    o, final_state = chunk_dplr_fwd(
        q,
        k,
        v,
        a,
        b,
        gk,
        scale,
        initial_state,
        output_final_state,
        chunk_size=chunk_size,
        head_first=head_first,
    )
    return o, final_state
@jax.jit
def cal_log_w(w: jax.Array) -> jax.Array:
    return -jnp.exp(w)
def chunk_rwkv7(
    r: jax.Array,
    k: jax.Array,
    v: jax.Array,
    a: jax.Array,
    b: jax.Array,
    w: jax.Array = None,
    log_w: jax.Array = None,
    scale: float = 1.0,
    initial_state: jax.Array = None,
    output_final_state: bool = True,
    cu_seqlens  = None,
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
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.
        head_first (bool):
            whether to use head first. Recommended to be False to avoid extra transposes.
    """

    if w is not None:
        log_w = cal_log_w(w)
    else:
        assert log_w is not None, "Either w or log_w must be provided!"
    scale = k.shape[-1] ** -0.5 if scale is None else scale
    return chunk_dplr_delta_rule(
        q=r,
        k=k,
        v=v,
        a=a,
        b=b,
        gk=log_w,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        head_first=head_first,
    )
