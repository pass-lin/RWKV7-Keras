# -*- coding: utf-8 -*-
# Copyright (c) 2024-2025, Songlin Yang, Yu Zhang

from typing import Optional
from typing import Tuple

import torch
import triton

from ops.get_torch_devices_info import autocast_custom_bwd
from ops.get_torch_devices_info import autocast_custom_fwd
from ops.torch_kernel.chunk import chunk_dplr_fwd, chunk_dplr_bwd

from ops.torch_kernel.fused_recurrent import fused_recurrent_dplr_delta_rule
from ops.torch_kernel.utils import input_guard
from ops.torch_kernel.utils import prepare_chunk_indices
from ops.triton_kernel.fuse_rwkv import fused_rwkv7_kernel


class ChunkDPLRDeltaRuleFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        gk: torch.Tensor,
        scale: float,
        initial_state: torch.Tensor,
        output_final_state: bool,
        offsets: Optional[torch.LongTensor] = None,
        head_first: bool = True,
    ):
        chunk_size = 16

        # 2-d indices denoting the offsets of chunks in each sequence
        # for example, if the passed `offsets` is [0, 100, 356] and `chunk_size` is 64,
        # then there are 2 and 4 chunks in the 1st and 2nd sequences respectively, and `indices` will be
        # [[0, 0], [0, 1], [1, 0], [1, 1], [1, 2], [1, 3]]
        indices = (
            prepare_chunk_indices(offsets, chunk_size) if offsets is not None else None
        )

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
            offsets=offsets,
            indices=indices,
            head_first=head_first,
            chunk_size=chunk_size,
        )
        ctx.save_for_backward(q, k, v, a, b, gk, initial_state)
        ctx.head_first = head_first
        ctx.offsets = offsets
        ctx.indices = indices
        ctx.scale = scale
        ctx.chunk_size = chunk_size
        return o.to(q.dtype), final_state

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do: torch.Tensor, dht: torch.Tensor):
        q, k, v, a, b, gk, initial_state = ctx.saved_tensors
        BT = ctx.chunk_size
        head_first = ctx.head_first
        offsets = ctx.offsets
        indices = ctx.indices
        scale = ctx.scale
        return chunk_dplr_bwd(
            q=q,
            k=k,
            v=v,
            a=a,
            b=b,
            gk=gk,
            initial_state=initial_state,
            BT=BT,
            head_first=head_first,
            scale=scale,
            do=do,
            dht=dht,
            offsets=offsets,
            indices=indices,
        )


@torch.compiler.disable
def chunk_dplr_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    gk: torch.Tensor,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
    head_first: bool = False,
):
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
        k (torch.Tensor):
            keys of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
        v (torch.Tensor):
            values of shape `[B, H, T, V]` if `head_first=True` else `[B, T, H, V]`.
        a (torch.Tensor):
            activations of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
        b (torch.Tensor):
            betas of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
        gk (torch.Tensor):
            gk of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`. decay term in log space!
        scale (Optional[int]):
            Scale factor for the RetNet attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
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
        o (torch.Tensor):
            Outputs of shape `[B, H, T, V]` if `head_first=True` else `[B, T, H, V]`.
        final_state (torch.Tensor):
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

    o, final_state = ChunkDPLRDeltaRuleFunction.apply(
        q,
        k,
        v,
        a,
        b,
        gk,
        scale,
        initial_state,
        output_final_state,
        cu_seqlens,
        head_first,
    )
    return o.to(q.dtype), final_state


@torch.compile(fullgraph=True)
def cal_log_w(w: torch.Tensor) -> torch.Tensor:
    return -torch.exp(w)


class FusedRecurrentRWKV7Function(torch.autograd.Function):
    @staticmethod
    @input_guard
    def forward(
        ctx,
        q,
        k,
        v,
        w,
        a,
        b,
        scale=None,
        initial_state=None,
        output_final_state=False,
        offsets=None,
        head_first=False,
    ):
        if head_first:
            B, H, L, K, V = *k.shape, v.shape[-1]
        else:
            B, L, H, K, V = *k.shape, v.shape[-1]
        N = B if offsets is None else len(offsets) - 1
        output = torch.empty_like(v)

        BK = triton.next_power_of_2(K)
        if initial_state is not None:
            final_state = torch.empty_like(initial_state)
            use_initial_state = True
        elif output_final_state:
            final_state = q.new_empty(B, H, K, V, dtype=torch.float32)
            use_initial_state = False
        else:
            final_state = None
            use_initial_state = False

        def grid(meta):
            return (triton.cdiv(V, meta["BV"]), N * H)

        fused_rwkv7_kernel[grid](
            q,
            k,
            v,
            w,
            a,
            b,
            initial_state,
            output,
            final_state,
            K,
            V,
            L,
            H,
            offsets=offsets,
            scale=scale,
            BK=BK,
            HEAD_FIRST=head_first,
            USE_INITIAL_STATE=use_initial_state,
            STORE_FINAL_STATE=output_final_state,
        )
        return output, final_state

    @staticmethod
    @input_guard
    def backward(ctx, do, dht):
        raise NotImplementedError(
            "Fused wkv7 backward function is not implemented. "
            "Please use chunk_rwkv7 for training!"
        )


def fused_recurrent_rwkv7(
    r: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor = None,
    log_w: torch.Tensor = None,
    scale: float = 1.0,
    initial_state: torch.Tensor = None,
    output_final_state: bool = True,
    cu_seqlens: Optional[torch.LongTensor] = None,
    head_first: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        r (torch.Tensor):
            r of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
        k (torch.Tensor):
            k of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
        v (torch.Tensor):
            v of shape `[B, H, T, V]` if `head_first=True` else `[B, T, H, V]`.
        a (torch.Tensor):
            a of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
        b (torch.Tensor):
            b of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
        w (torch.Tensor):
            decay of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`, kernel
            will apply log_w = -torch.exp(w)
        log_w (torch.Tensor):
            log decay of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
        scale (float):
            scale of the attention.
        initial_state (Optional[torch.Tensor]):
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
        if cu_seqlens is not None:
            if r.shape[0] != 1:
                raise ValueError(
                    f"The batch size is expected to be 1 rather than {r.shape[0]} when using `cu_seqlens`."
                    f"Please flatten variable-length inputs before processing."
                )
            if head_first:
                raise RuntimeError(
                    "Sequences with variable lengths are not supported for head-first mode"
                )
            if (
                initial_state is not None
                and initial_state.shape[0] != len(cu_seqlens) - 1
            ):
                raise ValueError(
                    f"The number of initial states is expected to be equal to the number of input sequences, "
                    f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}."
                )
        if scale is None:
            scale = r.shape[-1] ** -0.5
        else:
            assert scale > 0, "scale must be positive"
        o, final_state = FusedRecurrentRWKV7Function.apply(
            r,
            k,
            v,
            w,
            a,
            b,
            scale,
            initial_state,
            output_final_state,
            cu_seqlens,
            head_first,
        )
        return o, final_state
    elif log_w is not None:
        return fused_recurrent_dplr_delta_rule(
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
    else:
        raise ValueError("Either `w` or `log_w` must be provided.")


def chunk_rwkv7(
    r: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor = None,
    log_w: torch.Tensor = None,
    scale: float = 1.0,
    initial_state: torch.Tensor = None,
    output_final_state: bool = True,
    cu_seqlens: Optional[torch.LongTensor] = None,
    head_first: bool = False,
):
    """
    Args:
        r (torch.Tensor):
            r of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
        k (torch.Tensor):
            k of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
        v (torch.Tensor):
            v of shape `[B, H, T, V]` if `head_first=True` else `[B, T, H, V]`.
        a (torch.Tensor):
            a of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
        b (torch.Tensor):
            b of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
        w (torch.Tensor):
            decay of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`, kernel
            will apply log_w = -torch.exp(w)
        log_w (torch.Tensor):
            log decay of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
        scale (float):
            scale of the attention.
        initial_state (Optional[torch.Tensor]):
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


def generalized_delta_rule(
    r: torch.Tensor,
    w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    initial_state: torch.Tensor = None,
    output_final_state: bool = True,
    head_first: bool = False,
    use_chunk: bool = True,
):
    """
    Args:
        r (torch.Tensor):
            r of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
        k (torch.Tensor):
            k of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
        v (torch.Tensor):
            v of shape `[B, H, T, V]` if `head_first=True` else `[B, T, H, V]`.
        a (torch.Tensor):
            a of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
        b (torch.Tensor):
            b of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
        w (torch.Tensor):
            decay of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`, kernel
            will apply log_w = -torch.exp(w)
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[N, H, K, V]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N, H, K, V]`. Default: `False`.
        head_first (bool):
            whether to use head first. Recommended to be False to avoid extra transposes.
    """
    if use_chunk:
        return chunk_rwkv7(
            r=r,
            k=k,
            v=v,
            a=a,
            b=b,
            w=w,
            initial_state=initial_state,
            output_final_state=output_final_state,
            head_first=head_first,
        )
    return fused_recurrent_rwkv7(
        r=r,
        k=k,
        v=v,
        a=a,
        b=b,
        w=w,
        initial_state=initial_state,
        output_final_state=output_final_state,
        head_first=head_first,
    )
