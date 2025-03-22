# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from typing import Optional

import torch
import triton
import triton.language as tl

from ops.triton_kernel.chunk_A_bwd import chunk_dplr_bwd_dqk_intra
from ops.triton_kernel.chunk_A_fwd import chunk_fwd_intra_dplr_fn
from ops.triton_kernel.chunk_h_bwd import chunk_dplr_bwd_dhu
from ops.triton_kernel.chunk_h_fwd import chunk_dplr_fwd_h
from ops.triton_kernel.chunk_o_bwd import chunk_dplr_bwd_dAu
from ops.triton_kernel.chunk_o_bwd import chunk_dplr_bwd_dv
from ops.triton_kernel.chunk_o_bwd import chunk_dplr_bwd_o
from ops.triton_kernel.chunk_o_fwd import chunk_dplr_fwd_o
from ops.triton_kernel.utils import autocast_custom_bwd
from ops.triton_kernel.utils import autocast_custom_fwd
from ops.triton_kernel.utils import input_guard
from ops.triton_kernel.utils import prepare_chunk_indices
from ops.triton_kernel.wy_fast_bwd import chunk_dplr_bwd_wy
from ops.triton_kernel.wy_fast_fwd import fwd_prepare_wy_repr


@triton.heuristics({"USE_OFFSETS": lambda args: args["offsets"] is not None})
@triton.autotune(
    configs=[
        triton.Config({"BS": BS}, num_warps=num_warps, num_stages=num_stages)
        for BS in [16, 32, 64]
        for num_warps in [4, 8, 16, 32]
        for num_stages in [2, 3, 4]
    ],
    key=["S", "BT"],
    use_cuda_graph=0,
)
@triton.jit(do_not_specialize=["T"])
def chunk_rwkv6_fwd_cumsum_kernel(
    s,
    oi,
    oe,
    offsets,
    indices,
    T,
    H: tl.constexpr,
    S: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    HEAD_FIRST: tl.constexpr,
    USE_OFFSETS: tl.constexpr,
):
    i_s, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    if USE_OFFSETS:
        i_n, i_t = (
            tl.load(indices + i_t * 2).to(tl.int32),
            tl.load(indices + i_t * 2 + 1).to(tl.int32),
        )
        bos, eos = (
            tl.load(offsets + i_n).to(tl.int32),
            tl.load(offsets + i_n + 1).to(tl.int32),
        )
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    o_i = tl.arange(0, BT)
    m_i = tl.where(o_i[:, None] >= o_i[None, :], 1.0, 0.0).to(tl.float32)
    m_e = tl.where(o_i[:, None] > o_i[None, :], 1.0, 0.0).to(tl.float32)

    if HEAD_FIRST:
        p_s = tl.make_block_ptr(
            s + i_bh * T * S,
            (T, S),
            (S, 1),
            (i_t * BT, i_s * BS),
            (BT, BS),
            (1, 0),
        )
        p_oi = tl.make_block_ptr(
            oi + i_bh * T * S,
            (T, S),
            (S, 1),
            (i_t * BT, i_s * BS),
            (BT, BS),
            (1, 0),
        )
        p_oe = tl.make_block_ptr(
            oe + i_bh * T * S,
            (T, S),
            (S, 1),
            (i_t * BT, i_s * BS),
            (BT, BS),
            (1, 0),
        )
    else:
        p_s = tl.make_block_ptr(
            s + (bos * H + i_h) * S,
            (T, S),
            (H * S, 1),
            (i_t * BT, i_s * BS),
            (BT, BS),
            (1, 0),
        )
        p_oi = tl.make_block_ptr(
            oi + (bos * H + i_h) * S,
            (T, S),
            (H * S, 1),
            (i_t * BT, i_s * BS),
            (BT, BS),
            (1, 0),
        )
        p_oe = tl.make_block_ptr(
            oe + (bos * H + i_h) * S,
            (T, S),
            (H * S, 1),
            (i_t * BT, i_s * BS),
            (BT, BS),
            (1, 0),
        )
    # [BT, BS]
    b_s = tl.load(p_s, boundary_check=(0, 1)).to(tl.float32)
    b_oi = tl.dot(m_i, b_s)
    b_oe = tl.dot(m_e, b_s)
    tl.store(
        p_oi,
        b_oi.to(p_oi.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )
    tl.store(
        p_oe,
        b_oe.to(p_oe.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )


def chunk_rwkv6_fwd_cumsum(
    g: torch.Tensor,
    chunk_size: int,
    offsets: Optional[torch.Tensor] = None,
    indices: Optional[torch.Tensor] = None,
    head_first: bool = True,
) -> torch.Tensor:
    if head_first:
        B, H, T, S = g.shape
    else:
        B, T, H, S = g.shape
    BT = chunk_size
    NT = triton.cdiv(T, BT) if offsets is None else len(indices)

    gi, ge = (
        torch.empty_like(g, dtype=torch.float),
        torch.empty_like(g, dtype=torch.float),
    )

    def grid(meta):
        return (triton.cdiv(meta["S"], meta["BS"]), NT, B * H)

    # keep cummulative normalizer in fp32
    chunk_rwkv6_fwd_cumsum_kernel[grid](
        g, gi, ge, offsets, indices, T=T, H=H, S=S, BT=BT, HEAD_FIRST=head_first
    )
    return gi, ge


def chunk_dplr_fwd(
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
    indices: Optional[torch.LongTensor] = None,
    head_first: bool = True,
    chunk_size: int = 64,
):
    T = q.shape[2] if head_first else q.shape[1]
    BT = min(chunk_size, max(triton.next_power_of_2(T), 16))
    gi, ge = chunk_rwkv6_fwd_cumsum(
        gk, BT, offsets=offsets, indices=indices, head_first=head_first
    )

    A_ab, A_qk, A_ak, A_qb, qg, kg, ag, bg = chunk_fwd_intra_dplr_fn(
        q=q,
        k=k,
        a=a,
        b=b,
        gi=gi,
        ge=ge,
        scale=scale,
        offsets=offsets,
        indices=indices,
        chunk_size=BT,
        head_first=head_first,
    )
    del ge

    # A_ab, A_ak, gi, ge torch.float32
    # A_qk, A_qb, qg, kg, ag, bg, dtype=q.dtype, eg: bf16
    w, u, _ = fwd_prepare_wy_repr(
        ag=ag,
        A_ab=A_ab,
        A_ak=A_ak,
        v=v,
        offsets=offsets,
        indices=indices,
        head_first=head_first,
        chunk_size=BT,
    )
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
        offsets=offsets,
        head_first=head_first,
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
        offsets=offsets,
        indices=indices,
        head_first=head_first,
        chunk_size=BT,
    )
    del v_new, h, A_qk, A_qb

    return o, final_state


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
            prepare_chunk_indices(offsets, chunk_size)
            if offsets is not None
            else None
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

        # ******* start recomputing everything, otherwise i believe the gpu memory will be exhausted *******
        gi, ge = chunk_rwkv6_fwd_cumsum(
            gk, BT, offsets=offsets, indices=indices, head_first=head_first
        )

        A_ab, A_qk, A_ak, A_qb, qg, kg, ag, bg = chunk_fwd_intra_dplr_fn(
            q=q,
            k=k,
            a=a,
            b=b,
            gi=gi,
            ge=ge,
            scale=scale,
            offsets=offsets,
            indices=indices,
            chunk_size=BT,
            head_first=head_first,
        )
        w, u, A_ab_inv = fwd_prepare_wy_repr(
            ag=ag,
            A_ab=A_ab,
            A_ak=A_ak,
            v=v,
            offsets=offsets,
            indices=indices,
            head_first=head_first,
            chunk_size=BT,
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
            offsets=offsets,
            head_first=head_first,
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
            offsets=offsets,
            indices=indices,
            head_first=head_first,
            chunk_size=BT,
        )

        dh, dh0, dv_new = chunk_dplr_bwd_dhu(
            qg=qg,
            bg=bg,
            w=w,
            gk=gi,
            h0=initial_state,
            dht=dht,
            do=do,
            dv=dv_new_intra,
            offsets=offsets,
            head_first=head_first,
            chunk_size=BT,
        )

        dv = chunk_dplr_bwd_dv(
            A_qk=A_qk,
            kg=kg,
            do=do,
            dh=dh,
            offsets=offsets,
            indices=indices,
            head_first=head_first,
            chunk_size=BT,
        )
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
            offsets=offsets,
            indices=indices,
            chunk_size=BT,
            scale=scale,
            head_first=head_first,
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
            offsets=offsets,
            indices=indices,
            head_first=head_first,
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
            head_first=head_first,
            offsets=offsets,
            indices=indices,
        )

        return (
            dq.to(q),
            dk.to(k),
            dv.to(v),
            da.to(a),
            db.to(b),
            dgk.to(gk),
            None,
            dh0,
            None,
            None,
            None,
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
        if (
            initial_state is not None
            and initial_state.shape[0] != len(cu_seqlens) - 1
        ):
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
    return o, final_state
