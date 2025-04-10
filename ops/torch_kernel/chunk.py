# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from typing import Optional

import torch
import triton

from ops.torch_kernel.chunk_A_fwd import chunk_fwd_intra_dplr_fn
from ops.torch_kernel.chunk_h_fwd import chunk_dplr_fwd_h
from ops.torch_kernel.chunk_o_fwd import chunk_dplr_fwd_o
from ops.torch_kernel.wy_fast_fwd import fwd_prepare_wy_repr
from ops.torch_kernel.chunk_A_bwd import chunk_dplr_bwd_dqk_intra
from ops.torch_kernel.chunk_h_bwd import chunk_dplr_bwd_dhu
from ops.torch_kernel.chunk_o_bwd import chunk_dplr_bwd_dAu
from ops.torch_kernel.chunk_o_bwd import chunk_dplr_bwd_dv
from ops.torch_kernel.chunk_o_bwd import chunk_dplr_bwd_o
from ops.torch_kernel.wy_fast_bwd import chunk_dplr_bwd_wy
from ops.triton_kernel.chunk import chunk_rwkv6_fwd_cumsum_kernel


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
    chunk_size: int = 16,
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


def chunk_dplr_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    gk: torch.Tensor,
    initial_state,
    BT,
    head_first,
    scale,
    do: torch.Tensor,
    dht: torch.Tensor,
    offsets: Optional[torch.LongTensor] = None,
    indices: Optional[torch.LongTensor] = None,
):
    DTYPE = do.dtype
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
        dq.to(dtype=DTYPE),
        dk.to(dtype=DTYPE),
        dv.to(dtype=DTYPE),
        da.to(dtype=DTYPE),
        db.to(dtype=DTYPE),
        dgk.to(dtype=DTYPE),
        None,
        dh0,
        None,
        None,
        None,
    )
