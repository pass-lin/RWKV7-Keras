# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from typing import Optional

import torch
import triton
from ops.triton_kernel.chunk_A_bwd import *
from ops.triton_kernel.utils import is_gather_supported
from ops.get_torch_devices_info import check_shared_mem, prepare_chunk_indices


def chunk_dplr_bwd_dqk_intra(
    q: torch.Tensor,
    k: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    gi: torch.Tensor,
    ge: torch.Tensor,
    dAqk: torch.Tensor,
    dAqb: torch.Tensor,
    dAak: torch.Tensor,
    dAab: torch.Tensor,
    dqg: torch.Tensor,
    dkg: torch.Tensor,
    dag: torch.Tensor,
    dbg: torch.Tensor,
    dgk_last: torch.Tensor,
    scale: float = 1.0,
    cu_seqlens: Optional[torch.LongTensor] = None,
    chunk_size: int = 64,
):
    B, T, H, K = q.shape
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    BK = (
        min(64, triton.next_power_of_2(K))
        if check_shared_mem()
        else min(32, triton.next_power_of_2(K))
    )

    chunk_indices = (
        prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    )
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    NK = triton.cdiv(K, BK)

    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    da = torch.empty_like(a)
    db = torch.empty_like(b)
    dgk = torch.empty_like(gi, dtype=torch.float)
    dgk_offset = torch.empty_like(gi, dtype=torch.float)

    grid = (NK, NT, B * H)
    chunk_dplr_bwd_kernel_intra[grid](
        q=q,
        k=k,
        a=a,
        b=b,
        gi=gi,
        ge=ge,
        dAqk=dAqk,
        dAqb=dAqb,
        dAak=dAak,
        dAab=dAab,
        dq=dq,
        dk=dk,
        dgk=dgk,
        dgk_offset=dgk_offset,
        dqg=dqg,
        dkg=dkg,
        dag=dag,
        dbg=dbg,
        da=da,
        db=db,
        scale=scale,
        T=T,
        H=H,
        K=K,
        BT=BT,
        BC=BT,
        BK=BK,
        GATHER_SUPPORTED=is_gather_supported,
    )
    dgk_output = torch.empty_like(dgk)

    def grid(meta):
        return (NT, triton.cdiv(K, meta["BK"]), B * H)

    chunk_dplr_bwd_dgk_kernel[grid](
        dgk=dgk,
        dgk_offset=dgk_offset,
        dgk_last=dgk_last,
        T=T,
        dgk_output=dgk_output,
        H=H,
        K=K,
        BT=BT,
    )
    return dq, dk, da, db, dgk_output
