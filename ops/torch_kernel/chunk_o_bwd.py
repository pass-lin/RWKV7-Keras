# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from typing import Optional, Tuple

import torch
import triton

from ops.get_torch_devices_info import prepare_chunk_indices, check_shared_mem
from ops.triton_kernel.chunk_o_bwd import *


def chunk_dplr_bwd_dv(
    A_qk: torch.Tensor,
    kg: torch.Tensor,
    do: torch.Tensor,
    dh: torch.Tensor,
    cu_seqlens: Optional[torch.LongTensor] = None,
    chunk_size: int = 64,
) -> torch.Tensor:
    B, T, H, K, V = *kg.shape, do.shape[-1]
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))

    chunk_indices = (
        prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    )
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    dv = torch.empty_like(do)

    def grid(meta):
        return (triton.cdiv(V, meta["BV"]), NT, B * H)

    chunk_dplr_bwd_kernel_dv[grid](
        A_qk=A_qk,
        kg=kg,
        do=do,
        dv=dv,
        dh=dh,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
    )
    return dv


def chunk_dplr_bwd_dv(
    A_qk: torch.Tensor,
    kg: torch.Tensor,
    do: torch.Tensor,
    dh: torch.Tensor,
    cu_seqlens: Optional[torch.LongTensor] = None,
    chunk_size: int = 64,
) -> torch.Tensor:
    B, T, H, K, V = *kg.shape, do.shape[-1]
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))

    chunk_indices = (
        prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    )
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    dv = torch.empty_like(do)

    def grid(meta):
        return (triton.cdiv(V, meta["BV"]), NT, B * H)

    chunk_dplr_bwd_kernel_dv[grid](
        A_qk=A_qk,
        kg=kg,
        do=do,
        dv=dv,
        dh=dh,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
    )
    return dv


def chunk_dplr_bwd_o(
    k: torch.Tensor,
    b: torch.Tensor,
    v: torch.Tensor,
    v_new: torch.Tensor,
    gk: torch.Tensor,
    do: torch.Tensor,
    h: torch.Tensor,
    dh: torch.Tensor,
    dv: torch.Tensor,
    w: torch.Tensor,
    cu_seqlens: Optional[torch.LongTensor] = None,
    chunk_size: int = 64,
    scale: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *w.shape, v.shape[-1]

    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    chunk_indices = (
        prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    )
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    BK = (
        min(triton.next_power_of_2(K), 64)
        if check_shared_mem()
        else min(triton.next_power_of_2(K), 32)
    )
    BV = (
        min(triton.next_power_of_2(V), 64)
        if check_shared_mem()
        else min(triton.next_power_of_2(K), 32)
    )
    NK = triton.cdiv(K, BK)
    dq = torch.empty_like(k)
    dk = torch.empty_like(k)
    dw = torch.empty_like(w)
    db = torch.empty_like(b)
    grid = (NK, NT, B * H)

    dgk_last = torch.empty(B, NT, H, K, dtype=torch.float, device=w.device)

    chunk_dplr_bwd_o_kernel[grid](
        k=k,
        b=b,
        v=v,
        v_new=v_new,
        h=h,
        do=do,
        dh=dh,
        dq=dq,
        dk=dk,
        db=db,
        dgk_last=dgk_last,
        w=w,
        dv=dv,
        dw=dw,
        gk=gk,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
    )
    return dq, dk, dw, db, dgk_last


def chunk_dplr_bwd_dAu(
    v: torch.Tensor,
    v_new: torch.Tensor,
    do: torch.Tensor,
    A_qb: torch.Tensor,
    scale: float,
    cu_seqlens: Optional[torch.LongTensor] = None,
    chunk_size: int = 64,
) -> torch.Tensor:
    B, T, H, V = v.shape
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    chunk_indices = (
        prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    )
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    if check_shared_mem("ampere"):  # A100
        BV = min(triton.next_power_of_2(V), 128)
    elif check_shared_mem("ada"):  # 4090
        BV = min(triton.next_power_of_2(V), 64)
    else:
        BV = min(triton.next_power_of_2(V), 32)

    grid = (NT, B * H)
    dA_qk = torch.empty(B, T, H, BT, dtype=torch.float, device=v.device)
    dA_qb = torch.empty(B, T, H, BT, dtype=torch.float, device=v.device)
    dv_new = torch.empty_like(v_new)
    chunk_dplr_bwd_kernel_dAu[grid](
        v=v,
        do=do,
        v_new=v_new,
        A_qb=A_qb,
        dA_qk=dA_qk,
        dA_qb=dA_qb,
        dv_new=dv_new,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        T=T,
        H=H,
        V=V,
        BT=BT,
        BV=BV,
    )
    return dv_new, dA_qk, dA_qb
