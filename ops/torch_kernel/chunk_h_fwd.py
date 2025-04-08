# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from typing import Optional
from typing import Tuple

import torch
import triton

from ops.get_torch_devices_info import is_triton_shared_mem_enough
from ops.torch_kernel.utils import prepare_chunk_offsets
from ops.triton_kernel.chunk_h_fwd import chunk_dplr_fwd_kernel_h


def chunk_dplr_fwd_h(
    kg: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    bg: torch.Tensor,
    gk: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    offsets: Optional[torch.LongTensor] = None,
    head_first: bool = True,
    chunk_size: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if head_first:
        B, H, T, K, V = *kg.shape, u.shape[-1]
    else:
        B, T, H, K, V = *kg.shape, u.shape[-1]
    BT = min(chunk_size, max(triton.next_power_of_2(T), 16))
    # N: the actual number of sequences in the batch with either equal or variable lengths
    if offsets is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        N = len(offsets) - 1
        chunk_offsets = prepare_chunk_offsets(offsets, BT)
        NT = chunk_offsets[-1]
    BK = triton.next_power_of_2(K)
    assert BK <= 256, "current kernel does not support head dimension larger than 256."
    # H100 can have larger block size

    if is_triton_shared_mem_enough(233472, kg.device.index):
        BV = 64
        BC = 64 if K <= 128 else 32
    elif is_triton_shared_mem_enough(131072, kg.device.index):  # A100
        BV = 32
        BC = 32
    else:
        BV = 16
        BC = 16

    BC = min(BT, BC)
    NK = triton.cdiv(K, BK)
    NV = triton.cdiv(V, BV)
    assert NK == 1, (
        "NK > 1 is not supported because it involves time-consuming synchronization"
    )

    if head_first:
        h = kg.new_empty(B, H, NT, K, V)
    else:
        h = kg.new_empty(B, NT, H, K, V)
    final_state = (
        kg.new_empty(N, H, K, V, dtype=torch.float32) if output_final_state else None
    )
    v_new = torch.empty_like(u)
    grid = (NK, NV, N * H)
    chunk_dplr_fwd_kernel_h[grid](
        kg=kg,
        v=v,
        w=w,
        bg=bg,
        u=u,
        v_new=v_new,
        h=h,
        gk=gk,
        h0=initial_state,
        ht=final_state,
        offsets=offsets,
        chunk_offsets=chunk_offsets,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BC=BC,
        BK=BK,
        BV=BV,
        NT=NT,
        HEAD_FIRST=head_first,
    )
    return h, v_new, final_state
