from typing import Optional
from typing import Tuple

import torch
import triton

from ops.triton_kernel.wy_fast_fwd import fwd_prepare_wy_repr_kernel_chunk32
from ops.triton_kernel.wy_fast_fwd import fwd_prepare_wy_repr_kernel_chunk64
from ops.triton_kernel.wy_fast_fwd import fwd_wu_kernel


def fwd_prepare_wy_repr(
    ag: torch.Tensor,
    v: torch.Tensor,
    A_ak: torch.Tensor,
    A_ab: torch.Tensor,
    offsets: Optional[torch.LongTensor],
    indices: Optional[torch.LongTensor],
    head_first: bool = True,
    chunk_size: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if head_first:
        B, H, T, K = ag.shape
    else:
        B, T, H, K = ag.shape
    BT = min(chunk_size, max(triton.next_power_of_2(T), 16))

    if offsets is None:
        NT = triton.cdiv(T, BT)
    else:
        if indices is None:
            indices = torch.cat(
                [
                    torch.arange(n)
                    for n in triton.cdiv(
                        offsets[1:] - offsets[:-1], BT
                    ).tolist()
                ]
            )
            indices = torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(
                offsets
            )
        NT = len(indices)
    BC = min(BT, 32)
    fwd_fn = (
        fwd_prepare_wy_repr_kernel_chunk64
        if BT == 64
        else fwd_prepare_wy_repr_kernel_chunk32
    )
    A_ab_inv = torch.empty_like(A_ab)
    fwd_fn[(NT, B * H)](
        A_ab=A_ab,
        A_ab_inv=A_ab_inv,
        offsets=offsets,
        indices=indices,
        T=T,
        H=H,
        BT=BT,
        BC=BC,
        HEAD_FIRST=head_first,
    )
    w, u = fwd_wu(
        ag=ag,
        v=v,
        A_ak=A_ak,
        A_ab_inv=A_ab_inv,
        offsets=offsets,
        indices=indices,
        head_first=head_first,
        chunk_size=BT,
    )
    return w, u, A_ab_inv


def fwd_wu(
    ag: torch.Tensor,
    v: torch.Tensor,
    A_ak: torch.Tensor,
    A_ab_inv: torch.Tensor,
    offsets: Optional[torch.LongTensor],
    indices: Optional[torch.LongTensor],
    head_first: bool,
    chunk_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if head_first:
        B, H, T, K, V = *ag.shape, v.shape[-1]
    else:
        B, T, H, K, V = *ag.shape, v.shape[-1]
    BT = min(chunk_size, max(triton.next_power_of_2(T), 16))

    if offsets is None:
        NT = triton.cdiv(T, BT)
    else:
        if indices is None:
            indices = torch.cat(
                [
                    torch.arange(n)
                    for n in triton.cdiv(
                        offsets[1:] - offsets[:-1], BT
                    ).tolist()
                ]
            )
            indices = torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(
                offsets
            )
        NT = len(indices)
    BK = min(triton.next_power_of_2(K), 64)
    BV = min(triton.next_power_of_2(V), 64)

    u = torch.empty_like(v)
    w = torch.empty_like(ag)
    fwd_wu_kernel[(NT, B * H)](
        ag=ag,
        v=v,
        A_ak=A_ak,
        A_ab_inv=A_ab_inv,
        w=w,
        u=u,
        offsets=offsets,
        indices=indices,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
        HEAD_FIRST=head_first,
    )
    return w, u
