from ops.triton_kernel.cumsum import *
import torch
from ops.get_torch_devices_info import prepare_chunk_indices


def chunk_rwkv6_fwd_cumsum(
    g: torch.Tensor,
    chunk_size: int,
    cu_seqlens=None,
) -> torch.Tensor:
    B, T, H, S = g.shape
    BT = chunk_size
    chunk_indices = (
        prepare_chunk_indices(cu_seqlens, chunk_size)
        if cu_seqlens is not None
        else None
    )
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    gi, ge = (
        torch.empty_like(g, dtype=torch.float),
        torch.empty_like(g, dtype=torch.float),
    )

    def grid(meta):
        return (triton.cdiv(meta["S"], meta["BS"]), NT, B * H)

    # keep cummulative normalizer in fp32
    chunk_rwkv6_fwd_cumsum_kernel[grid](
        g,
        T,
        gi,
        ge,
        H=H,
        S=S,
        BT=BT,
    )
    return gi, ge
