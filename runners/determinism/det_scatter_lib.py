"""Deterministic scatter for the mass fold: fixed-order segment sums.

scatter_add_atomic is the accumulation the fold has used so far: zeros.scatter_add_(2, index, src).  On CUDA it adds in float32 with
atomics, so the summation order, and with it the low bits of the folded values and of the logit bias, can change from run to run
unless torch.use_deterministic_algorithms(True) is on.
scatter_add_sorted computes the same sums without atomics: a stable sort of the sources by (batch * head, target row) followed by a
segment reduction, so every target adds its sources in a fixed order.  It does not depend on a global flag.
index: (B, H, L0) int64 in [0, size); src: (B, H, L0) or (B, H, L0, D) float32; result: (B, H, size) or (B, H, size, D) float32.
"""
import torch


def scatter_add_atomic(index: torch.Tensor, src: torch.Tensor, size: int) -> torch.Tensor:
    assert index.dtype == torch.long and index.dim() == 3, f"index must be (B, H, L0) int64, got {index.dtype} {tuple(index.shape)}"
    assert src.dtype == torch.float32 and tuple(src.shape[:3]) == tuple(index.shape), f"src {src.dtype} {tuple(src.shape)} vs index {tuple(index.shape)}"
    bsz, heads, _ = index.shape
    if src.dim() == 3:
        return torch.zeros(bsz, heads, size, dtype=src.dtype, device=src.device).scatter_add_(2, index, src)
    d = src.shape[-1]
    return torch.zeros(bsz, heads, size, d, dtype=src.dtype, device=src.device).scatter_add_(2, index.unsqueeze(-1).expand(-1, -1, -1, d), src)


def scatter_add_sorted(index: torch.Tensor, src: torch.Tensor, size: int) -> torch.Tensor:
    assert index.dtype == torch.long and index.dim() == 3, f"index must be (B, H, L0) int64, got {index.dtype} {tuple(index.shape)}"
    assert src.dtype == torch.float32 and tuple(src.shape[:3]) == tuple(index.shape), f"src {src.dtype} {tuple(src.shape)} vs index {tuple(index.shape)}"
    bsz, heads, l0 = index.shape
    trailing = tuple(src.shape[3:])
    rows = torch.arange(bsz * heads, device=index.device).view(bsz, heads, 1)
    seg = (rows * size + index).reshape(-1)  # segment id of every source row, in [0, B * H * size)
    order = torch.sort(seg, stable=True).indices  # equal ids keep their source order (ascending position)
    seg_sorted = seg[order]
    data = src.reshape(bsz * heads * l0, -1)[order].contiguous()
    nseg = bsz * heads * size
    bounds = torch.searchsorted(seg_sorted, torch.arange(nseg + 1, device=index.device))
    lengths = bounds[1:] - bounds[:-1]
    out = torch.segment_reduce(data, "sum", lengths=lengths, axis=0)
    return out.reshape(bsz, heads, size, *trailing)
