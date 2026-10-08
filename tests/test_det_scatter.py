"""Tests of the deterministic scatter of the mass fold (research release, 2026-10-07)."""
import hashlib

import numpy as np
import pytest
import torch

from kvfold.merging_any_press import KeptSet, MassFold, scatter_add_atomic, scatter_add_sorted


def _case(seed, bsz=1, heads=3, l0=257, l1=19, d=8):
    g = torch.Generator().manual_seed(seed)
    p = torch.rand(l1, generator=g) ** 3
    p = p / p.sum()
    idx = torch.multinomial(p.expand(bsz * heads, l1), l0, replacement=True, generator=g).view(bsz, heads, l0)
    w = torch.exp(torch.randn(bsz, heads, l0, generator=g) * 2.0 - 4.0)
    vals = torch.randn(bsz, heads, l0, d, generator=g).to(torch.bfloat16).float()
    return idx, w, vals, l1


def _sequential(index, src, size):
    out = np.zeros((*index.shape[:2], size, *src.shape[3:]), dtype=np.float32)
    idx, s = index.numpy(), src.numpy()
    for b in range(idx.shape[0]):
        for h in range(idx.shape[1]):
            for j in range(idx.shape[2]):
                out[b, h, idx[b, h, j]] += s[b, h, j]
    return torch.from_numpy(out)


@pytest.mark.parametrize("seed", range(4))
def test_sorted_equals_sequential_reference_on_cpu(seed):
    idx, w, vals, l1 = _case(seed)
    for src in (w.unsqueeze(-1) * vals, w):
        ref = _sequential(idx, src, l1)
        assert torch.equal(scatter_add_sorted(idx, src, l1), ref)
        assert torch.equal(scatter_add_atomic(idx, src, l1), ref)


def test_sorted_is_repeatable():
    idx, w, vals, l1 = _case(5)
    src = w.unsqueeze(-1) * vals
    first = scatter_add_sorted(idx, src, l1)
    assert all(torch.equal(first, scatter_add_sorted(idx, src, l1)) for _ in range(5))


def test_empty_targets_are_zero():
    idx = torch.zeros(1, 1, 4, dtype=torch.long)
    src = torch.ones(1, 1, 4)
    out = scatter_add_sorted(idx, src, 6)
    assert out[0, 0, 0] == 4 and float(out[0, 0, 1:].abs().sum()) == 0.0


def _keptset(seed, dtype, bsz=1, heads=2, l0=512, l1=48, d=16, n_sink=4):
    g = torch.Generator().manual_seed(seed)
    keys, values = torch.randn(bsz, heads, l0, d, generator=g), torch.randn(bsz, heads, l0, d, generator=g)
    scores = torch.rand(bsz, heads, l0, generator=g) ** 2 + 1e-3
    src = torch.empty(bsz, heads, l1, dtype=torch.long)
    for b in range(bsz):
        for h in range(heads):
            top = torch.topk(scores[b, h, n_sink:], l1 - n_sink).indices + n_sink
            src[b, h] = torch.cat([torch.arange(n_sink), top.sort().values])
    gi = src.unsqueeze(-1).expand(-1, -1, -1, d)
    evicted = torch.ones(bsz, heads, l0, dtype=torch.bool)
    evicted.scatter_(2, src, False)
    return KeptSet(layer_idx=0, module=None, keys=keys.to(dtype), values=values.to(dtype), cache_keys=torch.gather(keys, 2, gi).to(dtype),
                   cache_values=torch.gather(values, 2, gi).to(dtype), src=src, kept=torch.ones(bsz, heads, l1, dtype=torch.bool), target=src >= n_sink,
                   evicted=evicted, path="in-place", scores=scores)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_massfold_sorted_matches_atomic_on_cpu(dtype):
    ks = _keptset(3, dtype)
    a = MassFold(score_map="log", scatter="atomic")(ks)
    s = MassFold(score_map="log", scatter="sorted")(ks)
    assert torch.equal(a.values, s.values) and torch.equal(a.logit_bias, s.logit_bias)
    assert int((a.logit_bias != 0).sum()) > 0


def test_massfold_default_is_atomic_and_rejects_unknown():
    assert MassFold().scatter == "atomic"
    with pytest.raises(AssertionError):
        MassFold(scatter="bogus")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_sorted_is_repeatable_on_cuda():
    idx, w, vals, l1 = _case(1, heads=8, l0=4096, l1=256, d=128)
    idx, src = idx.cuda(), (w.unsqueeze(-1) * vals).cuda()
    hs = {hashlib.sha1(scatter_add_sorted(idx, src, l1).cpu().numpy().tobytes()).hexdigest() for _ in range(20)}
    assert len(hs) == 1
