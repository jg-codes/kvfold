# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests of the MassFold hooks (exclude_targets, observer): default off, arithmetic unchanged; observer routing consistent with the fold result."""

import types

import torch

from kvfold.merging_any_press import FoldRouting, KeptSet, MassFold


def toy_keptset(seed: int, heads: int = 3, l0: int = 48, dim: int = 16, n_app: int = 2, keep_frac: float = 0.2, n_sink: int = 4) -> KeptSet:
    """KeptSet as MergingAnyPress builds it for a mask-based press with appended (restore) rows."""
    g = torch.Generator().manual_seed(seed)
    keys = torch.randn(1, heads, l0, dim, generator=g)
    keys = keys + 0.5 * torch.roll(keys, 1, dims=2)
    values = torch.randn(1, heads, l0, dim, generator=g)
    cache_keys = torch.cat([keys, torch.randn(1, heads, n_app, dim, generator=g)], 2)
    cache_values = torch.cat([values, torch.randn(1, heads, n_app, dim, generator=g)], 2)
    src = torch.cat([torch.arange(l0).expand(1, heads, l0), torch.full((1, heads, n_app), -1)], 2).clone()
    masked = torch.rand(1, heads, l0, generator=g) > keep_frac
    masked[..., :n_sink] = False
    kept = torch.cat([~masked, torch.zeros(1, heads, n_app, dtype=torch.bool)], 2)
    return KeptSet(
        layer_idx=0,
        module=types.SimpleNamespace(layer_idx=0),
        keys=keys,
        values=values,
        cache_keys=cache_keys,
        cache_values=cache_values,
        src=src,
        kept=kept,
        target=kept & (src >= n_sink),
        evicted=masked.clone(),
        path="in-place+appended+mask",
        scores=torch.rand(1, heads, l0, generator=g) * 0.9 + 0.01,
    )


def test_hooks_do_not_change_the_fold():
    ks = toy_keptset(0)
    plain = MassFold(score_map="log")(ks)
    seen = []
    hooked = MassFold(score_map="log", observer=lambda k, r: seen.append(r), exclude_targets=lambda k: torch.zeros_like(k.target))(ks)
    assert torch.equal(plain.values, hooked.values)
    assert torch.equal(plain.logit_bias, hooked.logit_bias)
    assert plain.info == hooked.info
    assert len(seen) == 1 and isinstance(seen[0], FoldRouting)


def test_observer_routing_is_consistent_with_the_result():
    ks = toy_keptset(1, keep_frac=0.15)
    seen = []
    res = MassFold(score_map="log", observer=lambda k, r: seen.append(r))(ks)
    r = seen[0]
    assert r.layer_idx == 0 and r.target_idx.shape == r.cosine.shape == r.valid.shape == r.weight.shape == ks.evicted.shape
    assert torch.equal(r.valid, ks.evicted & ks.target.any(-1, keepdim=True))
    assert bool(ks.target.gather(2, r.target_idx)[r.valid].all()), "an evicted row was routed to a non-target row"
    den = torch.zeros_like(r.den).scatter_add_(2, r.target_idx, r.weight)
    assert torch.allclose(den, r.den, rtol=1e-5, atol=1e-6)
    assert torch.allclose(res.logit_bias, torch.log1p(r.den))
    assert int(res.info["merged"]) == int(r.valid.sum())
    assert bool((r.cosine[r.valid] <= 1 + 1e-5).all())


def test_exclude_targets_removes_rows_from_the_target_set():
    ks = toy_keptset(2, l0=64, keep_frac=0.25)
    excl_rows = (ks.src >= 6) & (ks.src < 20)
    seen = []
    res = MassFold(score_map="log", exclude_targets=lambda k: excl_rows & k.target, observer=lambda k, r: seen.append(r))(ks)
    assert not bool(((res.logit_bias != 0) & excl_rows).any()), "bias on an excluded row"
    assert torch.equal(res.values[excl_rows], ks.cache_values[excl_rows]), "value written on an excluded row"
    assert torch.equal(seen[0].target, ks.target & ~excl_rows)
    assert not bool(excl_rows.gather(2, seen[0].target_idx)[seen[0].valid].any()), "an evicted row was routed to an excluded row"
