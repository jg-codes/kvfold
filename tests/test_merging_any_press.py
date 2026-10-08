# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
import types

import pytest
import torch
from transformers import DynamicCache

from kvpress import AdaKVPress, DecodingPress, ExpectedAttentionPress, KnormPress, KVComposePress, KVzipPress, ThinKPress
from kvfold.merging_any_press import PATH_PASSTHROUGH, CompensationResult, KeptSet, MassFold, MergingAnyPress, _biased_mask, dispatch_press, recover_rows
from tests.fixtures import unit_test_model  # noqa: F401


def _kv(b=1, h=2, n=10, d=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(b, h, n, d, generator=g), torch.randn(b, h, n, d, generator=g)


def test_recover_rows_gather_permuted():
    k0, v0 = _kv()
    idx = torch.tensor([[[7, 2, 5, 0], [1, 9, 3, 4]]])
    gi = idx.unsqueeze(-1).expand(-1, -1, -1, 4)
    src, path, reason = recover_rows(k0, v0, k0.gather(2, gi), v0.gather(2, gi))
    assert reason == "" and path == "gather-match"
    assert torch.equal(src, idx)


def test_recover_rows_inplace_with_appended_slots():
    k0, v0 = _kv()
    extra_k, extra_v = _kv(n=3, seed=1)
    src, path, reason = recover_rows(k0, v0, torch.cat([k0, extra_k], 2), torch.cat([v0, extra_v], 2))
    assert reason == "" and path == "in-place+appended"
    assert (src[..., :10] == torch.arange(10)).all() and (src[..., 10:] == -1).all()


def test_recover_rows_detects_altered_keys_and_values():
    k0, v0 = _kv()
    _, path, reason = recover_rows(k0, v0, k0[:, :, :6] * 1.5, v0[:, :, :6])
    assert path == PATH_PASSTHROUGH and "keys altered" in reason
    _, path, reason = recover_rows(k0, v0, k0[:, :, :6], v0[:, :, :6] + 1.0)
    assert path == PATH_PASSTHROUGH and "values altered" in reason


def _kept_set(scores=None):
    k0, v0 = _kv(h=1, n=4, d=2)
    src = torch.tensor([[[0, 2]]])  # rows 0 and 2 kept (cache order), rows 1 and 3 evicted
    kept = torch.ones(1, 1, 2, dtype=torch.bool)
    target = torch.tensor([[[False, True]]])  # row 0 is a sink
    evicted = torch.tensor([[[False, True, False, True]]])
    gi = src.unsqueeze(-1).expand(-1, -1, -1, 2)
    return KeptSet(
        layer_idx=0,
        module=types.SimpleNamespace(layer_idx=0),
        keys=k0,
        values=v0,
        cache_keys=k0.gather(2, gi),
        cache_values=v0.gather(2, gi),
        src=src,
        kept=kept,
        target=target,
        evicted=evicted,
        path="gather-match",
        scores=scores,
    )


def test_mass_fold_formula_and_protection():
    s = torch.tensor([[[0.4, 0.2, 0.3, 0.1]]])
    ks = _kept_set(scores=s)
    res = MassFold(score_map="log")(ks)
    # both evicted rows can only go to the single target row (cache row 1 = snapshot row 2)
    w = torch.tensor([0.2 / 0.3, 0.1 / 0.3])
    v = ks.values[0, 0]
    expected = (v[2] + w[0] * v[1] + w[1] * v[3]) / (1 + w.sum())
    assert torch.allclose(res.values[0, 0, 1], expected, atol=1e-6)
    assert torch.equal(res.values[0, 0, 0], ks.cache_values[0, 0, 0])  # sink row untouched
    assert res.logit_bias[0, 0, 0] == 0
    assert math.isclose(float(res.logit_bias[0, 0, 1]), math.log(1 + float(w.sum())), rel_tol=1e-6)


def test_mass_fold_without_scores_is_skipped():
    res = MassFold()(_kept_set(scores=None))
    assert res.values is None and "no full-length" in res.info["skipped"]


def test_apply_refuses_writes_outside_targets():
    ks = _kept_set()
    press = MergingAnyPress(press=KnormPress(compression_ratio=0.5))
    press._cache = types.SimpleNamespace(layers=[types.SimpleNamespace(values=ks.cache_values)])
    bad = ks.cache_values.clone()
    bad[0, 0, 0] += 1.0  # the sink row
    with pytest.raises(RuntimeError, match="outside the target rows"):
        press._apply(ks.module, 0, ks, CompensationResult(values=bad), {})


def test_biased_mask_shapes_and_causality():
    q = torch.zeros(1, 4, 3, 2)  # 4 query heads, 3 queries
    k = torch.zeros(1, 2, 5, 2)  # 2 kv heads, 5 keys
    bias = torch.tensor([[[1.0, 0.0], [0.0, 2.0]]])  # bias on the first 2 cache rows
    m = _biased_mask(None, bias, q, k)
    assert m.shape == (1, 4, 3, 5)
    assert m[0, 0, 0, 0] == 1.0 and m[0, 2, 0, 1] == 2.0 and m[0, 1, 0, 0] == 1.0
    assert m[0, 0, 0, 3] < -1e30 and m[0, 0, 2, 4] == 0.0  # causal over the 3 new queries
    boolmask = torch.ones(1, 1, 1, 5, dtype=torch.bool)
    boolmask[..., 4] = False
    m1 = _biased_mask(boolmask, bias, q[:, :, :1], k)
    assert m1[0, 3, 0, 1] == 2.0 and m1[0, 0, 0, 4] < -1e30


def test_dispatch_and_static_passthrough():
    inner = DecodingPress(base_press=KnormPress(compression_ratio=0.5))
    w = MergingAnyPress(press=inner, compensation="mass_fold")
    assert dispatch_press(w) is inner
    assert w.static_reason is not None and "decoding-time" in w.static_reason


@torch.no_grad()
def test_none_is_bit_identical(unit_test_model):  # noqa: F811
    input_ids = torch.randint(0, 1024, (1, 64), generator=torch.Generator().manual_seed(1))
    caches = []
    for press in (KnormPress(compression_ratio=0.5), MergingAnyPress(press=KnormPress(compression_ratio=0.5))):
        cache = DynamicCache()
        with press(unit_test_model):
            unit_test_model(input_ids, past_key_values=cache)
        caches.append(cache)
    for a, b in zip(caches[0].layers, caches[1].layers):
        assert torch.equal(a.keys, b.keys) and torch.equal(a.values, b.values)


@torch.no_grad()
def test_mass_fold_keeps_kept_set_on_mask_press(unit_test_model):  # noqa: F811
    input_ids = torch.randint(0, 1024, (1, 64), generator=torch.Generator().manual_seed(2))
    out = {}
    for name, press in (
        ("P", AdaKVPress(press=ExpectedAttentionPress(compression_ratio=0.5))),
        (
            "W",
            MergingAnyPress(
                press=AdaKVPress(press=ExpectedAttentionPress(compression_ratio=0.5)), compensation="mass_fold"
            ),
        ),
    ):
        cache = DynamicCache()
        with press(unit_test_model):
            unit_test_model(input_ids, past_key_values=cache)
        masks = [layer.self_attn.masked_key_indices for layer in unit_test_model.model.layers]
        out[name] = (cache, [tuple(t.clone() for t in m) for m in masks], press)
    for (ca, ma), (cb, mb) in zip(zip(out["P"][0].layers, out["P"][1]), zip(out["W"][0].layers, out["W"][1])):
        assert torch.equal(ca.keys, cb.keys)
        assert all(torch.equal(x, y) for x, y in zip(ma, mb))
    rep = out["W"][2].last_report
    assert sum(r.get("writes", 0) for r in rep) > 0
    assert all(r.get("protected_writes", 0) == 0 for r in rep)


@torch.no_grad()
def test_key_altering_press_passes_through(unit_test_model):  # noqa: F811
    input_ids = torch.randint(0, 1024, (1, 64), generator=torch.Generator().manual_seed(3))
    w = MergingAnyPress(press=ThinKPress(key_channel_compression_ratio=0.5, window_size=2), compensation="mass_fold")
    with w(unit_test_model):
        unit_test_model(input_ids, past_key_values=DynamicCache())
    assert all(r["path"] == PATH_PASSTHROUGH for r in w.last_report)


@torch.no_grad()
def test_reentry_without_prefill_keeps_bias(unit_test_model):  # noqa: F811
    input_ids = torch.randint(0, 1024, (1, 64), generator=torch.Generator().manual_seed(4))
    w = MergingAnyPress(press=KnormPress(compression_ratio=0.5), compensation=MassFold(score_map="z"))
    cache = DynamicCache()
    with w(unit_test_model):
        unit_test_model(input_ids, past_key_values=cache)
    biases = [layer.self_attn.anypress_logit_bias for layer in unit_test_model.model.layers]
    assert all(b is not None for b in biases)
    report = w.last_report
    with w(unit_test_model):  # e.g. a pipeline decoding phase: no context prefill inside
        pass
    assert all(layer.self_attn.anypress_logit_bias is b for layer, b in zip(unit_test_model.model.layers, biases))
    assert w.last_report == report


@torch.no_grad()
def test_stale_bias_does_not_leak_into_a_later_run(unit_test_model):  # noqa: F811
    """A later run whose prefill bypasses the attention registry (KVComposePress prefills in eager mode) must not
    inherit the logit bias of an earlier W(P) run."""
    g = torch.Generator().manual_seed(5)
    ids, q = torch.randint(0, 1024, (1, 64), generator=g), torch.randint(0, 1024, (1, 4), generator=g)

    def bare_logits():
        cache = DynamicCache()
        with KVComposePress(compression_ratio=0.5)(unit_test_model):
            unit_test_model.model(input_ids=ids, past_key_values=cache)
        pos = torch.arange(64, 68).unsqueeze(0)
        return unit_test_model(input_ids=q, past_key_values=cache, position_ids=pos).logits

    before = bare_logits()
    w = MergingAnyPress(press=KnormPress(compression_ratio=0.5), compensation=MassFold(score_map="z"))
    with w(unit_test_model):
        unit_test_model(ids, past_key_values=DynamicCache())
    assert any(layer.self_attn.anypress_logit_bias is not None for layer in unit_test_model.model.layers)
    assert torch.equal(before, bare_logits())


@torch.no_grad()
def test_stale_bias_cleared_when_the_same_press_prefills_outside_the_registry(unit_test_model):  # noqa: F811
    """Review P20: a biased W(P) run, then bare P on the SAME context whose prefill bypasses the attention registry
    (eager attention, as in KVComposePress). The bare run keeps the same kept rows, so the probe of the old bias
    matches; only the pre-hook at cache position 0 removes the stale bias before the question pass."""
    g = torch.Generator().manual_seed(6)
    ids, q = torch.randint(0, 1024, (1, 64), generator=g), torch.randint(0, 1024, (1, 4), generator=g)
    layers = unit_test_model.model.layers

    def bare_logits():
        cache = DynamicCache()
        impl = unit_test_model.config._attn_implementation
        unit_test_model.config._attn_implementation = "eager"  # prefill outside ALL_ATTENTION_FUNCTIONS
        try:
            with KnormPress(compression_ratio=0.5)(unit_test_model):
                unit_test_model.model(input_ids=ids, past_key_values=cache)
        finally:
            unit_test_model.config._attn_implementation = impl
        assert all(getattr(layer.self_attn, "anypress_logit_bias", None) is None for layer in layers)
        pos = torch.arange(64, 68).unsqueeze(0)
        return unit_test_model(input_ids=q, past_key_values=cache, position_ids=pos).logits

    before = bare_logits()
    w = MergingAnyPress(press=KnormPress(compression_ratio=0.5), compensation=MassFold(score_map="z"))
    with w(unit_test_model):
        unit_test_model(ids, past_key_values=DynamicCache())
    assert all(layer.self_attn.anypress_logit_bias is not None for layer in layers)
    assert torch.equal(before, bare_logits())














