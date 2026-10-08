# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Black-box kept-set wrapper around any press, with a pluggable compensation hook.

``MergingAnyPress(press, compensation)`` runs ``press`` unchanged. It snapshots each layer's K/V at the end of the
context prefill (a forward hook registered *before* the inner press registers its own hooks), lets the inner press
finish (including call-overriding presses such as KVzip, whose eviction happens when their context manager exits),
and then recovers, per (layer, batch, kv-head), which cache rows are original context rows and which of them are
still attended. A compensation may then rewrite the values of eligible kept rows and attach an additive logit bias.
With ``compensation="none"`` nothing is written, so the wrapped press is reproduced bit for bit.
"""

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Union

import torch
from torch import nn
from transformers import PreTrainedModel, QuantizedCache
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from kvpress.presses.base_press import BasePress, is_prefilling
from kvpress.presses.scorer_press import ScorerPress
from kvpress.utils import extract_keys_and_values

try:  # the research build of kvpress defines get_query_states; upstream kvpress 0.5.5 has get_prerope_query_states only
    from kvpress.utils import get_query_states
except ImportError:
    from kvpress.utils import get_prerope_query_states

    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def get_query_states(module: nn.Module, hidden_states: torch.Tensor, position_embeddings: tuple) -> torch.Tensor:
        """Rotated query states (same arithmetic as the helper of the research build of kvpress)."""
        cos, sin = position_embeddings
        query_states = get_prerope_query_states(module, hidden_states)
        return (query_states * cos.unsqueeze(1)) + (_rotate_half(query_states) * sin.unsqueeze(1))

logger = logging.getLogger(__name__)

_EPS = 1e-6
_MAX_LOG_WEIGHT = 30.0
_HASH_SEED = 20260929

# Extraction paths recorded per layer in ``MergingAnyPress.last_report``
PATH_INPLACE = "in-place"  # cache rows [0, L) are the snapshot rows (mask-based presses, KVzip family)
PATH_MATCH = "gather-match"  # truncated / reordered cache; rows recovered by exact (key, value) row matching
PATH_APPENDED = "+appended"  # suffix of cache rows that are not snapshot rows (e.g. RestoreKV restore slots)
PATH_MASK = "+mask"  # masked_key_indices present
PATH_IDENTITY = "identity"  # cache equals the snapshot and nothing is masked (no eviction on this layer)
PATH_PASSTHROUGH = "passthrough"


# --------------------------------------------------------------------------------------------------------------------
# press tree helpers
# --------------------------------------------------------------------------------------------------------------------
def press_tree(press: BasePress) -> list[BasePress]:
    """All presses reachable from ``press`` through the wrapper attributes used in kvpress."""
    seen: list[BasePress] = []
    stack = [press]
    while stack:
        p = stack.pop()
        if not isinstance(p, BasePress) or any(p is s for s in seen):
            continue
        seen.append(p)
        for name in ("press", "base_press", "prefilling_press", "decoding_press"):
            child = getattr(p, name, None)
            if isinstance(child, BasePress):
                stack.append(child)
        for child in getattr(p, "presses", None) or []:
            stack.append(child)
    return seen


def static_passthrough_reason(press: BasePress) -> Optional[str]:
    """Presses whose eviction happens (also) during decoding. The kept set is not fixed after the prefill, so a
    compensation computed at prefill time would be misaligned with later cache rows: route to pass-through."""
    from kvpress.presses.decoding_press import DecodingPress
    from kvpress.presses.dms_press import DMSPress
    from kvpress.presses.prefill_decoding_press import PrefillDecodingPress

    for p in press_tree(press):
        if isinstance(p, DecodingPress):
            return f"decoding-time press ({type(p).__name__} evicts during generation)"
        if isinstance(p, PrefillDecodingPress) and p.decoding_press is not None:
            return "decoding-time press (PrefillDecodingPress with a decoding_press)"
        if isinstance(p, DMSPress) and p.decoding:
            return "decoding-time press (DMSPress with decoding=True)"
        if isinstance(p, MergingAnyPress) and p is not press:
            return "nested MergingAnyPress"
    return None


def dispatch_press(press: Optional[BasePress]) -> Optional[BasePress]:
    """The press whose type decides pipeline behaviour (prefill vs decoding compression, RestoreKV positions, key
    rerotation). ``MergingAnyPress`` is transparent: the pipeline must dispatch on the wrapped press."""
    while isinstance(press, MergingAnyPress):
        press = press.press
    return press


def _language_model(model: PreTrainedModel) -> nn.Module:
    return model.model.language_model if hasattr(model.model, "language_model") else model.model


# --------------------------------------------------------------------------------------------------------------------
# row recovery
# --------------------------------------------------------------------------------------------------------------------
def _row_hash(x: torch.Tensor) -> torch.Tensor:
    """Exact-content hash of the last dimension, (B, H, L, D) -> (B, H, L) int64 (wrapping arithmetic)."""
    word = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[x.element_size()]
    raw = x.contiguous().view(word).long()
    g = torch.Generator(device="cpu").manual_seed(_HASH_SEED)
    coef = torch.randint(1, 2**31 - 1, (raw.shape[-1],), generator=g, dtype=torch.long).to(x.device)
    return (raw * coef).sum(-1) * 1000003 + (raw * coef.flip(0)).sum(-1)


def _match(h0: torch.Tensor, h1: torch.Tensor) -> torch.Tensor:
    """For every row of ``h1`` (B, H, L1) the index of an equal hash in ``h0`` (B, H, L0), else -1."""
    s0, perm = h0.sort(dim=-1)
    pos = torch.searchsorted(s0, h1).clamp(max=h0.shape[-1] - 1)
    ok = s0.gather(-1, pos) == h1
    return torch.where(ok, perm.gather(-1, pos), torch.full_like(pos, -1))


def _verify(src: torch.Tensor, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
    """Exact row equality x1[r] == x0[src[r]] (rows with src < 0 are False)."""
    idx = src.clamp(min=0).unsqueeze(-1).expand(-1, -1, -1, x0.shape[-1])
    eq = (x0.gather(2, idx) == x1).all(-1)
    return eq & (src >= 0)


def recover_rows(
    k0: torch.Tensor, v0: torch.Tensor, k1: torch.Tensor, v1: torch.Tensor
) -> tuple[torch.Tensor, str, str]:
    """Map cache rows (k1, v1) back to snapshot rows (k0, v0).

    Returns ``src`` (B, H, L1) long with the snapshot index of every cache row or -1, the extraction path, and a
    pass-through reason ("" when the rows are recovered).
    """
    bsz, heads, l0, _ = k0.shape
    l1 = k1.shape[2]
    ar = torch.arange(l1, device=k1.device)
    if l1 >= l0 and torch.equal(k1[:, :, :l0], k0) and torch.equal(v1[:, :, :l0], v0):
        src = torch.where(ar < l0, ar, torch.full_like(ar, -1)).expand(bsz, heads, l1).clone()
        return src, PATH_INPLACE + (PATH_APPENDED if l1 > l0 else ""), ""
    hk0, hv0, hk1, hv1 = _row_hash(k0), _row_hash(v0), _row_hash(k1), _row_hash(v1)
    src = _match(hk0 * 31 + hv0, hk1 * 31 + hv1)
    ok = _verify(src, k0, k1) & _verify(src, v0, v1)
    src = torch.where(ok, src, torch.full_like(src, -1))
    n_ok = ok.sum(-1)  # (B, H)
    if bool(ok.all()):
        return src, PATH_MATCH, ""
    # unmatched rows are allowed only as a common suffix of every head (appended slots)
    same_n = bool((n_ok == n_ok.flatten()[0]).all())
    suffix = bool((ok == (ar < n_ok.unsqueeze(-1))).all())
    if same_n and suffix and int(n_ok.flatten()[0]) > 0:
        return src, PATH_MATCH + PATH_APPENDED, ""
    v_only = _verify(_match(hv0, hv1), v0, v1)
    k_only = _verify(_match(hk0, hk1), k0, k1)
    if bool(v_only.all()):
        return src, PATH_PASSTHROUGH, "kept keys altered (values match, keys do not: rerotation or channel pruning)"
    if bool(k_only.all()):
        return src, PATH_PASSTHROUGH, "kept values altered (keys match, values do not: merged or rewritten values)"
    return src, PATH_PASSTHROUGH, "cache rows not recoverable from the snapshot (quantized, learned or rewritten rows)"


# --------------------------------------------------------------------------------------------------------------------
# compensation interface
# --------------------------------------------------------------------------------------------------------------------
@dataclass
class KeptSet:
    """Everything a compensation may use for one layer. Snapshot tensors are in context order (length L0); cache
    tensors are in cache order (length L1). ``target`` rows are the only rows a compensation may modify."""

    layer_idx: int
    module: nn.Module
    keys: torch.Tensor  # (B, H, L0, D) snapshot keys
    values: torch.Tensor  # (B, H, L0, D) snapshot values
    cache_keys: torch.Tensor  # (B, H, L1, D)
    cache_values: torch.Tensor  # (B, H, L1, D)
    src: torch.Tensor  # (B, H, L1) snapshot index of each cache row, -1 for appended / learned rows
    kept: torch.Tensor  # (B, H, L1) bool: snapshot rows that stay attended
    target: torch.Tensor  # (B, H, L1) bool: kept rows that may receive compensation (no sinks, no appended rows)
    evicted: torch.Tensor  # (B, H, L0) bool: snapshot rows no longer attended
    path: str
    scores: Optional[torch.Tensor] = None  # (B, H, L0) inner press score in context order (if exposed)
    hidden_tail: Optional[torch.Tensor] = None  # (B, w, hidden) last w context hidden states (context-only)
    position_tail: Optional[tuple[torch.Tensor, torch.Tensor]] = None  # matching (cos, sin)
    repeat: Optional[list] = None  # [(hidden (B, n, hidden), (cos, sin))] of post-context forwards inside P

    def reference_queries(self) -> torch.Tensor:
        """RoPE'd queries of the last w context positions, (B, H_q, w, D). Context-only by construction."""
        if self.hidden_tail is None or self.position_tail is None:
            raise RuntimeError("reference queries were not captured (set compensation.query_window > 0)")
        return get_query_states(self.module, self.hidden_tail, self.position_tail)

    def repeat_queries(self) -> Optional[torch.Tensor]:
        """RoPE'd queries of the press's own post-context forwards (KVzip-family reconstruction), (B, H_q, m, D),
        or None if the press ran none. They sit after the context and see every context key."""
        if not self.repeat:
            return None
        return torch.cat([get_query_states(self.module, h, pe) for h, pe in self.repeat], dim=2)


@dataclass
class CompensationResult:
    values: Optional[torch.Tensor] = None  # (B, H, L1, D) full cache-order values; only target rows may differ
    logit_bias: Optional[torch.Tensor] = None  # (B, H, L1) additive logit bias; zero outside target rows
    info: dict = field(default_factory=dict)


class Compensation:
    """Base class. ``needs_scores``: capture the inner press score; ``query_window``: capture the last w context
    hidden states for reference queries; ``repeat_blocks`` > 0: keep that many blocks of ``repeat_block_len``
    positions of every post-context forward the press runs (KVzip reconstruction queries). ``__call__`` returns None
    (no change) or a CompensationResult."""

    name = "base"
    needs_scores = False
    query_window = 0
    repeat_blocks = 0
    repeat_block_len = 32

    def __call__(self, ks: KeptSet) -> Optional[CompensationResult]:
        raise NotImplementedError


class NoCompensation(Compensation):
    name = "none"

    def __call__(self, ks: KeptSet) -> Optional[CompensationResult]:
        return None


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


def map_scores(scores: torch.Tensor, score_map: str) -> torch.Tensor:
    """Press score -> log-mass units (as in the C86 rule): ``log`` for non-negative attention-type scores, ``z``
    (per-head z-score) for norm-type scores. ``auto`` picks log when all scores are >= 0, else z; it is a smoke-test
    convenience, not a validated rule."""
    s = scores.float()
    if score_map == "auto":
        score_map = "log" if bool((s >= 0).all()) else "z"
    if score_map == "log":
        if bool((s < 0).any()):
            raise ValueError("score_map='log' needs non-negative scores")
        return torch.log(s.clamp(min=1e-30))
    if score_map == "z":
        mu = s.mean(dim=-1, keepdim=True)
        sd = s.std(dim=-1, keepdim=True).clamp(min=1e-6)
        return (s - mu) / sd
    raise ValueError(f"unknown score_map {score_map!r}")


@dataclass
class FoldRouting:
    """Routing of one layer's :class:`MassFold`, handed to ``MassFold.observer`` (read-only views, nothing is copied).

    Row axes: ``L0`` snapshot (context) rows, ``L1`` cache rows. ``target_idx`` / ``cosine`` / ``weight`` are defined for
    every snapshot row but only meaningful where ``valid``; the target of snapshot row j is cache row ``target_idx[j]``
    whose snapshot position is ``ks.src[target_idx[j]]``.
    """

    layer_idx: int
    target_idx: torch.Tensor  # (B, H, L0) long, cache row that receives snapshot row j
    cosine: torch.Tensor  # (B, H, L0) float32, cosine of the snapshot key and the key of its target row
    valid: torch.Tensor  # (B, H, L0) bool, evicted rows that are folded
    weight: torch.Tensor  # (B, H, L0) float32, w_j = exp(s_j - s_i), 0 where not valid
    den: torch.Tensor  # (B, H, L1) float32, sum of w_j per cache row; the logit bias is log1p(den)
    target: torch.Tensor  # (B, H, L1) bool, rows that were eligible as targets in this call (after exclude_targets)


@dataclass
class MassFold(Compensation):
    """C86 rule: fold every evicted token j into its most cosine-similar target row i with weight
    ``w_j = exp(s_j - s_i)``: ``v_i' = (v_i + sum_j w_j v_j) / (1 + sum_j w_j)``, and add ``b_i = log(1 + sum_j w_j)``
    to the logit of row i. Needs a full-length inner press score.

    Two optional hooks, both off by default (with both ``None`` the arithmetic is the C86 rule, operation for
    operation):

    ``exclude_targets(ks) -> (B, H, L1) bool``: cache rows that may not receive a fold (they stay unmodified and the
    evicted rows route to the next most similar target). Rows that are not targets in ``ks.target`` are unaffected.

    ``observer(ks, routing)``: called once per layer with the :class:`FoldRouting` (target index, key cosine, weights,
    denominators) after the fold is computed; used for compression-time structure logging. It must not modify tensors.
    """

    score_map: str = "log"
    chunk: int = 1024
    exclude_targets: Optional[Callable[[KeptSet], torch.Tensor]] = None
    observer: Optional[Callable[[KeptSet, FoldRouting], None]] = None
    scatter: str = "atomic"  # "atomic": zeros.scatter_add_ (the C86 arithmetic); "sorted": fixed-order segment sums (bit-reproducible without a global flag)
    name = "mass_fold"
    needs_scores = True

    def __post_init__(self):
        assert self.scatter in ("atomic", "sorted"), f"scatter must be 'atomic' or 'sorted', got {self.scatter!r}"

    def __call__(self, ks: KeptSet) -> Optional[CompensationResult]:
        if ks.scores is None:
            return CompensationResult(info={"skipped": "no full-length inner score exposed"})
        s = map_scores(ks.scores, self.score_map)  # (B, H, L0)
        s = torch.where(torch.isfinite(s), s, torch.full_like(s, -1e4))
        bsz, heads, l0, d = ks.keys.shape
        l1 = ks.cache_keys.shape[2]
        tk = ks.cache_keys.float()
        tk = tk / tk.norm(dim=-1, keepdim=True).clamp(min=_EPS)
        ek = ks.keys.float()
        ek = ek / ek.norm(dim=-1, keepdim=True).clamp(min=_EPS)
        tgt = ks.target
        if self.exclude_targets is not None:
            excl = self.exclude_targets(ks)
            assert excl.dtype == torch.bool and excl.shape == ks.target.shape, (
                f"exclude_targets must return a bool tensor of shape {tuple(ks.target.shape)}, "
                f"got {excl.dtype} {tuple(excl.shape)}"
            )
            tgt = ks.target & ~excl
        target_idx = torch.empty(bsz, heads, l0, dtype=torch.long, device=tk.device)
        cos_best = (
            torch.empty(bsz, heads, l0, dtype=torch.float32, device=tk.device) if self.observer is not None else None
        )
        for a in range(0, l0, self.chunk):
            sim = ek[:, :, a : a + self.chunk] @ tk.transpose(-2, -1)  # (B, H, c, L1)
            sim = sim.masked_fill(~tgt.unsqueeze(2), float("-inf"))
            idx = sim.argmax(-1)
            target_idx[:, :, a : a + self.chunk] = idx
            if cos_best is not None:
                cos_best[:, :, a : a + self.chunk] = sim.gather(-1, idx.unsqueeze(-1)).squeeze(-1)
        valid = ks.evicted & tgt.any(-1, keepdim=True)
        s_row = s.gather(2, ks.src.clamp(min=0))  # (B, H, L1) score of the snapshot row behind each cache row
        log_w = (s - s_row.gather(2, target_idx)).clamp(max=_MAX_LOG_WEIGHT)
        w = torch.exp(log_w) * valid.float()  # (B, H, L0)
        if self.scatter == "sorted":
            num = scatter_add_sorted(target_idx, w.unsqueeze(-1) * ks.values.float(), l1)
            den = scatter_add_sorted(target_idx, w, l1)
        else:
            num = scatter_add_atomic(target_idx, w.unsqueeze(-1) * ks.values.float(), l1)
            den = scatter_add_atomic(target_idx, w, l1)
        v1 = ks.cache_values
        folded = ((v1.float() + num) / (1.0 + den).unsqueeze(-1)).to(v1.dtype)
        new_v = torch.where((den > 0).unsqueeze(-1), folded, v1)
        if self.observer is not None:
            self.observer(
                ks,
                FoldRouting(
                    layer_idx=ks.layer_idx,
                    target_idx=target_idx,
                    cosine=cos_best,  # type: ignore[arg-type]
                    valid=valid,
                    weight=w,
                    den=den,
                    target=tgt,
                ),
            )
        return CompensationResult(values=new_v, logit_bias=torch.log1p(den), info={"merged": int(valid.sum())})


COMPENSATIONS = {"none": NoCompensation, "mass_fold": MassFold}


# --------------------------------------------------------------------------------------------------------------------
# additive logit bias at attention time
# --------------------------------------------------------------------------------------------------------------------
def _biased_mask(attention_mask, bias: torch.Tensor, query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
    bsz, num_heads, q_len, _ = query.shape
    k_len = key.shape[2]
    num_kv = bias.shape[1]
    full = torch.zeros(bsz, num_kv, k_len, dtype=torch.float32, device=query.device)
    n = min(bias.shape[2], k_len)
    full[:, :, :n] = bias[:, :, :n].to(full.device, torch.float32)
    full = full.repeat_interleave(num_heads // num_kv, dim=1).unsqueeze(2)  # (B, Hq, 1, K)
    neg = torch.finfo(torch.float32).min
    if attention_mask is None:
        if q_len > 1:
            qpos = torch.arange(k_len - q_len, k_len, device=query.device)
            allowed = torch.arange(k_len, device=query.device)[None, :] <= qpos[:, None]
            base = torch.zeros(q_len, k_len, device=query.device).masked_fill(~allowed, neg)[None, None]
        else:
            base = torch.zeros(1, 1, 1, k_len, device=query.device)
    elif attention_mask.dtype == torch.bool:
        m = attention_mask[..., :k_len]
        base = torch.zeros(m.shape, device=query.device).masked_fill(~m, neg)
    else:
        base = attention_mask[..., :k_len].float()
    return (base + full).clamp(min=neg).to(query.dtype)


def _bias_is_current(module, key: torch.Tensor) -> bool:
    """The bias is tied to one cache through a probe: one kept, unmasked target row of that cache's keys."""
    probe = getattr(module, "anypress_bias_probe", None)
    if probe is None:
        return False
    b, h, r, row = probe
    return key.shape[2] > r and torch.equal(key[b, h, r].to(row.device), row)


def _bias_patch(func):
    def wrapper(module, query, key, value, attention_mask, *args, **kwargs):
        bias = getattr(module, "anypress_logit_bias", None)
        if bias is not None:
            if query.shape[2] == key.shape[2] or not _bias_is_current(module, key):
                # a fresh prefill, or a cache that is not the one the bias was computed for (e.g. a later run whose
                # prefill bypassed this registry, as eager attention does): the bias is stale and is dropped
                module.anypress_logit_bias = None
                module.anypress_bias_probe = None
            else:
                attention_mask = _biased_mask(attention_mask, bias, query, key)
        return func(module, query, key, value, attention_mask, *args, **kwargs)

    wrapper._anypress_bias = True  # type: ignore[attr-defined]
    return wrapper


def patch_attention_bias():
    """Idempotent: wrap every registered attention function so that ``module.anypress_logit_bias`` is added to the
    attention logits of the first L1 key positions. Eager attention is not routed through this registry."""
    for name, func in ALL_ATTENTION_FUNCTIONS.items():
        if not getattr(func, "_anypress_bias", False):
            ALL_ATTENTION_FUNCTIONS[name] = _bias_patch(func)


patch_attention_bias()

# every per-module attribute through which a merge / compensation bias reaches the attention logits
_BIAS_ATTRS = ("anypress_logit_bias", "anypress_bias_probe", "merge_logit_bias")


def _starts_new_context(module, kwargs) -> bool:
    cache_position = kwargs.get("cache_position")
    if cache_position is not None:
        return cache_position.numel() > 0 and int(cache_position.reshape(-1)[0]) == 0
    cache = kwargs.get("past_key_values", kwargs.get("past_key_value"))
    return cache is None or cache.get_seq_length(int(getattr(module, "layer_idx", 0) or 0)) == 0


def _clear_stale_bias(module, args, kwargs):
    """Forward pre-hook: a forward that starts at cache position 0 is a new context, so every merge / compensation
    bias left on the module by an earlier run is stale. It is cleared before attention runs, whatever attention
    implementation the forward uses (the attention-registry patch alone misses prefills that bypass the registry,
    e.g. eager attention)."""
    if _starts_new_context(module, kwargs):
        for name in _BIAS_ATTRS:
            if getattr(module, name, None) is not None:
                setattr(module, name, None)
    return None


def install_bias_clearing(model: PreTrainedModel) -> int:
    """Idempotent: register :func:`_clear_stale_bias` on every attention module of ``model``. Returns the number of
    newly registered hooks."""
    n = 0
    for layer in _language_model(model).layers:
        module = layer.self_attn
        if getattr(module, "_anypress_clear_hook", None) is None:
            module._anypress_clear_hook = module.register_forward_pre_hook(_clear_stale_bias, with_kwargs=True)
            n += 1
    return n


# --------------------------------------------------------------------------------------------------------------------
# the wrapper
# --------------------------------------------------------------------------------------------------------------------
@dataclass
class MergingAnyPress(BasePress):
    """
    Wrap any press P. W(P) runs P unchanged, recovers P's kept set per (layer, kv-head) and applies a compensation to
    the values (and an additive logit bias) of eligible kept rows. Keys and the kept set are never changed, so kept
    counts and the effective compression ratio equal those of P. ``compensation="none"`` reproduces P bit for bit.

    Parameters
    ----------
    press : BasePress
        Any kvpress press.
    compensation : str or Compensation, default="none"
        ``"none"``, ``"mass_fold"`` (C86 rule, needs a full-length inner score) or a :class:`Compensation`.
    n_sink : int, optional
        Leading context positions that never receive compensation. Default: the largest ``n_sink`` found in the
        press tree (0 if none).
    """

    press: BasePress = None  # type: ignore[assignment]
    compensation: Union[str, Compensation] = "none"
    n_sink: Optional[int] = None

    def __post_init__(self):
        assert isinstance(self.press, BasePress), f"MergingAnyPress wraps a BasePress, got {type(self.press)}"
        if isinstance(self.compensation, str):
            assert self.compensation in COMPENSATIONS, f"compensation must be one of {list(COMPENSATIONS)}"
            self._comp: Compensation = COMPENSATIONS[self.compensation]()
        else:
            assert isinstance(self.compensation, Compensation)
            self._comp = self.compensation
        self.last_report: list[dict[str, Any]] = []
        self.static_reason: Optional[str] = static_passthrough_reason(self.press)
        self._reset_state()

    def _reset_state(self):
        self._snap: dict[int, Optional[tuple[torch.Tensor, torch.Tensor]]] = {}
        self._tail: dict[int, tuple] = {}
        self._scores: dict[int, torch.Tensor] = {}
        self._repeat: dict[int, list] = {}
        self._cache = None

    def _capture_repeat(self, layer_idx: int, hidden_states: torch.Tensor, kwargs: dict):
        """Keep ``repeat_blocks`` evenly spaced blocks of ``repeat_block_len`` positions of a post-context forward
        (hidden states + rotary embeddings, so queries can be formed later with the layer's own projection)."""
        n_blocks = int(getattr(self._comp, "repeat_blocks", 0) or 0)
        pe = kwargs.get("position_embeddings")
        if n_blocks <= 0 or pe is None or self._snap.get(layer_idx) is None:
            return
        cache_position = kwargs.get("cache_position")
        l0 = self._snap[layer_idx][0].shape[2]
        if cache_position is None or int(cache_position.reshape(-1)[0]) < l0:
            return  # not after the context: its queries could not see every context key
        bl = int(getattr(self._comp, "repeat_block_len", 32))
        n = hidden_states.shape[1]
        starts = torch.linspace(0, max(n - bl, 0), n_blocks).round().long().unique()
        idx = torch.cat([torch.arange(s, min(s + bl, n)) for s in starts.tolist()]).to(hidden_states.device)
        cos, sin = pe
        self._repeat.setdefault(layer_idx, []).append(
            (hidden_states[:, idx].detach().clone(), (cos[:, idx].clone(), sin[:, idx].clone()))
        )

    # --- passthrough of the inner press's public knobs ---
    def post_init_from_model(self, model):
        self.press.post_init_from_model(model)

    @property
    def compression_ratio(self):  # type: ignore[override]
        return self.press.compression_ratio  # type: ignore[attr-defined]

    @compression_ratio.setter
    def compression_ratio(self, value):
        self.press.compression_ratio = value  # type: ignore[attr-defined]

    def _sinks(self) -> int:
        if self.n_sink is not None:
            return int(self.n_sink)
        return max([int(getattr(p, "n_sink", 0) or 0) for p in press_tree(self.press)] + [0])

    # --- hooks ---
    def _snapshot_hook(self, module, args, kwargs, output):
        hidden_states = kwargs["hidden_states"]
        layer_idx = int(module.layer_idx)
        if layer_idx in self._snap:
            # a later forward inside the press context (KVzip-family context reconstruction): its queries sit after
            # the context and see all of it, so they are context-only reference queries
            self._capture_repeat(layer_idx, hidden_states, kwargs)
            return None
        if not is_prefilling(kwargs["cache_position"], hidden_states.shape[1]):
            return None
        module.anypress_logit_bias = None  # a new context: any earlier bias is stale
        module.anypress_bias_probe = None
        cache = kwargs["past_key_values"]
        self._cache = cache
        if isinstance(cache, QuantizedCache):
            self._snap[layer_idx] = None
            return None
        keys, values = extract_keys_and_values(cache, layer_idx)
        self._snap[layer_idx] = (keys.detach().clone(), values.detach().clone())
        w = int(getattr(self._comp, "query_window", 0) or 0)
        if w > 0:
            cos, sin = kwargs["position_embeddings"]
            tail = (hidden_states[:, -w:].detach().clone(), (cos[:, -w:].clone(), sin[:, -w:].clone()))
            self._tail[layer_idx] = tail
        return None

    def _install_score_capture(self) -> list[tuple[BasePress, str]]:
        shadows: list[tuple[BasePress, str]] = []
        for p in press_tree(self.press):
            if isinstance(p, ScorerPress) or callable(getattr(p, "score", None)):
                original = getattr(p, "score")

                def score_capturing(*args, _original=original, **kw):
                    s = _original(*args, **kw)
                    module = args[0] if args else kw.get("module")
                    li = getattr(module, "layer_idx", None)
                    snap = self._snap.get(int(li)) if li is not None else None
                    if (
                        isinstance(s, torch.Tensor)
                        and snap is not None
                        and int(li) not in self._scores
                        and tuple(s.shape) == tuple(snap[0].shape[:3])
                    ):
                        self._scores[int(li)] = s.detach().float().clone()  # before in-place edits (AdaKV safeguard)
                    return s

                setattr(p, "score", score_capturing)  # instance attribute shadows the class method
                shadows.append((p, "score"))
            if callable(getattr(p, "compress_post", None)):  # KVzip family, FastKVzip: score_val after compress_post
                original_post = getattr(p, "compress_post")

                def post_capturing(model_, _original=original_post, _p=p):
                    _original(model_)
                    sv = getattr(_p, "score_val", None)
                    if isinstance(sv, torch.Tensor):
                        for li in range(sv.shape[0]):
                            self._scores.setdefault(li, sv[li].detach().float().clone())

                setattr(p, "compress_post", post_capturing)
                shadows.append((p, "compress_post"))
        return shadows

    @contextmanager
    def __call__(self, model: PreTrainedModel):
        # Entering W(P) without a context prefill (e.g. a pipeline's decoding phase) leaves the state of the last
        # prefill (logit bias, report) untouched: only a new context prefill resets it (snapshot hook, attention patch).
        layers = _language_model(model).layers
        install_bias_clearing(model)  # stale biases are dropped at the next context prefill, whatever runs it
        self._reset_state()
        if self.static_reason is not None:
            self.last_report = [{"layer": -1, "path": PATH_PASSTHROUGH, "reason": self.static_reason}]
            with self.press(model):
                yield
            return
        hooks = [layer.self_attn.register_forward_hook(self._snapshot_hook, with_kwargs=True) for layer in layers]
        shadows = self._install_score_capture() if self._comp.needs_scores else []
        try:
            with self.press(model):
                yield
        finally:
            for h in hooks:
                h.remove()
            for p, name in shadows:
                delattr(p, name)
        try:
            if self._snap:  # this context saw a context prefill
                self._reconcile(model)
        finally:
            self._reset_state()

    # --- post-exit reconciliation ---
    def _reconcile(self, model: PreTrainedModel):
        eager = getattr(model.config, "_attn_implementation", None) == "eager"
        n_sink = self._sinks()
        self.last_report = []
        for layer in _language_model(model).layers:
            module = layer.self_attn
            li = int(module.layer_idx)
            rec: dict[str, Any] = {"layer": li, "compensation": self._comp.name}
            self.last_report.append(rec)
            snap = self._snap.get(li)
            if snap is None or self._cache is None:
                rec.update(path=PATH_PASSTHROUGH, reason="no prefill snapshot (quantized cache or skipped layer)")
                continue
            k0, v0 = snap
            k1, v1 = extract_keys_and_values(self._cache, li)
            bsz, heads, l0, _ = k0.shape
            l1 = k1.shape[2]
            src, path, reason = recover_rows(k0, v0, k1, v1)
            masked = torch.zeros(bsz, heads, l1, dtype=torch.bool, device=k1.device)
            mask_idx = getattr(module, "masked_key_indices", None)
            if mask_idx is not None and len(mask_idx[0]) > 0:
                masked[tuple(i.to(k1.device) for i in mask_idx)] = True
                path += PATH_MASK
            rec.update(path=path, reason=reason, n_ctx=l0, n_cache=l1, n_masked=int(masked.sum()))
            if reason:
                rec["path"] = PATH_PASSTHROUGH
                continue
            kept = (src >= 0) & ~masked
            # evicted snapshot rows: every context row that is not the source of a kept cache row
            ev = torch.ones(bsz, heads, l0 + 1, dtype=torch.bool, device=k1.device)
            ev.scatter_(2, torch.where(kept, src, torch.full_like(src, l0)), False)
            evicted = ev[..., :l0]
            target = kept & (src >= n_sink)
            if path == PATH_INPLACE and not bool(masked.any()) and l1 == l0:
                rec["path"] = PATH_IDENTITY
            rec.update(n_kept=int(kept.sum()), n_evicted=int(evicted.sum()), n_appended=int((src < 0).sum()))
            if not bool(evicted.any()) or isinstance(self._comp, NoCompensation):
                rec.update(writes=0, bias_nonzero=0)
                continue
            tail = self._tail.get(li) or (None, None)
            ks = KeptSet(
                layer_idx=li,
                module=module,
                keys=k0,
                values=v0,
                cache_keys=k1,
                cache_values=v1,
                src=src,
                kept=kept,
                target=target,
                evicted=evicted,
                path=path,
                scores=self._scores.get(li),
                hidden_tail=tail[0],
                position_tail=tail[1],
                repeat=self._repeat.get(li),
            )
            res = self._comp(ks)
            if res is None or (res.values is None and res.logit_bias is None):
                rec.update(writes=0, bias_nonzero=0, skipped=(res.info.get("skipped", "") if res else ""))
                continue
            if res.logit_bias is not None and bool((res.logit_bias != 0).any()) and eager:
                rec.update(writes=0, bias_nonzero=0, skipped="eager attention cannot carry the logit bias")
                continue
            self._apply(module, li, ks, res, rec)

    def _apply(self, module, layer_idx: int, ks: KeptSet, res: CompensationResult, rec: dict):
        new_v = ks.cache_values if res.values is None else res.values
        assert new_v.shape == ks.cache_values.shape, "compensation returned values of the wrong shape"
        changed = (new_v != ks.cache_values).any(-1)
        protected_writes = int((changed & ~ks.target).sum())
        bias = res.logit_bias
        protected_bias = int(((bias != 0) & ~ks.target).sum()) if bias is not None else 0
        if protected_writes or protected_bias:  # G4: never write into sinks, appended/learned rows or evicted rows
            raise RuntimeError(
                f"layer {layer_idx}: compensation wrote {protected_writes} values / {protected_bias} "
                "bias entries outside the target rows"
            )
        cache_layer = self._cache.layers[layer_idx]
        cache_layer.values = new_v.to(ks.cache_values.dtype).contiguous()
        if bias is not None and bool((bias != 0).any()):
            b, h, r = (int(i) for i in (bias != 0).nonzero()[0])  # a target row: kept, never masked, never rewritten
            module.anypress_bias_probe = (b, h, r, ks.cache_keys[b, h, r].detach().clone())
            module.anypress_logit_bias = bias.float()
        rec.update(
            writes=int(changed.sum()),
            bias_nonzero=int((bias != 0).sum()) if bias is not None else 0,
            protected_writes=0,
            **res.info,
        )
