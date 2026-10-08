"""Control compensations for the C86 mass fold (pre-registered in hypothesis_C113.json and in revision 1.1 of
hypothesis_C111.json). All three reuse MassFold of kvfold.merging_any_press unchanged, so the
routing (most cosine-similar target key), the weights w_j = exp(s_j - s_i) (log w capped at 30), the bias
b_i = log(1 + sum_j w_j) and the targets (no sinks, no appended/restore rows) are those of arm C.

PlaceboMassFold  (arm PL)    every snapshot value row v_j is replaced by a Gaussian row with the same L2 norm,
                             v~_j = ||v_j|| g_j / ||g_j||, g_j ~ N(0, I_D) in float32; only evicted rows have w_j > 0, so the
                             shrink v_i / (1 + sum w), the bias and the biased-mask kernel path of C stay, the content goes.
PermutedMassFold (arm Cperm) the evicted value rows are permuted within each (layer, kv-head) before folding (a random
                             permutation of that head's evicted positions; kept rows are untouched): real value vectors,
                             broken key-value correspondence.
ZeroBiasPathFold (arm C0)    MassFold is computed as in C, then the values are left untouched and the bias is forced to 0.
                             The runner installs the zero bias on every layer where C's bias would be non-zero, so C0 takes
                             C's biased-mask attention path with a zero-content bias and no value change (kernel-path control).
SelfMassFold     (arm B)     added for the C113 confirmation (revision 2 of this file): C's routing, weights and bias with
                             every evicted value row replaced by its routing target's own value row before folding, so
                             v_i' = (v_i + sum_j w_j v_i) / (1 + sum_j w_j) = v_i up to float rounding. The routing is recomputed
                             with MassFold's own code (keys only; values do not enter it), so the bias equals C's bias.
                             Measures the mass/bias component with zero foreign content.
Random draws use a generator on the tensor's device seeded with  seed + 1000003 * item_key + 7919 * layer_idx; the runner sets
item_key (the item id) before every item.
"""
import dataclasses
from dataclasses import dataclass

import torch

from kvpress.presses.merging_any_press import _EPS, CompensationResult, KeptSet, MassFold


def _gen(device, seed: int, item_key: int, layer_idx: int) -> torch.Generator:
    g = torch.Generator(device=device)
    g.manual_seed(int(seed) + 1000003 * int(item_key) + 7919 * int(layer_idx))
    return g


@dataclass
class PlaceboMassFold(MassFold):
    seed: int = 20260930
    item_key: int = 0
    name = "placebo_mass_fold"

    def __call__(self, ks: KeptSet):
        if ks.scores is None:
            return CompensationResult(info={"skipped": "no full-length inner score exposed"})
        v = ks.values.float()
        g = _gen(v.device, self.seed, self.item_key, ks.layer_idx)
        z = torch.randn(v.shape, generator=g, device=v.device, dtype=torch.float32)
        vn = v.norm(dim=-1, keepdim=True)
        noise = z * (vn / z.norm(dim=-1, keepdim=True).clamp(min=1e-12))
        nn_ = noise.norm(dim=-1)
        vn_ = vn.squeeze(-1)
        nz = vn_ > 0
        relerr = ((nn_ - vn_).abs()[nz] / vn_[nz]).max() if bool(nz.any()) else torch.zeros((), device=v.device)
        cos = (noise * v).sum(-1)[nz] / (nn_[nz] * vn_[nz]).clamp(min=1e-12)
        res = MassFold.__call__(self, dataclasses.replace(ks, values=noise))
        res.info["pl_norm_relerr_max"] = float(relerr)
        res.info["pl_cos_abs_mean"] = float(cos.abs().mean()) if cos.numel() else 0.0
        return res


@dataclass
class PermutedMassFold(MassFold):
    seed: int = 20260931
    item_key: int = 0
    name = "permuted_mass_fold"

    def __call__(self, ks: KeptSet):
        if ks.scores is None:
            return CompensationResult(info={"skipped": "no full-length inner score exposed"})
        v = ks.values
        bsz, heads, l0, d = v.shape
        ev = ks.evicted
        ar = torch.arange(l0, device=v.device).expand(bsz, heads, l0)
        g = _gen(v.device, self.seed, self.item_key, ks.layer_idx)
        rnd = torch.rand((bsz, heads, l0), generator=g, device=v.device)
        rnd = torch.where(ev, rnd, torch.full_like(rnd, 2.0))  # evicted positions first, in random order
        rand_order = torch.sort(rnd, dim=-1, stable=True).indices
        nat_key = torch.where(ev, ar.to(torch.float32), torch.full_like(rnd, 3.0 * l0))  # evicted first, natural order
        nat_order = torch.sort(nat_key, dim=-1, stable=True).indices
        src_idx = torch.empty_like(rand_order)
        src_idx.scatter_(-1, nat_order, rand_order)  # evicted position -> random evicted source; kept rows -> themselves
        vp = v.gather(2, src_idx.unsqueeze(-1).expand(-1, -1, -1, d))
        n_ev = ev.sum()
        fixed = ((src_idx == ar) & ev).sum().float() / n_ev.clamp(min=1).float()
        moved_kept = ((src_idx != ar) & ~ev).sum()
        s0 = (v.float() * ev.unsqueeze(-1)).sum(2)
        s1 = (vp.float() * ev.unsqueeze(-1)).sum(2)
        relsum = (s1 - s0).norm() / s0.norm().clamp(min=1e-12)
        res = MassFold.__call__(self, dataclasses.replace(ks, values=vp))
        res.info["perm_fixed_frac"] = float(fixed)
        res.info["perm_moved_kept"] = int(moved_kept)
        res.info["perm_sum_relerr"] = float(relsum)
        return res


@dataclass
class ZeroBiasPathFold(MassFold):
    name = "zero_bias_path"

    def __call__(self, ks: KeptSet):
        res = MassFold.__call__(self, ks)
        if res.logit_bias is None:
            return res
        b = res.logit_bias
        info = dict(res.info)
        info["c0_force_bias_path"] = bool((b != 0).any())  # C would carry a bias on this layer
        info["c0_bias_nonzero_c"] = int((b != 0).sum())
        info["c0_bias_sum_c"] = float(b.double().sum())
        return CompensationResult(values=None, logit_bias=torch.zeros_like(b), info=info)


@dataclass
class SelfMassFold(MassFold):
    name = "self_mass_fold"

    def __call__(self, ks: KeptSet):
        if ks.scores is None:
            return CompensationResult(info={"skipped": "no full-length inner score exposed"})
        bsz, heads, l0, d = ks.keys.shape
        # routing: verbatim MassFold (most cosine-similar target key, chunked); values do not enter it
        tk = ks.cache_keys.float()
        tk = tk / tk.norm(dim=-1, keepdim=True).clamp(min=_EPS)
        ek = ks.keys.float()
        ek = ek / ek.norm(dim=-1, keepdim=True).clamp(min=_EPS)
        target_idx = torch.empty(bsz, heads, l0, dtype=torch.long, device=tk.device)
        for a in range(0, l0, self.chunk):
            sim = ek[:, :, a : a + self.chunk] @ tk.transpose(-2, -1)  # (B, H, c, L1)
            sim = sim.masked_fill(~ks.target.unsqueeze(2), float("-inf"))
            target_idx[:, :, a : a + self.chunk] = sim.argmax(-1)
        own = ks.cache_values.gather(2, target_idx.unsqueeze(-1).expand(-1, -1, -1, d))  # (B, H, L0, D) target's own value
        v_self = torch.where(ks.evicted.unsqueeze(-1), own.to(ks.values.dtype), ks.values)
        res = MassFold.__call__(self, dataclasses.replace(ks, values=v_self))
        if res.values is not None:  # self-fold identity check on every row: v_i' == v_i up to rounding
            diff = (res.values.float() - ks.cache_values.float()).abs().amax(-1)
            ref = ks.cache_values.float().abs().amax(-1).clamp(min=1e-12)
            res.info["b_self_maxrel"] = float((diff / ref).max())
            res.info["b_self_rows_changed"] = int((res.values != ks.cache_values).any(-1).sum())
        return res
