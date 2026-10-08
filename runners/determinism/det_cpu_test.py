"""det_cpu_test.py - operator-level bit-identity tests of the fold's scatter (CPU).  usage: python det_cpu_test.py out_json
Imports kvpress from PYTHONPATH (patched fork).  Every number printed is computed here; nothing is typed in."""
import hashlib, json, sys, time
import numpy as np
import torch
import kvpress.presses.merging_any_press as MAP
from kvpress.presses.merging_any_press import MassFold, KeptSet, scatter_add_atomic, scatter_add_sorted

def h(t):
    return hashlib.sha1(t.detach().contiguous().cpu().float().numpy().tobytes()).hexdigest()[:16]

def make_case(seed, B=1, H=8, L0=4096, L1=256, D=128):
    g = torch.Generator().manual_seed(seed)
    p = torch.rand(L1, generator=g) ** 3
    p = p / p.sum()
    idx = torch.multinomial(p.expand(B * H, L1), L0, replacement=True, generator=g).view(B, H, L0)
    w = torch.exp(torch.randn(B, H, L0, generator=g) * 2.0 - 4.0)
    vals = torch.randn(B, H, L0, D, generator=g).to(torch.bfloat16).float()
    return idx, w, vals

def reference_sequential(index, src, size):
    B, H, L0 = index.shape
    idx = index.numpy(); s = src.numpy()
    out = np.zeros((B, H, size) + tuple(src.shape[3:]), dtype=np.float32)
    for b in range(B):
        for hh in range(H):
            for j in range(L0):
                out[b, hh, idx[b, hh, j]] += s[b, hh, j]
    return torch.from_numpy(out)

def diff(a, b):
    ne = (a != b)
    d = (a - b).abs()
    return {"equal": bool(not ne.any()), "n_diff": int(ne.sum()), "n": int(a.numel()), "max_abs": float(d.max())}

res = {"torch": torch.__version__, "threads_default": torch.get_num_threads(), "tests": {}}
# A1-A3: realistic shapes
for seed in (0, 1, 2):
    idx, w, vals = make_case(seed)
    src_n = w.unsqueeze(-1) * vals
    sizes = {"num": (src_n, 256), "den": (w, 256)}
    for nm, (src, size) in sizes.items():
        a1 = scatter_add_atomic(idx, src, size); a2 = scatter_add_atomic(idx, src, size)
        s1 = scatter_add_sorted(idx, src, size); s2 = scatter_add_sorted(idx, src, size)
        res["tests"][f"seed{seed}_{nm}"] = {"atomic_twice": diff(a1, a2), "sorted_twice": diff(s1, s2), "atomic_vs_sorted": diff(a1, s1),
                                            "hash_atomic": h(a1), "hash_sorted": h(s1), "max_group": int(torch.bincount(idx[0, 0]).max())}
# A4: sequential numpy reference on a smaller case
idx, w, vals = make_case(7, B=1, H=2, L0=768, L1=48, D=16)
src_n = w.unsqueeze(-1) * vals
ref = reference_sequential(idx, src_n, 48)
res["tests"]["sequential_reference"] = {"atomic": diff(scatter_add_atomic(idx, src_n, 48), ref), "sorted": diff(scatter_add_sorted(idx, src_n, 48), ref)}
# A5: thread counts and the global deterministic flag
idx, w, vals = make_case(3)
src_n = w.unsqueeze(-1) * vals
ref_a = scatter_add_atomic(idx, src_n, 256); ref_s = scatter_add_sorted(idx, src_n, 256)
thr = {}
for t in (1, 2, 3, 6):
    torch.set_num_threads(t)
    thr[str(t)] = {"atomic_vs_ref": diff(scatter_add_atomic(idx, src_n, 256), ref_a)["equal"], "sorted_vs_ref": diff(scatter_add_sorted(idx, src_n, 256), ref_s)["equal"]}
torch.set_num_threads(res["threads_default"])
torch.use_deterministic_algorithms(True, warn_only=True)
thr["det_flag_on"] = {"atomic_vs_ref": diff(scatter_add_atomic(idx, src_n, 256), ref_a)["equal"], "sorted_vs_ref": diff(scatter_add_sorted(idx, src_n, 256), ref_s)["equal"]}
torch.use_deterministic_algorithms(False)
res["tests"]["threads_and_flag"] = thr
# A6: the fold itself on a synthetic kept set (MassFold.__call__), both cache dtypes
def make_keptset(seed, dtype, B=1, H=2, L0=1024, L1=96, D=64, n_sink=4):
    g = torch.Generator().manual_seed(seed)
    keys = torch.randn(B, H, L0, D, generator=g); values = torch.randn(B, H, L0, D, generator=g)
    scores = torch.rand(B, H, L0, generator=g) ** 2 + 1e-3
    src = torch.empty(B, H, L1, dtype=torch.long)
    for b in range(B):
        for hh in range(H):
            top = torch.topk(scores[b, hh, n_sink:], L1 - n_sink).indices + n_sink
            src[b, hh] = torch.cat([torch.arange(n_sink), top.sort().values])
    gi = src.unsqueeze(-1).expand(-1, -1, -1, D)
    ck = torch.gather(keys, 2, gi).to(dtype); cv = torch.gather(values, 2, gi).to(dtype)
    evicted = torch.ones(B, H, L0, dtype=torch.bool); evicted.scatter_(2, src, False)
    return KeptSet(layer_idx=0, module=None, keys=keys.to(dtype), values=values.to(dtype), cache_keys=ck, cache_values=cv, src=src,
                   kept=torch.ones(B, H, L1, dtype=torch.bool), target=src >= n_sink, evicted=evicted, path="in-place", scores=scores)
fold = {}
for dt_name, dt in (("float32", torch.float32), ("bfloat16", torch.bfloat16)):
    ks = make_keptset(11, dt)
    ra = MassFold(score_map="log", scatter="atomic")(ks); ra2 = MassFold(score_map="log", scatter="atomic")(ks)
    rs = MassFold(score_map="log", scatter="sorted")(ks); rs2 = MassFold(score_map="log", scatter="sorted")(ks)
    fold[dt_name] = {"values_atomic_twice": diff(ra.values.float(), ra2.values.float()), "values_sorted_twice": diff(rs.values.float(), rs2.values.float()),
                     "values_atomic_vs_sorted": diff(ra.values.float(), rs.values.float()), "bias_atomic_vs_sorted": diff(ra.logit_bias, rs.logit_bias),
                     "bias_atomic_twice": diff(ra.logit_bias, ra2.logit_bias), "merged": ra.info["merged"], "bias_nonzero": int((ra.logit_bias != 0).sum())}
res["tests"]["massfold_synthetic"] = fold
json.dump(res, open(sys.argv[1], "w"), indent=1)
print(json.dumps({k: (v if k != "tests" else {kk: ("ok" if isinstance(vv, dict) else vv) for kk, vv in v.items()}) for k, v in res.items()}))
