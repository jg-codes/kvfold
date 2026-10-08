"""det_gpu_microbench.py - the fold's scatter variants on the GPU: run-to-run identity and cost. usage: python det_gpu_microbench.py out.json"""
import hashlib, json, os, sys, time
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from det_scatter_lib import scatter_add_atomic, scatter_add_sorted
dev = torch.device("cuda")
def h(t):
    return hashlib.sha1(t.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()[:16]
def make_case(seed, B=1, H=8, L0=4096, L1=256, D=128):
    g = torch.Generator().manual_seed(seed)
    p = torch.rand(L1, generator=g) ** 3; p = p / p.sum()
    idx = torch.multinomial(p.expand(B * H, L1), L0, replacement=True, generator=g).view(B, H, L0)
    w = torch.exp(torch.randn(B, H, L0, generator=g) * 2.0 - 4.0)
    vals = torch.randn(B, H, L0, D, generator=g).to(torch.bfloat16).float()
    return idx, w, vals
N = 40
res = {"device": torch.cuda.get_device_name(0), "torch": torch.__version__, "cuda": torch.version.cuda, "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"), "n_calls": N, "cases": {}}
variants = [("atomic_flag_off", scatter_add_atomic, False), ("atomic_flag_on", scatter_add_atomic, True), ("sorted_flag_off", scatter_add_sorted, False), ("sorted_flag_on", scatter_add_sorted, True)]
for seed in (0, 1, 2):
    idx, w, vals = [t.to(dev) for t in make_case(seed)]
    for nm, src in (("num", w.unsqueeze(-1) * vals), ("den", w)):
        row = {"max_group": int(torch.bincount(idx[0, 0]).max())}
        first = {}
        for vname, fn, flag in variants:
            torch.use_deterministic_algorithms(flag, warn_only=True)
            hs = [h(fn(idx, src, 256)) for _ in range(N)]
            first[vname] = fn(idx, src, 256)
            torch.cuda.synchronize(); t0 = time.time()
            for _ in range(N):
                fn(idx, src, 256)
            torch.cuda.synchronize()
            row[vname] = {"distinct_hashes": len(set(hs)), "ms_per_call": round((time.time() - t0) / N * 1000, 3)}
        torch.use_deterministic_algorithms(False)
        def dd(a, b):
            ne = (first[a] != first[b]); return {"n_diff": int(ne.sum()), "n": int(ne.numel()), "max_abs": float((first[a] - first[b]).abs().max())}
        row["atomic_on_vs_sorted_on"] = dd("atomic_flag_on", "sorted_flag_on")
        row["atomic_off_vs_atomic_on"] = dd("atomic_flag_off", "atomic_flag_on")
        row["sorted_off_vs_sorted_on"] = dd("sorted_flag_off", "sorted_flag_on")
        res["cases"][f"seed{seed}_{nm}"] = row
json.dump(res, open(sys.argv[1], "w"), indent=1)
print(json.dumps({k: {v: (r[v]["distinct_hashes"], r[v]["ms_per_call"]) for v in ("atomic_flag_off", "atomic_flag_on", "sorted_flag_off", "sorted_flag_on")} for k, r in res["cases"].items()}))
