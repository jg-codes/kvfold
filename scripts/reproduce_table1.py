#!/usr/bin/env python
"""Rebuild Table 1 of the kvfold preprint from the stored per-item records (CPU only; numpy + pandas).

Benchmark: RULER-4096 - Model: Qwen3-8B (bf16, sdpa) - Metric: containment score 0-100 (the leaderboard's string-match
score), pooled as the mean of 13 task means - Direction: higher is better - Paired items at equal effective compression.

usage: python scripts/reproduce_table1.py [--data data/per_item] [--expected data/expected_table1.json] [--out table1_rebuilt.csv]

Reads data/per_item/ruler4096_qwen3-8b__<operating point>.csv.gz (columns item_id, task, arm, score, predicted_answer; arms
bare, fold, published) and ruler4096_qwen3-8b__full_cache.csv.gz. Interval: paired item bootstrap of the score difference,
B = 10,000, numpy default_rng; seed 0 for the RestoreKV_plus row (rows in file order), seed 20261004 for the other four rows,
resampled in chunks of 500 as in the statistics script of the paper. Gap closed = (fold - bare) / (full cache - bare), with
the full cache scored on the same items as the row. Exit status 1 if a rebuilt value differs from data/expected_table1.json
(the numbers printed in Table 1 of the paper) at two decimals.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PFX = "ruler4096_qwen3-8b__"
B_BOOT = 10000
# label (as in the paper), file stem, bootstrap seed, chunk size (None = one draw)
OPS = [
    ("RestoreKV_plus, CR 0.9375", "restorekv_plus_cr0.9375", 0, None),
    ("KVzap linear, threshold \u22123", "kvzap_linear_thr-3", 20261004, 500),
    ("KVzap MLP, threshold \u22123", "kvzap_mlp_thr-3", 20261004, 500),
    ("KVzip, CR 0.875", "kvzip_cr0.875", 20261004, 500),
    ("KVzip, CR 0.97", "kvzip_cr0.97", 20261004, 500),
]


def load(data, stem):
    return pd.read_csv(Path(data) / f"{PFX}{stem}.csv.gz", dtype={"predicted_answer": str}, keep_default_na=False)


def pooled(scores, tasks):
    scores, tasks = np.asarray(scores, float), np.asarray(tasks)
    return float(np.mean([scores[tasks == t].mean() for t in sorted(set(tasks))]))


def boot_ci(d, seed, chunk=None, B=B_BOOT):
    d = np.asarray(d, float)
    n = len(d)
    rng = np.random.default_rng(seed)
    if chunk is None:
        res = d[rng.integers(0, n, size=(B, n))].mean(axis=1)
    else:
        res = np.empty(B)
        for b0 in range(0, B, chunk):
            bb = min(chunk, B - b0)
            res[b0:b0 + bb] = d[rng.integers(0, n, size=(bb, n))].mean(axis=1)
    lo, hi = np.percentile(res, [2.5, 97.5])
    return float(lo), float(hi)


def table1(data):
    full = load(data, "full_cache").set_index("item_id")["score"]
    rows = []
    for label, stem, seed, chunk in OPS:
        df = load(data, stem)
        fold = df[df.arm == "fold"]
        ids = fold.item_id.to_numpy()
        tasks = fold.task.to_numpy()
        bare = df[df.arm == "bare"].set_index("item_id")["score"].loc[ids].to_numpy(float)
        pub = df[df.arm == "published"].set_index("item_id")["score"].loc[ids].to_numpy(float)
        fs = fold.score.to_numpy(float)
        fc = full.loc[ids].to_numpy(float)
        pb, pf, pp, pc = pooled(bare, tasks), pooled(fs, tasks), pooled(pub, tasks), pooled(fc, tasks)
        lo, hi = boot_ci(fs - bare, seed, chunk)
        rows.append({"operating_point": label, "items": len(ids), "published_bare": pp, "bare_this_stack": pb, "fold": pf,
                     "delta": pf - pb, "ci_lo": lo, "ci_hi": hi, "full_cache": pc, "gap_closed": (pf - pb) / (pc - pb)})
    return pd.DataFrame(rows)


def full_set(data):
    df = load(data, "restorekv_plus_cr0.9375_all6500")
    f, p = df[df.arm == "fold"], df[df.arm == "published"].set_index("item_id")["score"].loc[df[df.arm == "fold"].item_id]
    return {"items": len(f), "fold": pooled(f.score, f.task), "published_bare": pooled(p.to_numpy(float), f.task)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default=str(ROOT / "data" / "per_item"))
    ap.add_argument("--expected", default=str(ROOT / "data" / "expected_table1.json"))
    ap.add_argument("--out", default="table1_rebuilt.csv")
    args = ap.parse_args()
    t = table1(args.data)
    fs = full_set(args.data)
    t.to_csv(args.out, index=False, float_format="%.6f")
    print("Table 1. RULER-4096 | Qwen3-8B (bf16, sdpa) | containment score 0-100, pooled over 13 task means | higher is better | paired items at equal effective CR")
    print("| Press, operating point | Items | Published bare press | Bare press (this stack) | Fold | Fold - bare [95% CI] | Full cache | Gap closed |")
    print("|:--|--:|--:|--:|--:|--:|--:|--:|")
    for r in t.itertuples():
        mark = "\u2020" if r.operating_point.endswith("0.97") else ""
        print(f"| {r.operating_point} | {r.items:,} | {r.published_bare:.2f}{mark} | {r.bare_this_stack:.2f} | {r.fold:.2f} | {r.delta:+.2f} [{r.ci_lo:+.2f}, {r.ci_hi:+.2f}] | {r.full_cache:.2f} | {r.gap_closed:.2f} |")
    print("\u2020 published KVzip entry at CR 0.96875 (leaderboard label 0.97), scored on the same 1,300 items; no published row exists at CR 0.97.")
    print(f"All {fs['items']:,} items, RestoreKV_plus CR 0.9375: published {fs['published_bare']:.2f}, fold {fs['fold']:.2f}, difference {fs['fold'] - fs['published_bare']:+.2f}")
    if args.expected and Path(args.expected).exists():
        exp = json.load(open(args.expected))
        bad = 0
        for r in t.itertuples():
            e = exp[r.operating_point]
            got = {"items": r.items, "published_bare": r.published_bare, "bare_this_stack": r.bare_this_stack, "fold": r.fold, "delta": r.delta,
                   "ci_lo": r.ci_lo, "ci_hi": r.ci_hi, "full_cache": r.full_cache, "gap_closed": r.gap_closed}
            diff = {k: (round(float(v), 2), e[k]) for k, v in got.items() if abs(round(float(v), 2) - float(e[k])) > 0.0051}
            print("check", r.operating_point, "OK" if not diff else f"MISMATCH {diff}")
            bad += bool(diff)
        e = exp["_all6500"]
        d2 = {k: (round(fs[k], 2), e[k]) for k in ("fold", "published_bare") if abs(round(fs[k], 2) - e[k]) > 0.0051}
        print("check all 6,500 items", "OK" if not d2 else f"MISMATCH {d2}")
        bad += bool(d2)
        sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
