#!/usr/bin/env python
"""KVzap smoke (wiki claim C111, pre-registration hypothesis_C111.json): MergingAnyPress over the leaderboard KVzap press.

Inner press P (verbatim press_init_command of hf.co/spaces/nvidia/kvpress-leaderboard
benchmark/ruler__4096__Qwen--Qwen3-8B__kvzap_<type>__-3.00/config.yaml):
    DMSPress(press=KVzapPress(compression_ratio=0.0, model_type=<type>, kvzip_model_name=None), threshold=-3,
             sliding_window_size=128, decoding=False)
Arms (one operating point per invocation, run as stages in the order given by --arms):
  A  P (bare)
  Z  MergingAnyPress(P, compensation='none', n_sink=4)            specificity: predictions must equal A exactly
  C  MergingAnyPress(P, compensation=MassFold(score_map='identity'), n_sink=4)
        C86 mass fold + score bias, s = raw KVzap output (KVzap regresses log s+, arXiv 2601.07891 s3.3)
  F  MergingAnyPress(P, compensation=FittedCompensation(arm='penalised', reference='window', query_window=256, n_sink=4,
        warm_start='mass_fold', score_map='identity'), n_sink=4)   context-only last-256 reference queries
Revision 1.1 (addendum of hypothesis_C111.json rev 1.1; the A, Z, C and F code paths are unchanged):
  AV P (bare) re-run on the first --av-items ids of a resumed run. Kept counts, masked index sets, effective CR and
     prediction strings are compared with the saved arm-A rows; the AV gate (all kept sets identical, >= --av-min-pred-match
     identical strings) must pass before PL is paired with the saved A and C rows.
  CV C (identical construction) re-run on the same first --av-items ids: run-to-run reproducibility of the float-mask
     path, read-out against the saved arm-C rows (not a stopping gate).
  C0 MergingAnyPress(P, compensation=ZeroBiasPathFold(score_map='identity'), n_sink=4): C's biased-mask attention path with
     the bias forced to 0 (installed on every layer where C's bias is non-zero) and no value change (kernel-path control).
  Cperm MergingAnyPress(P, compensation=PermutedMassFold(score_map='identity'), n_sink=4): C with the evicted value rows
     permuted within each (layer, kv-head) before folding; routing, weights and bias unchanged.
  PL MergingAnyPress(P, compensation=PlaceboMassFold(score_map='identity'), n_sink=4)   zero-content placebo (control_folds.py):
     C's routing, weights and bias with norm-matched Gaussian values in place of the evicted values.
  Priority when the cost cap binds (pre-registered, parent instruction 2026-09-30): AV, CV, C0, Cperm, PL.
  G6 C0/Cperm/PL vs the saved C row of the same item: merged rows equal, C's non-zero bias entries reproduced within
     --g6-bias-tol (relative). PL: replacement rows norm-matched (max relative error <= 1e-3). Cperm: kept rows unmoved,
     per-head sums of the evicted rows preserved (relative error <= 1e-3). PL/Cperm: value writes > 0 whenever merged > 0.
     C0: zero value writes, zero bias entries, and the zero bias installed on >= 1 layer whenever merged > 0.
  JOB_T0 (environment, epoch seconds of the job start) anchors the time budget and the cost guard when several runners share
  one job; the first-item watchdog stays anchored at this script's start.
Protocol: kvpress pipeline, the press compresses the CONTEXT only; question + answer prefix are prefilled afterwards.
Read-out: greedy RULER string_match (leaderboard scorer), per-task max_new_tokens of the dataset.
Gates (asserted in-script; a failure writes gates.json and exits 3):
  G1 per-(layer, kv-head) kept counts AND kept sets (sha1 of the masked index sets) identical to arm A of the same item
  G2 effective CR equal to arm A to 1e-6 (1 - kept / (layers * kv_heads * ctx_len))
  G3 after stage A: |pooled(A) - pooled(published)| <= floor and exact prediction-string match >= 0.5 on the same ids, and
     |mean effective CR(A) - mean published CR| <= 0.01; assert mode ends the job before any treatment stage
  G4 treatment arms: zero value writes / bias entries on sinks (src < n_sink), masked rows or appended rows; keys unchanged
  G5 C: scores captured on every evicting layer; merged > 0 => bias non-zero > 0 and the biased-mask path called > 0 times.
     F: bias non-zero > 0 => biased-mask calls > 0. A/Z: no bias set, biased-mask calls == 0, no value writes.
  SPEC heads with no evicted row receive zero writes and zero bias (all treatment arms); Z predictions == A predictions.
Economy: container-start / model-loaded / first-item timestamps; abort if the first item has not started 300 s after start;
nvidia-smi sampled every 60 s; rows appended + gates.json rewritten after every item (checkpoint), --resume from a previous
rows.jsonl; a time budget and a list-price cost guard stop the job between items/stages.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback

T0 = time.time()
JOB_T0 = float(os.environ.get("JOB_T0", T0))  # rev 1.1: job start when several runners share one GPU job

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

# ----------------------------------------------------------------------------- scorer (verbatim ruler_score.py)
QA_TASKS = {"qa_1", "qa_2"}


def golds(a):
    g = re.findall(r"'([^']*)'", str(a))
    return g or [str(a).strip("[]").strip()]


def ruler_score(pred, answers, task):
    p = str(pred).lower()
    gs = golds(answers)
    if not gs:
        return 0.0
    h = [1.0 if str(g).lower() in p else 0.0 for g in gs]
    return 100.0 * (max(h) if task in QA_TASKS else float(np.mean(h)))


# ----------------------------------------------------------------------------- args
ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3-8B")
ap.add_argument("--op", choices=["linear", "mlp"], required=True)
ap.add_argument("--threshold", type=float, default=-3.0)
ap.add_argument("--arms", default="A,Z,C,F")
ap.add_argument("--z-items", type=int, default=26, help="arm Z runs on the first N items only")
ap.add_argument("--item-ids", required=True)
ap.add_argument("--max-items", type=int, default=None, help="dry-run only")
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
ap.add_argument("--dtype", default="auto")
ap.add_argument("--attn", default="sdpa")
ap.add_argument("--max-context-length", type=int, default=None)
ap.add_argument("--max-new-tokens", type=int, default=None)
ap.add_argument("--leaderboard", default=None)
ap.add_argument("--g3-floor-pp", type=float, default=10.22)
ap.add_argument("--g3-mode", choices=["assert", "report"], default="assert")
ap.add_argument("--mock-kvzap", action="store_true", help="dry-run: random KVzap surrogate (tests/presses pattern)")
ap.add_argument("--out", default="out")
ap.add_argument("--resume", default=None, help="rows.jsonl of a previous job (copied into --out and continued)")
ap.add_argument("--time-budget-s", type=float, default=3300.0)
ap.add_argument("--watchdog-s", type=float, default=300.0)
ap.add_argument("--stop-after-items", type=int, default=None, help="dry-run resume test: stop after N new item-arms")
ap.add_argument("--usd-per-s", type=float, default=0.000902, help="list-price container rate (A100-40GB + 4 cores + 24 GiB, sandbox tier)")
ap.add_argument("--eur-usd", type=float, default=1.17)
ap.add_argument("--spent-eur", type=float, default=0.0, help="list-price spend of earlier jobs of this task")
ap.add_argument("--cost-cap-eur", type=float, default=float("inf"), help="optional stage/item guard in EUR at list price")
ap.add_argument("--startup-overhead-s", type=float, default=60.0)
ap.add_argument("--av-items", type=int, default=10, help="rev 1.1: arm AV re-runs bare P on the first N ids")
ap.add_argument("--av-min-pred-match", type=int, default=9, help="rev 1.1: AV gate, identical prediction strings needed")
ap.add_argument("--pl-seed", type=int, default=20260930, help="rev 1.1: PlaceboMassFold seed")
ap.add_argument("--g6-bias-tol", type=float, default=0.001, help="rev 1.1: G6 relative tolerance on non-zero bias entries")
args = ap.parse_args()
assert "AV" not in args.arms.split(",") or args.av_min_pred_match <= args.av_items, "AV threshold exceeds --av-items"

os.makedirs(args.out, exist_ok=True)
LOG = open(os.path.join(args.out, "progress.log"), "a")


def log(*a):
    s = f"[{time.time() - T0:7.1f}s] " + " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.write(s + "\n")
    LOG.flush()


log(f"container-start (script start) T0={T0:.1f} argv={sys.argv[1:]}")
torch.manual_seed(0)
np.random.seed(0)

# ----------------------------------------------------------------------------- watchdog + nvidia-smi sampling
STATE = {"first_item_started": None, "stop": False}


def _watchdog():
    while STATE["first_item_started"] is None:
        if time.time() - T0 > args.watchdog_s:
            log(f"WATCHDOG: first item not started within {args.watchdog_s:.0f} s of container start; aborting")
            LOG.flush()
            os._exit(4)
        time.sleep(5)


threading.Thread(target=_watchdog, daemon=True).start()
NVSMI_PATH = os.path.join(args.out, "nvsmi.csv")


def _nvsmi():
    if shutil.which("nvidia-smi") is None:
        return
    with open(NVSMI_PATH, "a") as fh:
        while not STATE["stop"]:
            try:
                r = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                                   capture_output=True, text=True, timeout=20)
                fh.write(f"{time.time() - T0:.0f},{r.stdout.strip()}\n")
                fh.flush()
            except Exception as e:  # sampling must never kill the job
                fh.write(f"{time.time() - T0:.0f},ERR {e}\n")
            for _ in range(60):
                if STATE["stop"]:
                    break
                time.sleep(1)


threading.Thread(target=_nvsmi, daemon=True).start()

# ----------------------------------------------------------------------------- kvpress imports + instrumentation
import importlib.metadata as _md  # noqa: E402

import kvpress  # noqa: E402
import kvpress.presses.merging_any_press as map_mod  # noqa: E402
from kvpress import DMSPress, KVzapPress  # noqa: E402
from kvpress.presses.fitted_compensation import FittedCompensation  # noqa: E402
from kvpress.presses.kvzap_press import KVzapConfig, KVzapModel  # noqa: E402
from kvpress.presses.merging_any_press import MassFold, MergingAnyPress  # noqa: E402
from control_folds import PermutedMassFold, PlaceboMassFold, ZeroBiasPathFold  # noqa: E402  (rev 1.1)
from transformers import DynamicCache, pipeline  # noqa: E402

SRC_SHA = {n: hashlib.sha256(open(m.__file__, "rb").read()).hexdigest() for n, m in
           [("merging_any_press.py", map_mod), ("fitted_compensation.py", sys.modules["kvpress.presses.fitted_compensation"]),
            ("kvzap_press.py", sys.modules["kvpress.presses.kvzap_press"]), ("dms_press.py", sys.modules["kvpress.presses.dms_press"]),
            ("pipeline.py", sys.modules["kvpress.pipeline"]), ("run_kvzap_smoke.py", sys.modules["__main__"])]}
log(f"kvpress from {kvpress.__file__}; sha256 " + json.dumps({k: v[:16] for k, v in SRC_SHA.items()}))

# score map 'identity': s = raw inner score (KVzap output is already log s+). Fixed before any outcome (hypothesis_C111.json).
_orig_map_scores = map_mod.map_scores


def map_scores_ext(scores, score_map):
    if score_map == "identity":
        return scores.float()
    return _orig_map_scores(scores, score_map)


map_mod.map_scores = map_scores_ext

N_SINK = 4
ITEM = {}  # per item-arm instrumentation counters


def new_item_counters():
    return {"apply_calls": 0, "writes": 0, "bias_nonzero": 0, "merged": 0, "w_sink": 0, "w_masked": 0, "w_appended": 0,
            "b_sink": 0, "b_masked": 0, "b_appended": 0, "keys_changed": 0, "noevict_heads": 0, "noevict_heads_touched": 0,
            "evicting_layers_seen": 0, "scores_missing_layers": 0, "heads_applied": 0, "heads_fold": 0, "bias_mask_calls": 0,
            "skipped_layers": 0, "passthrough_layers": 0, "bias_sum": 0.0}


_orig_apply = map_mod.MergingAnyPress._apply


def apply_recording(self, module, layer_idx, ks, res, rec):
    c = ITEM
    c["apply_calls"] += 1
    new_v = ks.cache_values if res.values is None else res.values
    changed = (new_v != ks.cache_values).any(-1)  # (B, H, L1)
    bias = res.logit_bias
    bnz = (bias != 0) if bias is not None else torch.zeros_like(changed)
    sink = (ks.src >= 0) & (ks.src < N_SINK)
    masked = (ks.src >= 0) & ~ks.kept
    appended = ks.src < 0
    c["writes"] += int(changed.sum())
    c["bias_nonzero"] += int(bnz.sum())
    c["bias_sum"] = c.get("bias_sum", 0.0) + (float(bias.double().sum()) if bias is not None else 0.0)
    c["w_sink"] += int((changed & sink).sum())
    c["w_masked"] += int((changed & masked).sum())
    c["w_appended"] += int((changed & appended).sum())
    c["b_sink"] += int((bnz & sink).sum())
    c["b_masked"] += int((bnz & masked).sum())
    c["b_appended"] += int((bnz & appended).sum())
    ev_head = ks.evicted.sum(-1)  # (B, H)
    touched = (changed | bnz).any(-1)  # (B, H)
    c["noevict_heads"] += int((ev_head == 0).sum())
    c["noevict_heads_touched"] += int(((ev_head == 0) & touched).sum())
    c["merged"] += int(res.info.get("merged", 0) or 0)
    c["heads_applied"] += int(res.info.get("heads_applied", 0) or 0)
    c["heads_fold"] += int(res.info.get("heads_fold", 0) or 0)
    if bias is not None and bool(bnz.any()):  # rev 1.1: bias magnitudes and window share (reviewer instrumentation)
        c.setdefault("_b_vals", []).append(bias[bnz].detach().float().cpu())
        win = (ks.src >= ks.keys.shape[2] - 128) & bnz
        c["b_in_window"] = c.get("b_in_window", 0) + int(win.sum())
    keys_before = ks.cache_keys.detach().clone()
    out = _orig_apply(self, module, layer_idx, ks, res, rec)
    k_after = self._cache.layers[layer_idx].keys
    c["keys_changed"] += int((k_after != keys_before).any(-1).sum())
    if res.info.get("c0_force_bias_path"):  # rev 1.1 arm C0: take C's biased-mask path with a zero bias
        tgt = ks.target.nonzero()
        if len(tgt):
            b_, h_, r_ = (int(i) for i in tgt[0])  # a target row: kept, never masked, never rewritten
            module.anypress_bias_probe = (b_, h_, r_, ks.cache_keys[b_, h_, r_].detach().clone())
            module.anypress_logit_bias = res.logit_bias.float()
            c["c0_layers_forced"] = c.get("c0_layers_forced", 0) + 1
        c["c0_bias_nonzero_c"] = c.get("c0_bias_nonzero_c", 0) + int(res.info.get("c0_bias_nonzero_c", 0))
        c["c0_bias_sum_c"] = c.get("c0_bias_sum_c", 0.0) + float(res.info.get("c0_bias_sum_c", 0.0))
    for k in ("perm_moved_kept",):
        if k in res.info:
            c[k] = c.get(k, 0) + int(res.info[k])
    for k in ("perm_sum_relerr", "perm_fixed_frac"):
        if k in res.info:
            c[k] = max(c.get(k, 0.0), float(res.info[k]))
    return out


map_mod.MergingAnyPress._apply = apply_recording

_orig_biased_mask = map_mod._biased_mask


def biased_mask_counting(*a, **kw):
    ITEM["bias_mask_calls"] = ITEM.get("bias_mask_calls", 0) + 1
    return _orig_biased_mask(*a, **kw)


map_mod._biased_mask = biased_mask_counting


class MockKVzapPress(KVzapPress):
    """Dry-run only (tests/presses/default_presses.py pattern): random linear surrogate instead of the HF checkpoint."""

    def post_init_from_model(self, model):
        if getattr(self, "kvzap_model", None) is None:
            g = torch.random.fork_rng()
            with g:
                torch.manual_seed(1234)
                cfg = KVzapConfig(input_dim=model.config.hidden_size, output_dim=model.config.num_key_value_heads,
                                  hidden_dim=None, n_modules=model.config.num_hidden_layers)
                self.kvzap_model = KVzapModel(cfg)
                for lin in self.kvzap_model.layers:  # scale so that threshold 0 evicts roughly half
                    torch.nn.init.normal_(lin.weight, std=1.0 / model.config.hidden_size ** 0.5)
                    torch.nn.init.zeros_(lin.bias)


def make_inner():
    cls = MockKVzapPress if args.mock_kvzap else KVzapPress
    # verbatim leaderboard press_init_command (kvzip_model_name is init=False in the dataclass; left at its default None)
    return DMSPress(press=cls(compression_ratio=0.0, model_type=args.op), threshold=args.threshold,
                    sliding_window_size=128, decoding=False)


def make_arm(a):
    if a in ("A", "AV"):
        return make_inner()
    if a == "PL":  # rev 1.1: zero-content placebo of C (same routing, weights, bias; norm-matched Gaussian values)
        return MergingAnyPress(make_inner(), compensation=PlaceboMassFold(score_map="identity", seed=args.pl_seed),
                               n_sink=N_SINK)
    if a == "Cperm":  # rev 1.1: evicted values permuted within each (layer, kv-head) before folding
        return MergingAnyPress(make_inner(), compensation=PermutedMassFold(score_map="identity", seed=args.pl_seed + 1),
                               n_sink=N_SINK)
    if a == "C0":  # rev 1.1: C's biased-mask path with a zero bias and no value change
        return MergingAnyPress(make_inner(), compensation=ZeroBiasPathFold(score_map="identity"), n_sink=N_SINK)
    if a == "CV":  # rev 1.1: C re-run (identical construction to arm C below)
        return MergingAnyPress(make_inner(), compensation=MassFold(score_map="identity"), n_sink=N_SINK)
    if a == "Z":
        return MergingAnyPress(make_inner(), compensation="none", n_sink=N_SINK)
    if a == "C":
        return MergingAnyPress(make_inner(), compensation=MassFold(score_map="identity"), n_sink=N_SINK)
    if a == "F":
        comp = FittedCompensation(arm="penalised", reference="window", query_window=256, n_sink=N_SINK,
                                  warm_start="mass_fold", score_map="identity")
        return MergingAnyPress(make_inner(), compensation=comp, n_sink=N_SINK)
    raise SystemExit(f"unknown arm {a}")


def attn_modules(model):
    lm = model.model.language_model if hasattr(model.model, "language_model") else model.model
    return [layer.self_attn for layer in lm.layers]


def reset_modules(model):
    for m in attn_modules(model):
        m.masked_key_indices = None
        for n in ("anypress_logit_bias", "anypress_bias_probe", "merge_logit_bias"):
            if getattr(m, n, None) is not None:
                setattr(m, n, None)


def kept_record(model, cache):
    ctx_len = int(cache.get_seq_length())
    kept, mask_h = [], hashlib.sha1()
    evicting = 0
    n_heads = None
    for m in attn_modules(model):
        k = cache.layers[int(m.layer_idx)].keys
        n_heads = int(k.shape[1])
        assert int(k.shape[2]) == ctx_len, "cache length differs across layers"
        mk = getattr(m, "masked_key_indices", None)
        if mk is None or len(mk[0]) == 0:
            nm = torch.zeros(n_heads, dtype=torch.long)
            mask_h.update(b"none")
        else:
            evicting += 1
            h_idx, t_idx = mk[1].cpu().long(), mk[2].cpu().long()
            nm = torch.bincount(h_idx, minlength=n_heads)
            flat = torch.unique(h_idx * (ctx_len + 1) + t_idx)
            assert flat.numel() == h_idx.numel(), "duplicate masked indices"
            mask_h.update(flat.numpy().tobytes())
        kept.append((ctx_len - nm).tolist())
    kept_total = int(sum(sum(x) for x in kept))
    n_layers = len(kept)
    return {"ctx_len": ctx_len, "n_layers": n_layers, "n_kv_heads": n_heads, "kept": kept, "kept_total": kept_total,
            "masked_total": n_layers * n_heads * ctx_len - kept_total, "evicting_layers": evicting,
            "mask_sha1": mask_h.hexdigest()[:16], "kept_sha1": hashlib.sha1(json.dumps(kept).encode()).hexdigest()[:16],
            "effective_cr": 1.0 - kept_total / float(n_layers * n_heads * ctx_len)}


# ----------------------------------------------------------------------------- data
from huggingface_hub import hf_hub_download  # noqa: E402

pq = hf_hub_download("simonjegou/ruler", "4096/test-00000-of-00001.parquet", repo_type="dataset")
df = pd.read_parquet(pq)
df["item_id"] = np.arange(len(df))
ids_src = json.load(open(args.item_ids))
ids = [int(i) for i in (ids_src["item_ids"] if isinstance(ids_src, dict) else ids_src)]
assert len(set(ids)) == len(ids) and max(ids) < len(df)
if args.max_items:
    ids = ids[: args.max_items]
IDS_SHA = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
sample = df.loc[ids].reset_index(drop=True)
log(f"dataset rows={len(df)} items={len(sample)} ids_sha256={IDS_SHA[:16]} tasks={sample.task.value_counts().to_dict()}")

lb = None
if args.leaderboard is None:
    args.leaderboard = f"leaderboard/kvzap_{args.op}__-3__predictions.csv"
if os.path.exists(args.leaderboard):
    lb = pd.read_csv(args.leaderboard)
    assert len(lb) == len(df), f"leaderboard rows {len(lb)} != dataset rows {len(df)}"
    mis = int((lb["question"].astype(str).values[ids] != df["question"].astype(str).values[ids]).sum())
    mis += int((lb["task"].astype(str).values[ids] != df["task"].astype(str).values[ids]).sum())
    log(f"leaderboard {args.leaderboard} alignment mismatches on ids={mis}")
    assert mis == 0, "leaderboard misaligned with the dataset"
else:
    log(f"leaderboard file not found: {args.leaderboard}")
    assert args.g3_mode == "report", "G3 in assert mode needs the published predictions"

# ----------------------------------------------------------------------------- state / resume
ROWS_PATH = os.path.join(args.out, "rows.jsonl")
GATES_PATH = os.path.join(args.out, "gates.json")
rows = []
if args.resume and os.path.abspath(args.resume) != os.path.abspath(ROWS_PATH):
    shutil.copy(args.resume, ROWS_PATH)
if args.resume and os.path.exists(ROWS_PATH):
    with open(ROWS_PATH) as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    rows = [r for r in rows if r["op"] == args.op]
    log(f"resume: {len(rows)} rows for op={args.op} loaded")
elif os.path.exists(ROWS_PATH):
    os.remove(ROWS_PATH)
done = {(r["arm"], int(r["item_id"])) for r in rows}
A_REF = {int(r["item_id"]): r for r in rows if r["arm"] == "A"}
C_REF = {int(r["item_id"]): r for r in rows if r["arm"] == "C"}  # rev 1.1: G6 reference for PL
ROWS_FH = open(ROWS_PATH, "a")

gates = {"claim": "C111", "op": args.op, "threshold": args.threshold, "ids_sha256": IDS_SHA, "src_sha256": SRC_SHA,
         "installed_dist_kvpress": _md.version("kvpress"), "kvpress_file": kvpress.__file__, "argv": sys.argv[1:],
         "G1": {"status": "pass", "items_checked": 0}, "G2": {"status": "pass", "items_checked": 0, "max_abs_diff": 0.0},
         "G3": {"status": "not run", "mode": args.g3_mode},
         "G4": {"status": "pass", "item_arms_checked": 0}, "G5": {"status": "pass", "item_arms_checked": 0},
         "SPEC": {"status": "pass", "z_items_checked": 0, "noevict_heads": 0, "noevict_heads_touched": 0},
         "stages": {}, "timestamps": {}, "cost": {}}
if os.path.exists(GATES_PATH) and args.resume:
    old = json.load(open(GATES_PATH))
    if old.get("op") == args.op:
        for k in ("G1", "G2", "G3", "G4", "G5", "SPEC", "stages"):
            gates[k] = old.get(k, gates[k])
        gates["previous_jobs"] = old.get("previous_jobs", []) + [old.get("timestamps", {})]
        for k in ("AV", "G6", "CV"):
            if k in old:
                gates[k] = old[k]
gates.setdefault("AV", {"status": "not run", "items_checked": 0, "mask_equal": 0, "kept_equal": 0, "effcr_equal": 0,
                        "pred_equal": 0, "score_equal": 0, "min_pred_match": args.av_min_pred_match, "items": []})
gates.setdefault("G6", {"status": "pass", "items_checked": 0, "merged_equal": 0, "bias_nonzero_equal": 0,
                        "bias_nonzero_max_rel_diff": 0.0, "norm_relerr_max": 0.0, "tol": args.g6_bias_tol})
gates["job_t0"] = JOB_T0


def flush():
    gates["wall_s"] = time.time() - T0
    gates["cost"] = {"job_list_price_eur": job_cost_eur(), "spent_before_eur": args.spent_eur,
                     "usd_per_s": args.usd_per_s, "eur_usd": args.eur_usd, "label": "list-price estimate"}
    json.dump(gates, open(GATES_PATH + ".tmp", "w"), indent=1, default=str)
    os.replace(GATES_PATH + ".tmp", GATES_PATH)


def job_cost_eur(extra_s=0.0):
    return (time.time() - JOB_T0 + args.startup_overhead_s + extra_s) * args.usd_per_s / args.eur_usd


def gate_fail(name, msg):
    gates[name]["status"] = "FAIL"
    gates[name].setdefault("failures", []).append(msg)
    log(f"{name} FAIL: {msg}")
    flush()
    STATE["stop"] = True
    sys.exit(3)


# ----------------------------------------------------------------------------- rev 1.1: exit before the model load if idle
_pending = sum(len([i for i in (ids[: args.z_items] if a == "Z" else ids[: args.av_items] if a in ("AV", "CV") else ids)
                    if (a, i) not in done]) for a in args.arms.split(","))
if _pending == 0 or time.time() - JOB_T0 > args.time_budget_s - 60.0:
    gates["stop_reason"] = "nothing to run" if _pending == 0 else "time budget reached before the model load"
    gates["complete"] = _pending == 0
    STATE["stop"] = True
    flush()
    log(f"early exit: {gates['stop_reason']} (pending item-arms {_pending})")
    sys.exit(0)

# ----------------------------------------------------------------------------- model
dtype = {"auto": "auto", "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
pipe = pipeline("kv-press-text-generation", model=args.model, model_kwargs={"dtype": dtype, "attn_implementation": args.attn},
                device=args.device)
pipe.model.eval()
gates["timestamps"]["model_loaded_s"] = time.time() - T0
log(f"model-loaded {args.model} dtype={pipe.model.dtype} attn={pipe.model.config._attn_implementation}")
flush()

PRIOR_S = {"A": 3.4, "Z": 3.4, "C": 4.97, "F": 12.0}  # s/item-arm priors for the cost guard (conventions s6; F unmeasured)
ADDENDUM_ARMS = ("AV", "CV", "C0", "Cperm", "PL")  # rev 1.1, priority order when the cap binds
PRIOR_S.update({"AV": 2.3, "CV": 3.6, "C0": 3.6, "Cperm": 3.6, "PL": 3.6})  # rev 1.1: measured C111 rates A 2.21 / C 3.53 s (linear), rounded up


def g3_check():
    a = pd.DataFrame([r for r in rows if r["arm"] == "A"])
    if lb is None or len(a) == 0:
        gates["G3"] = {"status": "not run", "mode": args.g3_mode}
        return True
    diff = float(a.score.mean() - a.lb_score.mean())
    match = float((a.predicted_answer.astype(str).str.strip() == a.lb_predicted_answer.astype(str).str.strip()).mean())
    cr_diff = float(a.effective_cr.mean() - a.lb_cr.mean())
    ok = abs(diff) <= args.g3_floor_pp and match >= 0.5 and abs(cr_diff) <= 0.01
    gates["G3"] = {"status": "pass" if ok else "FAIL", "mode": args.g3_mode, "ours_pooled": float(a.score.mean()),
                   "published_pooled": float(a.lb_score.mean()), "diff_pp": diff, "exact_pred_match": match,
                   "eff_cr_ours": float(a.effective_cr.mean()), "eff_cr_published": float(a.lb_cr.mean()), "eff_cr_diff": cr_diff,
                   "per_item_cr_max_abs_diff": float((a.effective_cr - a.lb_cr).abs().max()),
                   "n_items": int(len(a)), "floor_pp": args.g3_floor_pp,
                   "per_task_diff_pp": (a.groupby("task").score.mean() - a.groupby("task").lb_score.mean()).round(3).to_dict()}
    log(f"G3: ours={a.score.mean():.3f} published={a.lb_score.mean():.3f} diff={diff:+.3f} match={match:.3f} "
        f"effCR ours={a.effective_cr.mean():.5f} pub={a.lb_cr.mean():.5f} -> {gates['G3']['status']}")
    return ok


# ----------------------------------------------------------------------------- stages
arms_order = args.arms.split(",")
assert arms_order[0] == "A" or all(i in A_REF for i in ids), "arm A must run first (or be resumed) before treatment arms"
new_item_arms = 0
stop_reason = None
for arm in arms_order:
    stage_ids = ids[: args.z_items] if arm == "Z" else (ids[: args.av_items] if arm in ("AV", "CV") else ids)
    todo = [i for i in stage_ids if (arm, i) not in done]
    st = gates["stages"].setdefault(arm, {"items_total": len(stage_ids)})
    if arm != "A":
        missing = [i for i in stage_ids if i not in A_REF]
        if missing:
            stop_reason = f"arm A incomplete ({len(missing)} ids) - treatment stage {arm} not started"
            break
        if gates["G3"].get("status") != "pass" and args.g3_mode == "assert":
            if not g3_check():
                gate_fail("G3", "bare arm does not reproduce the published predictions; no treatment arm is run")
    if arm in ("CV", "C0", "Cperm", "PL"):  # rev 1.1: paired with the saved A and C rows only after the AV gate passed
        if "AV" in arms_order and gates["AV"].get("status") != "pass":
            stop_reason = f"AV gate status {gates['AV'].get('status')!r}: {arm} not started"
            break
        missing_c = [i for i in stage_ids if i not in C_REF]
        if missing_c:
            stop_reason = f"arm C rows missing for {len(missing_c)} ids - {arm} has no G6 reference; not started"
            break
    if not todo:
        st["status"] = "complete"
        continue
    press = make_arm(arm)
    rate = PRIOR_S[arm]
    measured = [r["wall_s"] for r in rows if r["arm"] == arm]
    if measured:
        rate = float(np.mean(measured))
    proj = args.spent_eur + job_cost_eur(extra_s=rate * len(todo))
    log(f"stage {arm}: {len(todo)} item-arms to run, rate {rate:.2f} s/item, projected cumulative list price EUR {proj:.2f}")
    st.update(status="running", projected_eur_at_start=proj, rate_used_s=rate)
    if proj > args.cost_cap_eur and arm not in ("A",) + ADDENDUM_ARMS:  # rev 1.1: addendum arms use the next-item guard
        # F's prior is unmeasured: run 3 items first, then re-project
        if arm == "F" and not measured:
            log("stage F: prior rate over cap; measuring on 3 items before deciding")
        else:
            stop_reason = f"cost guard: stage {arm} projected EUR {proj:.2f} > cap {args.cost_cap_eur}"
            st["status"] = "skipped (cost guard)"
            break
    for n_stage, iid in enumerate(todo):
        if time.time() - JOB_T0 > args.time_budget_s:
            stop_reason = f"time budget {args.time_budget_s:.0f} s reached"
            break
        if args.stop_after_items is not None and new_item_arms >= args.stop_after_items:
            stop_reason = f"--stop-after-items {args.stop_after_items}"
            break
        meas = [r["wall_s"] for r in rows if r["arm"] == arm]
        if arm in ADDENDUM_ARMS:  # rev 1.1: spend up to the cap; a partial addendum arm is declared, never silently cut
            proj = args.spent_eur + job_cost_eur(extra_s=(float(np.mean(meas)) if len(meas) >= 3 else rate))
            if proj > args.cost_cap_eur:
                stop_reason = f"cost guard: next {arm} item would bring the projection to EUR {proj:.2f} > cap {args.cost_cap_eur}"
                st["status"] = "stopped (cost guard)"
                break
        elif len(meas) >= 3:
            proj = args.spent_eur + job_cost_eur(extra_s=float(np.mean(meas)) * (len(todo) - n_stage))
            if proj > args.cost_cap_eur:
                stop_reason = f"cost guard: stage {arm} projected EUR {proj:.2f} > cap {args.cost_cap_eur} after {len(meas)} items"
                st["status"] = "stopped (cost guard)"
                break
        row = df.loc[iid]
        if STATE["first_item_started"] is None:
            STATE["first_item_started"] = time.time() - T0
            gates["timestamps"]["first_item_started_s"] = STATE["first_item_started"]
            log("first-item-started")
        mnt = int(args.max_new_tokens or row["max_new_tokens"])
        ITEM.clear()
        ITEM.update(new_item_counters())
        reset_modules(pipe.model)
        if arm in ("PL", "Cperm"):
            press._comp.item_key = int(iid)  # rev 1.1: placebo / permutation draw seeded per item and layer
        cache = DynamicCache()
        t1 = time.time()
        try:
            out = pipe(row["context"], question=row["question"], answer_prefix=row["answer_prefix"], press=press,
                       max_new_tokens=mnt, max_context_length=args.max_context_length, cache=cache)
        except Exception:
            log(f"item {iid} arm {arm} EXCEPTION\n{traceback.format_exc()}")
            flush()
            raise
        wall = time.time() - t1
        pred = out["answer"]
        rec = kept_record(pipe.model, cache)
        c = dict(ITEM)
        _bv = c.pop("_b_vals", None)  # rev 1.1: bias magnitude summary (quantiles over all non-zero entries of the item)
        if _bv:
            _allb = torch.cat(_bv)
            _q = torch.quantile(_allb, torch.tensor([0.5, 0.9, 0.99]))
            c["b_q50"], c["b_q90"], c["b_q99"] = [float(x) for x in _q]
            c["b_max"] = float(_allb.max())
            c["b_window_share"] = float(c.get("b_in_window", 0)) / max(1, int(_allb.numel()))
        report = getattr(press, "last_report", []) if isinstance(press, MergingAnyPress) else []
        c["passthrough_layers"] = sum(1 for r in report if r.get("path") == "passthrough")
        c["skipped_layers"] = sum(1 for r in report if r.get("skipped"))
        c["evicting_layers_seen"] = sum(1 for r in report if int(r.get("n_evicted", 0) or 0) > 0)
        FOLD_ARMS = ("C", "PL", "Cperm", "CV", "C0")  # rev 1.1: every arm that runs MassFold's routing on the inner score
        if arm in FOLD_ARMS:
            c["scores_missing_layers"] = sum(1 for r in report if int(r.get("n_evicted", 0) or 0) > 0 and
                                             "no full-length inner score" in str(r.get("skipped", "")))
        if arm == "PL":
            c["pl_norm_relerr_max"] = max([float(r.get("pl_norm_relerr_max", 0.0) or 0.0) for r in report] + [0.0])
            _cos = [float(r["pl_cos_abs_mean"]) for r in report if r.get("pl_cos_abs_mean") is not None]
            c["pl_cos_abs_mean"] = float(np.mean(_cos)) if _cos else None
        bias_left = sum(1 for m in attn_modules(pipe.model) if getattr(m, "anypress_logit_bias", None) is not None)
        # ---- G4 / G5 / SPEC (before the row is appended)
        tag = f"item {iid} arm {arm}"
        if arm in ("C", "F") + FOLD_ARMS:
            for k in ("w_sink", "w_masked", "w_appended", "b_sink", "b_masked", "b_appended", "keys_changed"):
                if c[k] != 0:
                    gate_fail("G4", f"{tag}: {k}={c[k]}")
            if c["passthrough_layers"]:
                gate_fail("G5", f"{tag}: {c['passthrough_layers']} layers took the pass-through path")
            if c["noevict_heads_touched"]:
                gate_fail("SPEC", f"{tag}: {c['noevict_heads_touched']} heads without eviction were modified")
            gates["G4"]["item_arms_checked"] += 1
            gates["SPEC"]["noevict_heads"] += c["noevict_heads"]
        if arm in FOLD_ARMS:
            if c["scores_missing_layers"] or (rec["evicting_layers"] > 0 and c["evicting_layers_seen"] != rec["evicting_layers"]):
                gate_fail("G5", f"{tag}: scores missing on {c['scores_missing_layers']} layers / evicting layers seen "
                                f"{c['evicting_layers_seen']} vs {rec['evicting_layers']}")
        if arm in ("C", "PL", "Cperm", "CV"):
            if c["merged"] > 0 and (c["bias_nonzero"] == 0 or c["bias_mask_calls"] == 0):
                gate_fail("G5", f"{tag}: merged={c['merged']} but bias_nonzero={c['bias_nonzero']} calls={c['bias_mask_calls']}")
        if arm == "C0":  # zero-content kernel-path control: no writes, no non-zero bias, the zero bias path taken
            if c["writes"] or c["bias_nonzero"]:
                gate_fail("G4", f"{tag}: C0 wrote {c['writes']} values / {c['bias_nonzero']} non-zero bias entries")
            if c["merged"] > 0 and (int(c.get("c0_layers_forced", 0)) == 0 or c["bias_mask_calls"] == 0):
                gate_fail("G5", f"{tag}: C0 merged={c['merged']} but zero-bias layers {c.get('c0_layers_forced', 0)} "
                                f"calls={c['bias_mask_calls']}")
        if arm == "F" and c["bias_nonzero"] > 0 and c["bias_mask_calls"] == 0:
            gate_fail("G5", f"{tag}: bias non-zero {c['bias_nonzero']} but never applied")
        if arm in ("A", "Z", "AV"):
            if c["bias_mask_calls"] or c["writes"] or c["bias_nonzero"] or bias_left:
                gate_fail("G5", f"{tag}: bias/writes in a no-compensation arm {c} bias_left={bias_left}")
        gates["G5"]["item_arms_checked"] += 1
        # ---- rev 1.1: AV compares the re-run bare press with the saved arm-A row (no per-item failure; gate after the stage)
        av_rec = None
        if arm == "AV":
            ref = A_REF[iid]
            av_rec = {"item_id": int(iid), "mask_equal": rec["mask_sha1"] == ref["mask_sha1"], "kept_equal": rec["kept"] == ref["kept"],
                      "effcr_equal": abs(rec["effective_cr"] - ref["effective_cr"]) <= 1e-9,
                      "pred_equal": pred == ref["predicted_answer"],
                      "score_equal": ruler_score(pred, row["answer"], row["task"]) == float(ref["score"])}
            gates["AV"]["items"].append(av_rec)
            gates["AV"]["items_checked"] += 1
            for k in ("mask_equal", "kept_equal", "effcr_equal", "pred_equal", "score_equal"):
                gates["AV"][k] += int(av_rec[k])
        # ---- rev 1.1: G6 control integrity against the saved arm-C row of the same item
        cv_rec = None
        if arm in ("PL", "Cperm", "C0"):
            refc = C_REF[iid]
            g6 = gates["G6"].setdefault("per_arm", {}).setdefault(arm, {"items_checked": 0, "merged_equal": 0,
                                                                       "bias_nonzero_equal": 0, "bias_nonzero_max_rel_diff": 0.0})
            if int(c["merged"]) != int(refc["c_merged"]):
                gate_fail("G6", f"{tag}: merged {c['merged']} vs C {refc['c_merged']}")
            bnz_w = int(c.get("c0_bias_nonzero_c", 0)) if arm == "C0" else int(c["bias_nonzero"])  # C's would-be bias for C0
            rel = abs(bnz_w - int(refc["c_bias_nonzero"])) / max(1, int(refc["c_bias_nonzero"]))
            gates["G6"]["bias_nonzero_max_rel_diff"] = max(gates["G6"]["bias_nonzero_max_rel_diff"], rel)
            g6["bias_nonzero_max_rel_diff"] = max(g6["bias_nonzero_max_rel_diff"], rel)
            if rel > args.g6_bias_tol:
                gate_fail("G6", f"{tag}: C-routing bias entries {bnz_w} vs saved C {refc['c_bias_nonzero']} (rel {rel:.2e})")
            if arm == "PL" and c["pl_norm_relerr_max"] > 1e-3:
                gate_fail("G6", f"{tag}: placebo rows not norm-matched (max rel err {c['pl_norm_relerr_max']:.2e})")
            if arm == "Cperm" and (int(c.get("perm_moved_kept", 0)) != 0 or float(c.get("perm_sum_relerr", 0.0)) > 1e-3):
                gate_fail("G6", f"{tag}: permutation moved {c.get('perm_moved_kept')} kept rows / evicted-sum rel err "
                                f"{c.get('perm_sum_relerr')}")
            if arm in ("PL", "Cperm") and c["merged"] > 0 and c["writes"] == 0:
                gate_fail("G6", f"{tag}: merged {c['merged']} but no value writes")
            if arm == "PL":
                gates["G6"]["norm_relerr_max"] = max(gates["G6"]["norm_relerr_max"], c["pl_norm_relerr_max"])
            gates["G6"]["merged_equal"] += 1
            g6["merged_equal"] += 1
            gates["G6"]["bias_nonzero_equal"] += int(bnz_w == int(refc["c_bias_nonzero"]))
            g6["bias_nonzero_equal"] += int(bnz_w == int(refc["c_bias_nonzero"]))
            gates["G6"]["items_checked"] += 1
            g6["items_checked"] += 1
        if arm == "CV":  # read-out: reproducibility of the C path against the saved C row (no stopping rule)
            refc = C_REF[iid]
            cv_rec = {"item_id": int(iid), "pred_equal": pred == refc["predicted_answer"],
                      "score_equal": ruler_score(pred, row["answer"], row["task"]) == float(refc["score"]),
                      "merged_equal": int(c["merged"]) == int(refc["c_merged"]),
                      "bias_nonzero_equal": int(c["bias_nonzero"]) == int(refc["c_bias_nonzero"]),
                      "writes_equal": int(c["writes"]) == int(refc["c_writes"])}
            cvg = gates.setdefault("CV", {"items_checked": 0, "pred_equal": 0, "score_equal": 0, "merged_equal": 0,
                                          "bias_nonzero_equal": 0, "writes_equal": 0, "items": []})
            cvg["items"].append(cv_rec)
            cvg["items_checked"] += 1
            for k in ("pred_equal", "score_equal", "merged_equal", "bias_nonzero_equal", "writes_equal"):
                cvg[k] += int(cv_rec[k])
        # ---- G1 / G2 against arm A of the same item
        if arm not in ("A", "AV"):
            ref = A_REF[iid]
            if rec["kept"] != ref["kept"] or rec["mask_sha1"] != ref["mask_sha1"]:
                nd = sum(int(x != y) for la, lb_ in zip(rec["kept"], ref["kept"]) for x, y in zip(la, lb_))
                gate_fail("G1", f"{tag}: {nd} (layer, head) kept counts differ; mask_sha1 {rec['mask_sha1']} vs {ref['mask_sha1']}")
            d = abs(rec["effective_cr"] - ref["effective_cr"])
            gates["G2"]["max_abs_diff"] = max(gates["G2"]["max_abs_diff"], d)
            if d > 1e-6:
                gate_fail("G2", f"{tag}: effective_cr {rec['effective_cr']} vs A {ref['effective_cr']}")
            gates["G1"]["items_checked"] += 1
            gates["G2"]["items_checked"] += 1
            if arm == "Z":
                if pred != ref["predicted_answer"]:
                    gate_fail("SPEC", f"{tag}: W(P, none) prediction differs from A: {pred!r} vs {ref['predicted_answer']!r}")
                gates["SPEC"]["z_items_checked"] += 1
        r = {"op": args.op, "threshold": args.threshold, "arm": arm, "item_id": int(iid), "task": row["task"],
             "predicted_answer": pred, "score": ruler_score(pred, row["answer"], row["task"]), "answer": str(row["answer"]),
             "max_new_tokens": mnt, "wall_s": wall,
             "lb_predicted_answer": (str(lb["predicted_answer"].iloc[iid]) if lb is not None else None),
             "lb_score": (ruler_score(lb["predicted_answer"].iloc[iid], row["answer"], row["task"]) if lb is not None else None),
             "lb_cr": (float(lb["compression_ratio"].iloc[iid]) if lb is not None else None),
             **{k: rec[k] for k in ("ctx_len", "n_layers", "n_kv_heads", "kept_total", "masked_total", "evicting_layers",
                                    "mask_sha1", "kept_sha1", "effective_cr")},
             **{f"c_{k}": v for k, v in c.items()}}
        if arm == "A":
            r["kept"] = rec["kept"]
            A_REF[iid] = r
        if av_rec is not None:
            r["av"] = av_rec
        if cv_rec is not None:
            r["cv"] = cv_rec
        rows.append(r)
        done.add((arm, iid))
        ROWS_FH.write(json.dumps(r, default=str) + "\n")
        ROWS_FH.flush()
        new_item_arms += 1
        flush()
        log(f"{arm} {n_stage + 1}/{len(todo)} id={iid} task={row['task']} ctx={rec['ctx_len']} effCR={rec['effective_cr']:.4f} "
            f"score={r['score']:.0f} lb={r['lb_score']} merged={c['merged']} bias_nz={c['bias_nonzero']} "
            f"calls={c['bias_mask_calls']} wall={wall:.2f}s cost~EUR{args.spent_eur + job_cost_eur():.2f}")
    if stop_reason:
        st.setdefault("status", "incomplete")
        if st["status"] == "running":
            st["status"] = "incomplete"
        break
    st["status"] = "complete"
    st["wall_mean_s"] = float(np.mean([r["wall_s"] for r in rows if r["arm"] == arm]))
    if arm == "A":
        ok = g3_check()
        flush()
        if not ok and args.g3_mode == "assert":
            gate_fail("G3", "bare arm does not reproduce the published predictions; no treatment arm is run")
    if arm == "AV":  # rev 1.1: AV gate (pre-registered): every kept set identical AND >= av_min_pred_match identical strings
        g = gates["AV"]
        n = int(g["items_checked"])
        ok = (n == len(stage_ids) and g["mask_equal"] == n and g["kept_equal"] == n and g["effcr_equal"] == n
              and g["pred_equal"] >= args.av_min_pred_match)
        g["status"] = "pass" if ok else "FAIL"
        g["rule"] = (f"mask/kept/effective-CR identical on {len(stage_ids)}/{len(stage_ids)} ids AND identical prediction "
                     f"strings on >= {args.av_min_pred_match}/{len(stage_ids)}")
        log(f"AV gate: n={n} mask_equal={g['mask_equal']} kept_equal={g['kept_equal']} effcr_equal={g['effcr_equal']} "
            f"pred_equal={g['pred_equal']} score_equal={g['score_equal']} -> {g['status']}")
        flush()
        if not ok:
            gate_fail("AV", "re-run bare press does not reproduce the saved arm-A rows; PL is not paired with them")

gates["stop_reason"] = stop_reason
gates["complete"] = stop_reason is None
dfp = pd.DataFrame(rows)
if len(dfp):
    log("per-arm pooled means: " + json.dumps(dfp.groupby("arm")["score"].mean().round(3).to_dict()))
    gates["wall_per_item_arm_s"] = {a: float(v) for a, v in dfp.groupby("arm")["wall_s"].mean().items()}
gates["timestamps"]["end_s"] = time.time() - T0
STATE["stop"] = True
flush()
log(f"done: stop_reason={stop_reason} rows={len(rows)} job list-price EUR {job_cost_eur():.3f}")
