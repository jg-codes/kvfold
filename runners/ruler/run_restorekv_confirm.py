#!/usr/bin/env python
"""RestoreKV_plus mass-fold CONFIRMATION of C113 (wiki claim C115, filed first; pre-registration hypothesis_C115.json, rev 1.1).

Revision 1.1 (pre-dispatch review P27, before any C115 outcome): (1) the void rule is scoped by stage: a gate failure while arm
C0 or B runs voids only the attribution read-outs (gates.json void.attribution; a resumed job skips C0/B), any other gate
failure voids the run (void.run; a resumed job is refused); (2) --av-tag sK prepends the AV probe, bare P on the 13 Z ids, at
the start of every sandbox after the first: kept set gated against arm A (tolerance 0, cross-sandbox drift voids the run),
prediction string, score and first-step logits hash recorded as re-run noise; (3) every row logs the first-step logits hash
(sha1 of the float32 logits of the question prefill, the call with logits_to_keep=1), its top-1 token and log-prob, and the
first-token gold log-prob (max over golds and over the variants g and " " + g); (4) Z runs right after the early G3 (stage
order A1 = first --g3-early-n ids, Z, A2 = the rest, C, C0, B), so the Z = A specificity check stays inside one container;
(5) the subset order comes from the revised item list (round-robin by the subset hash).

Revision 1.4 (full-set leaderboard run, wiki claim C127, pre-registration hypothesis_C127.json; arm selection and item list
only - no press, fold, gate tolerance, scorer or decoding change): (1) --no-bare-arm runs arm C without arm A in the run
(requires --g3-mode report and an --arms list without A; G3 is not run). G1/G2 and the inner-score hash check run on every item
that has a stored arm-A row (seeded through --resume with the C115 A rows of the AV and overlap ids); items without one are
counted in gates G1.items_without_reference, and the stage list may open with C. (2) The AV probe ids are read from the item
file's av_ids field when present (else ids[:--z-items], as before); their sha256 is written to gates.json. With --no-bare-arm
unset and no av_ids field the behaviour equals rev 1.1 (sha256 25cf013d).

Derived from run_restorekv_smoke.py (C113, sha256 b2b7292a...): same scorer, instrumentation, gates G1/G2/G4/G5, checkpointing,
first-item watchdog, nvidia-smi sampling, time budget and list-price cost guard. Changes for the confirmation:
  * fresh sealed item list (--item-ids: item_ids + subset_ids + z_ids; the full-list sha256 is asserted before any truncation)
  * arms A, Z, C, C0, B (no PL); C0 and B run on the hash-fixed subset only (30/task), Z on the first --z-items ids (1/task)
  * G3 asserted twice: early after --g3-early-n A items (round-robin order, 50/task) and final after stage A; tolerance
    --g3-floor-pp 2.80 (conventions per-task floor at 200 items/task, sd 0.202)
  * ATTN gate: model.config._attn_implementation must equal --expect-attn (sdpa); recorded in gates.json and on every row (A20)
  * dataset parquet and leaderboard predictions hashes asserted
  * watchdog anchored at JOB_T0 (job-script start), not at this script's start
  * dry-run switches (--max-items, --dry-subset, --stop-after-items, hash 'skip') are refused unless --mock-restorekv is set
Inner press P = the leaderboard RestoreKV_plus entry at CR 0.9375 run as the fork's
    RestoreKVPress(compression_ratio=0.9375, layerwise=False, n_sink=4, kvzip_plus_normalization=True)
(the leaderboard's checkpoint_path / budget_matched arguments do not exist in this code; adapter higokri/RestoreKV-Qwen3-8B_plus
from the volume, budget matching built in; G3 tests the mapping, as in C82/C91/C113).
Arms (stages in --arms order; A first, G3 asserted before any treatment stage):
  A   P (bare)
  Z   MergingAnyPress(P, compensation='none', n_sink=4) on the first --z-items ids     specificity: predictions == A exactly
  C   MergingAnyPress(P, compensation=MassFold(score_map='log'), n_sink=4)             as in C113
  C0  MergingAnyPress(P, compensation=ZeroBiasPathFold(score_map='log'), n_sink=4)     subset: C's biased-mask path, bias forced
      to 0 (installed on every layer where C's bias is non-zero), no value change      (kernel-path control, A21)
  B   MergingAnyPress(P, compensation=SelfMassFold(score_map='log'), n_sink=4)         subset: self-fold, C's routing, weights and
      bias with every evicted value replaced by its target's own value (values unchanged up to rounding; mass/bias component)
Protocol: kvpress pipeline, the press compresses the CONTEXT only; question + answer prefix are prefilled afterwards.
Read-out: greedy RULER string_match (leaderboard scorer) at the dataset max_new_tokens.
Gates (asserted in-script before rows.append; a failure writes gates.json and exits 3):
  G1 per-(layer, kv-head) kept counts AND masked index sets (sha1) identical to arm A of the same item
  G2 effective CR = 1 - kept / (layers * kv_heads * L0) (restore rows count as kept) equal to arm A to 1e-6
  G3 |pooled(A) - pooled(published)| <= floor, exact prediction-string match >= 0.5, |mean effective CR(A) - mean published CR|
     <= 0.01 on the same ids; checked early (--g3-early-n A items) and after stage A; a failure ends the job before any treatment
  G4 every arm: restore rows never masked. C/C0/B: zero value writes / bias entries on sinks, masked rows and restore rows; keys
     unchanged; the wrapper sees n_restore appended rows per kv-head
  G5 every arm: compress_post ran once, n_restore == --expect-restore. C/C0/B: score captured on every evicting layer, no
     pass-through layer. C/B: merged > 0 => bias non-zero > 0 and biased-mask calls > 0. C0: zero value writes and zero non-zero
     bias entries; merged > 0 => zero bias installed on >= 1 layer and biased-mask calls > 0. A/Z: no bias, no biased-mask calls,
     no value writes, no bias left on any module
  G6 C0/B vs the C row of the same item: merged equal; C's non-zero bias count reproduced within --g6-bias-tol (relative) and
     C's bias sum within 1e-3 (relative) (C0: the would-be bias before zeroing; B: the installed bias); B: self-fold identity,
     max relative value change <= --b-maxrel-tol
  SPEC heads without an evicted row receive zero writes and zero bias (C/C0/B); Z predictions == A predictions
  ATTN model.config._attn_implementation == --expect-attn
Economy: job-start (JOB_T0) / script-start / model-loaded / first-item timestamps; abort if the first item has not started
--watchdog-s after JOB_T0; nvidia-smi every 60 s; rows appended + gates.json rewritten after every item; --resume; a time
budget (from JOB_T0) and a list-price cost guard stop the job between items.
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
JOB_T0 = float(os.environ.get("JOB_T0", T0))

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
ap.add_argument("--cr", type=float, default=0.9375)
ap.add_argument("--arms", default="A,Z,C,C0,B")
ap.add_argument("--z-items", type=int, default=13, help="arm Z runs on the first N items only (1 per task)")
ap.add_argument("--item-ids", required=True)
ap.add_argument("--max-items", type=int, default=None, help="dry-run only (requires --mock-restorekv)")
ap.add_argument("--dry-subset", type=int, default=None, help="dry-run only: subset = the first N of the truncated ids")
ap.add_argument("--claim", default="C115")
ap.add_argument("--expect-ids-sha256", default="71f480003e8d3c4ecc183d9c75e6310c218006b413e85c60b0669286fbae8f83",
                help="sha256 of json.dumps(item_ids) of the sealed list, asserted before truncation; 'skip' = dry-run only")
ap.add_argument("--expect-subset-sha256", default="bde082c5e5b79f8e51bddc8aed5178338fa494ae0b6ed65a911ce1d95fa18856")
ap.add_argument("--av-tag", default=None, help="sandbox tag (e.g. s2): run the AV probe (bare P on the Z ids) first")
ap.add_argument("--no-bare-arm", action="store_true",
                help="rev 1.4 (C127 full-set run): arm C without arm A in the run; G1/G2 on items with a stored A row only")
ap.add_argument("--dataset-sha256", default="47951890333763a6eede43bc5dbdc9f4f028f312a312f7299db05f7485ad1444")
ap.add_argument("--expect-attn", default="sdpa")
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
ap.add_argument("--dtype", default="auto")
ap.add_argument("--attn", default="sdpa")
ap.add_argument("--max-context-length", type=int, default=None)
ap.add_argument("--max-new-tokens", type=int, default=None)
ap.add_argument("--leaderboard", default="leaderboard/RestoreKV_plus__0.94__predictions.csv")
ap.add_argument("--leaderboard-sha256", default="f9a2a90a8df163b60cee267989c111e28cefcc70e6438f3d52321d390734ef1d",
                help="assert the published predictions file hash; 'skip' = dry-run only")
ap.add_argument("--g3-floor-pp", type=float, default=2.80)
ap.add_argument("--g3-early-n", type=int, default=650, help="early G3 after this many A items (50/task in round-robin order)")
ap.add_argument("--g3-cr-tol", type=float, default=0.01)
ap.add_argument("--g3-mode", choices=["assert", "report"], default="assert")
ap.add_argument("--mock-restorekv", action="store_true", help="dry-run: random PEFT adapter (tests/default_presses.py pattern)")
ap.add_argument("--expect-restore", type=int, default=8)
ap.add_argument("--g6-bias-tol", type=float, default=0.001)
ap.add_argument("--b-maxrel-tol", type=float, default=0.01, help="B self-fold identity: max relative value change")
ap.add_argument("--out", default="out")
ap.add_argument("--resume", default=None, help="rows.jsonl of a previous job (copied into --out and continued)")
ap.add_argument("--time-budget-s", type=float, default=3400.0, help="no new item after JOB_T0 + this")
ap.add_argument("--watchdog-s", type=float, default=300.0)
ap.add_argument("--stop-after-items", type=int, default=None, help="dry-run resume test: stop after N new item-arms")
ap.add_argument("--usd-per-s", type=float, default=0.00082312,
                help="list-price container rate: A100-40GB 0.000583 + 2 cores x 0.00003942 + 24 GiB x 0.00000672 (sandbox tier)")
ap.add_argument("--eur-usd", type=float, default=1.17)
ap.add_argument("--spent-eur", type=float, default=0.0, help="list-price spend of earlier jobs of this task")
ap.add_argument("--cost-cap-eur", type=float, default=float("inf"), help="optional item guard in EUR at list price")
ap.add_argument("--startup-overhead-s", type=float, default=60.0)
args = ap.parse_args()
if not args.mock_restorekv:  # dry-run switches are refused on a real run
    assert args.max_items is None and args.dry_subset is None and args.stop_after_items is None, "dry-run switch on a real run"
    assert "skip" not in (args.expect_ids_sha256, args.leaderboard_sha256, args.dataset_sha256), "hash check skipped on a real run"
if args.dry_subset is not None:
    assert args.max_items is not None, "--dry-subset needs --max-items"
if args.no_bare_arm:  # rev 1.4: C-only mode; G3 needs a bare arm in the run and is not run
    assert "A" not in args.arms.split(",") and args.g3_mode == "report", "--no-bare-arm needs --arms without A and --g3-mode report"

os.makedirs(args.out, exist_ok=True)
LOG = open(os.path.join(args.out, "progress.log"), "a")


def log(*a):
    s = f"[{time.time() - T0:7.1f}s | job +{time.time() - JOB_T0:7.1f}s] " + " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.write(s + "\n")
    LOG.flush()


log(f"script-start T0={T0:.1f} JOB_T0={JOB_T0:.1f} argv={sys.argv[1:]}")
torch.manual_seed(0)
np.random.seed(0)

# ----------------------------------------------------------------------------- watchdog + nvidia-smi sampling
STATE = {"first_item_started": None, "stop": False}


def _watchdog():
    while STATE["first_item_started"] is None and not STATE["stop"]:
        if time.time() - min(T0, JOB_T0) > args.watchdog_s:
            log(f"WATCHDOG: first item not started within {args.watchdog_s:.0f} s of the job start; aborting")
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
import kvpress.presses.restorekv_press as rk_mod  # noqa: E402
from kvpress import RestoreKVPress  # noqa: E402
from kvpress.presses.merging_any_press import MassFold, MergingAnyPress  # noqa: E402
from control_folds import SelfMassFold, ZeroBiasPathFold  # noqa: E402
from transformers import DynamicCache, pipeline  # noqa: E402

SRC_SHA = {n: hashlib.sha256(open(m.__file__, "rb").read()).hexdigest() for n, m in
           [("merging_any_press.py", map_mod), ("restorekv_press.py", rk_mod),
            ("kvzip_press.py", sys.modules["kvpress.presses.kvzip_press"]), ("pipeline.py", sys.modules["kvpress.pipeline"]),
            ("control_folds.py", sys.modules["control_folds"]), ("run_restorekv_confirm.py", sys.modules["__main__"])]}
log(f"kvpress from {kvpress.__file__}; sha256 " + json.dumps({k: v[:16] for k, v in SRC_SHA.items()}))

N_SINK = 4
ITEM = {}  # per item-arm instrumentation counters
REC = {}  # per item-arm compress_post record


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
    appended = ks.src < 0  # RestoreKV restore rows
    c["writes"] += int(changed.sum())
    c["bias_nonzero"] += int(bnz.sum())
    c["bias_sum"] += float(bias.double().sum()) if bias is not None else 0.0
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
    if bias is not None and bool(bnz.any()):  # bias magnitudes (reviewer instrumentation, C111 review P23)
        c.setdefault("_b_vals", []).append(bias[bnz].detach().float().cpu())
    keys_before = ks.cache_keys.detach().clone()
    out = _orig_apply(self, module, layer_idx, ks, res, rec)
    k_after = self._cache.layers[layer_idx].keys
    c["keys_changed"] += int((k_after != keys_before).any(-1).sum())
    if res.info.get("c0_force_bias_path"):  # arm C0 (port of run_kvzap_smoke.py rev 1.1): C's biased-mask path, zero bias
        tgt = ks.target.nonzero()
        if len(tgt):
            b_, h_, r_ = (int(i) for i in tgt[0])  # a target row: kept, never masked, never rewritten
            module.anypress_bias_probe = (b_, h_, r_, ks.cache_keys[b_, h_, r_].detach().clone())
            module.anypress_logit_bias = res.logit_bias.float()
            c["c0_layers_forced"] = c.get("c0_layers_forced", 0) + 1
    if "c0_bias_nonzero_c" in res.info:  # C's would-be bias on this layer (before zeroing)
        c["c0_bias_nonzero_c"] = c.get("c0_bias_nonzero_c", 0) + int(res.info.get("c0_bias_nonzero_c", 0))
        c["c0_bias_sum_c"] = c.get("c0_bias_sum_c", 0.0) + float(res.info.get("c0_bias_sum_c", 0.0))
    if "b_self_maxrel" in res.info:  # arm B: self-fold identity
        c["b_self_maxrel"] = max(c.get("b_self_maxrel", 0.0), float(res.info["b_self_maxrel"]))
        c["b_self_rows_changed"] = c.get("b_self_rows_changed", 0) + int(res.info.get("b_self_rows_changed", 0))
    return out


map_mod.MergingAnyPress._apply = apply_recording

_orig_biased_mask = map_mod._biased_mask


def biased_mask_counting(*a, **kw):
    ITEM["bias_mask_calls"] = ITEM.get("bias_mask_calls", 0) + 1
    return _orig_biased_mask(*a, **kw)


map_mod._biased_mask = biased_mask_counting

_orig_rk_post = rk_mod.RestoreKVPress.compress_post


def rk_post_recording(self, model):
    REC["compress_post_calls"] = REC.get("compress_post_calls", 0) + 1
    REC["ctx_len"] = int(self.context_length)
    REC["requested_cr"] = float(self.compression_ratio)
    _orig_rk_post(self, model)
    REC["n_restore"] = int(self.num_restore_tokens)
    sv = getattr(self, "score_val", None)
    REC["score_sha1"] = (hashlib.sha1(sv.detach().float().cpu().numpy().tobytes()).hexdigest()[:16]
                         if isinstance(sv, torch.Tensor) else None)
    REC["score_shape"] = list(sv.shape) if isinstance(sv, torch.Tensor) else None


rk_mod.RestoreKVPress.compress_post = rk_post_recording


class MockRestoreKVPress(RestoreKVPress):
    """Dry-run only (tests/default_presses.py TestRestoreKVPress pattern): a PEFT LoRA adapter with peft's default init
    instead of the Hub adapter; restore embeddings seeded N(0, 0.02^2) instead of zeros so that the restore rows are non-zero
    at every layer (the G4 write protection is then exercised on non-trivial rows)."""

    def post_init_from_model(self, model):
        if self.restore_embeddings is not None:
            return
        from peft import LoraConfig

        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        if self.adapter_name not in getattr(model, "peft_config", {}):
            config = LoraConfig(r=8, lora_alpha=16, target_modules=target_modules, task_type="CAUSAL_LM")
            model.add_adapter(config, adapter_name=self.adapter_name)
        model.disable_adapters()
        g = torch.Generator(device="cpu").manual_seed(4321)
        emb = torch.randn(args.expect_restore, model.config.hidden_size, generator=g) * 0.02
        self.restore_embeddings = emb.to(model.device, dtype=model.dtype)


def make_inner():
    cls = MockRestoreKVPress if args.mock_restorekv else RestoreKVPress
    # leaderboard press_init_command minus checkpoint_path / budget_matched (not arguments of this RestoreKVPress; see docstring)
    return cls(compression_ratio=args.cr, layerwise=False, n_sink=4, kvzip_plus_normalization=True)


def make_arm(a):
    if a in ("A", "AV"):  # AV = the bare press re-run on the Z ids (rev 1.1)
        return make_inner()
    if a == "Z":
        return MergingAnyPress(make_inner(), compensation="none", n_sink=N_SINK)
    if a == "C":
        return MergingAnyPress(make_inner(), compensation=MassFold(score_map="log"), n_sink=N_SINK)
    if a == "C0":
        return MergingAnyPress(make_inner(), compensation=ZeroBiasPathFold(score_map="log"), n_sink=N_SINK)
    if a == "B":
        return MergingAnyPress(make_inner(), compensation=SelfMassFold(score_map="log"), n_sink=N_SINK)
    raise SystemExit(f"unknown arm {a}")


FOLD_ARMS = ("C", "C0", "B")  # every arm that runs MassFold's routing on the inner score


def attn_modules(model):
    lm = model.model.language_model if hasattr(model.model, "language_model") else model.model
    return [layer.self_attn for layer in lm.layers]


def reset_modules(model):
    for m in attn_modules(model):
        m.masked_key_indices = None
        for n in ("anypress_logit_bias", "anypress_bias_probe", "merge_logit_bias"):
            if getattr(m, n, None) is not None:
                setattr(m, n, None)


def kept_record(model, cache, ctx_len):
    kept, mask_h = [], hashlib.sha1()
    evicting = 0
    n_heads = None
    cache_len = None
    restore_masked = 0
    for m in attn_modules(model):
        k = cache.layers[int(m.layer_idx)].keys
        n_heads = int(k.shape[1])
        cl = int(k.shape[2])
        cache_len = cl if cache_len is None else cache_len
        assert cl == cache_len, "cache length differs across layers"
        mk = getattr(m, "masked_key_indices", None)
        if mk is None or len(mk[0]) == 0:
            nm = torch.zeros(n_heads, dtype=torch.long)
            mask_h.update(b"none")
        else:
            evicting += 1
            h_idx, t_idx = mk[1].cpu().long(), mk[2].cpu().long()
            restore_masked += int((t_idx >= ctx_len).sum())
            nm = torch.bincount(h_idx, minlength=n_heads)
            flat = torch.unique(h_idx * (cache_len + 1) + t_idx)
            assert flat.numel() == h_idx.numel(), "duplicate masked indices"
            mask_h.update(flat.numpy().tobytes())
        kept.append((cache_len - nm).tolist())
    kept_total = int(sum(sum(x) for x in kept))
    n_layers = len(kept)
    return {"ctx_len": int(ctx_len), "cache_len": cache_len, "n_restore_cache": cache_len - int(ctx_len), "n_layers": n_layers,
            "n_kv_heads": n_heads, "kept": kept, "kept_total": kept_total, "masked_total": n_layers * n_heads * cache_len - kept_total,
            "evicting_layers": evicting, "restore_masked": restore_masked, "mask_sha1": mask_h.hexdigest()[:16],
            "kept_sha1": hashlib.sha1(json.dumps(kept).encode()).hexdigest()[:16],
            "effective_cr": 1.0 - kept_total / float(n_layers * n_heads * ctx_len)}


# ----------------------------------------------------------------------------- data
from huggingface_hub import hf_hub_download  # noqa: E402

pq = hf_hub_download("simonjegou/ruler", "4096/test-00000-of-00001.parquet", repo_type="dataset")
DS_SHA = hashlib.sha256(open(pq, "rb").read()).hexdigest()
if args.dataset_sha256 != "skip":
    assert DS_SHA == args.dataset_sha256, f"dataset parquet sha256 {DS_SHA} != pre-registered {args.dataset_sha256}"
df = pd.read_parquet(pq)
df["item_id"] = np.arange(len(df))
ids_src = json.load(open(args.item_ids))
ids = [int(i) for i in ids_src["item_ids"]]
sub_all = [int(i) for i in ids_src.get("subset_ids", [])]
AV_IDS = [int(i) for i in ids_src.get("av_ids", [])]  # rev 1.4: AV probe ids from the item file (else ids[:--z-items])
assert len(set(AV_IDS)) == len(AV_IDS) and all(0 <= i < len(df) for i in AV_IDS), "av_ids must be distinct dataset ids"
IDS_SHA_FULL = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
SUB_SHA_FULL = hashlib.sha256(json.dumps(sub_all).encode()).hexdigest()
if args.expect_ids_sha256 != "skip":
    assert IDS_SHA_FULL == args.expect_ids_sha256, f"item list sha256 {IDS_SHA_FULL} != sealed {args.expect_ids_sha256}"
    assert SUB_SHA_FULL == args.expect_subset_sha256, f"subset sha256 {SUB_SHA_FULL} != sealed {args.expect_subset_sha256}"
assert len(set(ids)) == len(ids) and max(ids) < len(df) and set(sub_all) <= set(ids)
if args.max_items:
    ids = ids[: args.max_items]
subset = [i for i in sub_all if i in set(ids)]
if args.dry_subset is not None:
    subset = ids[: args.dry_subset]
IDS_SHA = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
SUBSET_SET = set(subset)
sample = df.loc[ids].reset_index(drop=True)
log(f"dataset rows={len(df)} sha256={DS_SHA[:16]} items={len(sample)} ids_sha256={IDS_SHA[:16]} (full list {IDS_SHA_FULL[:16]}) "
    f"subset={len(subset)} tasks={sample.task.value_counts().to_dict()}")

lb = None
LB_SHA = None
if os.path.exists(args.leaderboard):
    LB_SHA = hashlib.sha256(open(args.leaderboard, "rb").read()).hexdigest()
    if args.leaderboard_sha256 != "skip":
        assert LB_SHA == args.leaderboard_sha256, f"leaderboard sha256 {LB_SHA} != pre-registered {args.leaderboard_sha256}"
    lb = pd.read_csv(args.leaderboard)
    assert len(lb) == len(df), f"leaderboard rows {len(lb)} != dataset rows {len(df)}"
    mis = int((lb["question"].astype(str).values[ids] != df["question"].astype(str).values[ids]).sum())
    mis += int((lb["task"].astype(str).values[ids] != df["task"].astype(str).values[ids]).sum())
    log(f"leaderboard {args.leaderboard} sha256={LB_SHA[:16]} alignment mismatches on ids={mis}")
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
    log(f"resume: {len(rows)} rows loaded")
elif os.path.exists(ROWS_PATH):
    os.remove(ROWS_PATH)
done = {(r["arm"], int(r["item_id"])) for r in rows}
assert len(done) == len(rows), "duplicate (arm, item) rows in the resume file"
A_REF = {int(r["item_id"]): r for r in rows if r["arm"] == "A"}
C_REF = {int(r["item_id"]): r for r in rows if r["arm"] == "C"}
ROWS_FH = open(ROWS_PATH, "a")

gates = {"claim": args.claim, "press": "RestoreKV_plus", "cr": args.cr, "ids_sha256": IDS_SHA, "ids_sha256_full_list": IDS_SHA_FULL,
         "subset_sha256_full_list": SUB_SHA_FULL, "n_items": len(ids), "n_subset": len(subset), "dataset_sha256": DS_SHA,
         "src_sha256": SRC_SHA, "leaderboard_sha256": LB_SHA, "installed_dist_kvpress": _md.version("kvpress"),
         "kvpress_file": kvpress.__file__, "argv": sys.argv[1:], "job_t0": JOB_T0,
         "G1": {"status": "pass", "items_checked": 0}, "G2": {"status": "pass", "items_checked": 0, "max_abs_diff": 0.0},
         "G3": {"status": "not run", "mode": args.g3_mode}, "G3_early": {"status": "not run", "n_at": args.g3_early_n},
         "ATTN": {"status": "not run", "expected": args.expect_attn}, "void": {"run": False, "attribution": False},
         "AV": {"status": "not run", "probes": {}},
         "G4": {"status": "pass", "item_arms_checked": 0, "restore_masked_total": 0},
         "G5": {"status": "pass", "item_arms_checked": 0},
         "G6": {"status": "pass", "items_checked": 0, "merged_equal": 0, "bias_nonzero_equal": 0, "bias_nonzero_max_rel_diff": 0.0,
                "bias_sum_max_rel_diff": 0.0, "b_self_maxrel_max": 0.0, "tol": args.g6_bias_tol, "per_arm": {}},
         "SPEC": {"status": "pass", "z_items_checked": 0, "noevict_heads": 0, "noevict_heads_touched": 0},
         "score_sha1_equal_to_A": {"checked": 0, "equal": 0}, "stages": {}, "attn_history": [], "timestamps": {"job_t0": JOB_T0,
         "script_start_s_after_job_t0": T0 - JOB_T0}, "cost": {}}
if os.path.exists(GATES_PATH) and args.resume:
    old = json.load(open(GATES_PATH))
    for k in ("G1", "G2", "G3", "G3_early", "G4", "G5", "G6", "SPEC", "ATTN", "score_sha1_equal_to_A", "stages",
              "attn_history", "void", "AV"):
        gates[k] = old.get(k, gates[k])
    gates["previous_jobs"] = old.get("previous_jobs", []) + [{"timestamps": old.get("timestamps", {}), "cost": old.get("cost", {}),
                                                              "stop_reason": old.get("stop_reason")}]


def job_cost_eur(extra_s=0.0):
    return (time.time() - JOB_T0 + args.startup_overhead_s + extra_s) * args.usd_per_s / args.eur_usd


def flush():
    gates["wall_s"] = time.time() - T0
    gates["cost"] = {"job_list_price_eur": job_cost_eur(), "spent_before_eur": args.spent_eur,
                     "usd_per_s": args.usd_per_s, "eur_usd": args.eur_usd, "label": "list-price estimate"}
    json.dump(gates, open(GATES_PATH + ".tmp", "w"), indent=1, default=str)
    os.replace(GATES_PATH + ".tmp", GATES_PATH)


CUR = {"stage": None, "arm": None, "kind": None}  # the stage running now (scopes a gate failure)
ATTRIB_KINDS = ("C0", "B")


def gate_fail(name, msg):
    scope = "attribution" if CUR.get("kind") in ATTRIB_KINDS else "run"
    gates[name]["status"] = "FAIL"
    gates[name].setdefault("failures", []).append({"msg": msg, "stage": CUR.get("stage"), "arm": CUR.get("arm"), "scope": scope})
    gates["void"][scope] = True
    log(f"{name} FAIL ({scope}-scoped, stage {CUR.get('stage')}): {msg}")
    flush()
    STATE["stop"] = True
    sys.exit(3)


gates["void"] = gates.get("void") or {"run": False, "attribution": False}
_failed = [k for k in ("G1", "G2", "G3", "G3_early", "G4", "G5", "G6", "SPEC", "ATTN", "AV") if gates.get(k, {}).get("status") == "FAIL"]
if _failed and not gates["void"]["attribution"]:
    gates["void"]["run"] = True  # a failure without a recorded scope counts as run-scoped
if gates["void"]["run"]:  # a run voided by a failed gate is never continued
    gates["stop_reason"] = f"resume refused: run-scoped gate failure in an earlier job ({_failed})"
    STATE["stop"] = True
    flush()
    log(gates["stop_reason"])
    sys.exit(3)
gates["no_bare_arm"] = bool(args.no_bare_arm)  # rev 1.4
gates["av_ids_sha256"] = hashlib.sha256(json.dumps(AV_IDS).encode()).hexdigest() if AV_IDS else None
ATTRIB_VOID = bool(gates["void"]["attribution"])
if ATTRIB_VOID:
    log("attribution void (C0/B gate failure in an earlier job): C0 and B are skipped")
arms_order = args.arms.split(",")


def stage_ids_of(a):
    if a == "Z":
        return ids[: args.z_items]
    if a in ("C0", "B"):
        return subset
    return ids


# rev 1.1 stage list: [AV probe] -> A1 (first --g3-early-n ids; early G3) -> Z -> A2 (the rest; final G3) -> C -> C0 -> B
N_EARLY = min(args.g3_early_n, len(ids))
STAGES = []  # (stage name, row arm label, kind, stage ids)
if args.av_tag:
    STAGES.append((f"AV_{args.av_tag}", f"AV_{args.av_tag}", "AV", AV_IDS or ids[: args.z_items]))
for _a in arms_order:
    if _a == "A":
        STAGES.append(("A1", "A", "A", ids[:N_EARLY]))
        if "Z" in arms_order:
            STAGES.append(("Z", "Z", "Z", stage_ids_of("Z")))
        STAGES.append(("A2", "A", "A", ids[N_EARLY:]))
    elif _a == "Z" and "A" in arms_order:
        continue
    else:
        STAGES.append((_a, _a, _a, stage_ids_of(_a)))
STAGES = [st_ for st_ in STAGES if not (ATTRIB_VOID and st_[2] in ATTRIB_KINDS)]
gates["stage_plan"] = [(n_, a_, k_, len(i_)) for n_, a_, k_, i_ in STAGES]
_pending = sum(len([i for i in sids if (a, i) not in done]) for _n, a, _k, sids in STAGES)
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
gates["timestamps"]["model_loaded_job_s"] = time.time() - JOB_T0
ATTN = str(getattr(pipe.model.config, "_attn_implementation", None))
log(f"model-loaded {args.model} dtype={pipe.model.dtype} attn={ATTN}")
gates["ATTN"] = {"status": "pass" if (ATTN == args.expect_attn and args.attn == args.expect_attn) else "FAIL",
                 "expected": args.expect_attn, "requested": args.attn, "model_config": ATTN,
                 "torch": torch.__version__, "cuda": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)}
gates["attn_history"].append({"job_t0": JOB_T0, "model_config": ATTN, "requested": args.attn})
flush()
if gates["ATTN"]["status"] != "pass":
    gate_fail("ATTN", f"attn_implementation {ATTN!r} (requested {args.attn!r}) != expected {args.expect_attn!r}")

# rev 1.1: first-step logits of every item-arm = the question prefill, the only CausalLM call with logits_to_keep=1 (the
# context prefill and the presses' own passes call the base model, decode calls pass no logits_to_keep)
FIRST = {}


def _first_logits_hook(module, args_, kwargs, output):
    if kwargs.get("logits_to_keep") == 1 and "logits" not in FIRST:
        FIRST["logits"] = output.logits[0, -1].detach().float().cpu()
    return None


pipe.model.register_forward_hook(_first_logits_hook, with_kwargs=True)


def first_step_record(answer):
    lg = FIRST.get("logits")
    if lg is None:
        return {"l1_sha": None, "l1_top1": None, "l1_top1_lp": None, "gold_lp_first": None, "gold_lp_first_tok": None}
    lp = torch.log_softmax(lg, dim=-1)
    top = int(torch.argmax(lg))
    best, best_tok = None, None
    for g in golds(answer):
        for variant in (g, " " + g):
            toks = pipe.tokenizer.encode(variant, add_special_tokens=False)
            if toks and (best is None or float(lp[toks[0]]) > best):
                best, best_tok = float(lp[toks[0]]), int(toks[0])
    return {"l1_sha": hashlib.sha1(lg.numpy().tobytes()).hexdigest()[:16], "l1_top1": top, "l1_top1_lp": float(lp[top]),
            "gold_lp_first": best, "gold_lp_first_tok": best_tok}


# s/item-arm priors for the cost guard: measured in the C113 smoke on A100-40GB (A 4.25, Z 4.74, C 5.14 s); C0 and B unmeasured
# on RestoreKV_plus, set to C's rate (B + 0.11 s for the second routing pass); AV = A
PRIOR_S = {"A": 4.25, "Z": 4.74, "C": 5.14, "C0": 5.14, "B": 5.25, "AV": 4.25}


def g3_check(which="G3"):
    a = pd.DataFrame([r for r in rows if r["arm"] == "A"])
    if lb is None or len(a) == 0:
        gates[which] = {"status": "not run", "mode": args.g3_mode}
        return True
    diff = float(a.score.mean() - a.lb_score.mean())
    match = float((a.predicted_answer.astype(str).str.strip() == a.lb_predicted_answer.astype(str).str.strip()).mean())
    cr_diff = float(a.effective_cr.mean() - a.lb_cr.mean())
    ok = abs(diff) <= args.g3_floor_pp and match >= 0.5 and abs(cr_diff) <= args.g3_cr_tol
    gates[which] = {"status": "pass" if ok else "FAIL", "mode": args.g3_mode, "ours_pooled": float(a.score.mean()),
                   "published_pooled": float(a.lb_score.mean()), "diff_pp": diff, "exact_pred_match": match,
                   "eff_cr_ours": float(a.effective_cr.mean()), "eff_cr_published": float(a.lb_cr.mean()), "eff_cr_diff": cr_diff,
                   "n_items": int(len(a)), "floor_pp": args.g3_floor_pp, "cr_tol": args.g3_cr_tol,
                   "items_score_differs": int((a.score != a.lb_score).sum()),
                   "per_task_diff_pp": (a.groupby("task").score.mean() - a.groupby("task").lb_score.mean()).round(3).to_dict()}
    log(f"{which}: n={len(a)} ours={a.score.mean():.3f} published={a.lb_score.mean():.3f} diff={diff:+.3f} match={match:.3f} "
        f"effCR ours={a.effective_cr.mean():.5f} pub={a.lb_cr.mean():.5f} -> {gates[which]['status']}")
    return ok


# ----------------------------------------------------------------------------- stages
assert STAGES and (STAGES[0][2] in ("A", "AV") or args.no_bare_arm), "arm A (or the AV probe) must open the stage list"
new_item_arms = 0
stop_reason = None
for stage, arm, kind, stage_ids in STAGES:
    CUR.update(stage=stage, arm=arm, kind=kind)
    todo = [i for i in stage_ids if (arm, i) not in done]
    st = gates["stages"].setdefault(stage, {"items_total": len(stage_ids), "arm": arm})
    if kind != "A":
        missing = [i for i in stage_ids if i not in A_REF]
        if missing and kind == "AV":  # the probe needs the first sandbox's A rows on the Z ids; without them it is skipped
            st["status"] = f"skipped (A reference missing on {len(missing)} ids)"
            log(f"stage {stage} skipped: A reference missing on {len(missing)} ids")
            continue
        if missing and args.no_bare_arm and kind == "C":  # rev 1.4: G1/G2 run on the ids with a stored A row only
            st["ids_without_A_reference"] = len(missing)
            log(f"stage {stage}: {len(missing)} of {len(stage_ids)} ids have no stored A row (G1/G2 not checked on them, rev 1.4)")
        elif missing:
            stop_reason = f"arm A incomplete ({len(missing)} ids) - stage {stage} not started"
            break
        if args.g3_mode == "assert" and kind == "Z" and gates["G3_early"].get("status") != "pass":
            if not g3_check("G3_early"):
                gate_fail("G3_early", "bare arm does not reproduce the published predictions; Z and later stages are not run")
        if args.g3_mode == "assert" and kind in ("C", "C0", "B") and gates["G3"].get("status") != "pass":
            if not g3_check("G3"):
                gate_fail("G3", "bare arm does not reproduce the published predictions; no treatment arm is run")
    if kind in ATTRIB_KINDS:
        missing_c = [i for i in stage_ids if i not in C_REF]
        if missing_c:
            stop_reason = f"arm C incomplete on the subset ({len(missing_c)} ids) - {arm} has no G6 reference; not started"
            break
    if not todo:
        st["status"] = "complete"
        continue
    press = make_arm(kind)
    rate = PRIOR_S[kind]
    measured = [r["wall_s"] for r in rows if r["arm"] == arm]
    if measured:
        rate = float(np.mean(measured))
    proj = args.spent_eur + job_cost_eur(extra_s=rate * len(todo))
    log(f"stage {arm}: {len(todo)} item-arms to run, rate {rate:.2f} s/item, projected cumulative list price EUR {proj:.2f}")
    st.update(status="running", projected_eur_at_start=proj, rate_used_s=rate)
    # the guard does not stop a stage at its start: it stops between items once the projection with the stage's own
    # measured rate (>= 3 items) exceeds the cap, so every started stage yields paired items up to the cap
    for n_stage, iid in enumerate(todo):
        if time.time() - JOB_T0 > args.time_budget_s:
            stop_reason = f"time budget {args.time_budget_s:.0f} s reached"
            break
        if args.stop_after_items is not None and new_item_arms >= args.stop_after_items:
            stop_reason = f"--stop-after-items {args.stop_after_items}"
            break
        meas = [r["wall_s"] for r in rows if r["arm"] == arm]
        proj_now = args.spent_eur + job_cost_eur(extra_s=(float(np.mean(meas)) if len(meas) >= 3 else rate))
        if proj_now > args.cost_cap_eur:
            stop_reason = f"cost guard: next {arm} item would bring the projection to EUR {proj_now:.2f} > cap {args.cost_cap_eur}"
            st["status"] = "stopped (cost guard)"
            break
        row = df.loc[iid]
        if STATE["first_item_started"] is None:
            STATE["first_item_started"] = time.time() - T0
            gates["timestamps"]["first_item_started_s"] = STATE["first_item_started"]
            gates["timestamps"]["first_item_started_job_s"] = time.time() - JOB_T0
            log("first-item-started")
        mnt = int(args.max_new_tokens or row["max_new_tokens"])
        ITEM.clear()
        ITEM.update(new_item_counters())
        REC.clear()
        FIRST.clear()
        reset_modules(pipe.model)
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
        fs = first_step_record(row["answer"])
        tag = f"item {iid} arm {arm}"
        if int(REC.get("compress_post_calls", 0)) != 1 or "ctx_len" not in REC:
            gate_fail("G5", f"{tag}: compress_post calls {REC.get('compress_post_calls', 0)} (must be 1)")
        rec = kept_record(pipe.model, cache, REC["ctx_len"])
        c = dict(ITEM)
        _bv = c.pop("_b_vals", None)
        if _bv:
            _allb = torch.cat(_bv)
            _q = torch.quantile(_allb, torch.tensor([0.5, 0.9, 0.99]))
            c["b_q50"], c["b_q90"], c["b_q99"] = [float(x) for x in _q]
            c["b_max"] = float(_allb.max())
        report = getattr(press, "last_report", []) if isinstance(press, MergingAnyPress) else []
        c["passthrough_layers"] = sum(1 for r in report if r.get("path") == "passthrough")
        c["skipped_layers"] = sum(1 for r in report if r.get("skipped"))
        c["evicting_layers_seen"] = sum(1 for r in report if int(r.get("n_evicted", 0) or 0) > 0)
        c["appended_rows_seen"] = max([int(r.get("n_appended", 0) or 0) for r in report] + [0])
        if kind in FOLD_ARMS:
            c["scores_missing_layers"] = sum(1 for r in report if int(r.get("n_evicted", 0) or 0) > 0 and
                                             "no full-length inner score" in str(r.get("skipped", "")))
        bias_left = sum(1 for m in attn_modules(pipe.model) if getattr(m, "anypress_logit_bias", None) is not None)
        # ---- G4 / G5 / SPEC (before the row is appended)
        if rec["restore_masked"] != 0:
            gate_fail("G4", f"{tag}: {rec['restore_masked']} restore rows masked")
        if int(REC.get("n_restore", -1)) != args.expect_restore or rec["n_restore_cache"] != args.expect_restore:
            gate_fail("G5", f"{tag}: n_restore {REC.get('n_restore')} / cache {rec['n_restore_cache']} != {args.expect_restore}")
        if kind in FOLD_ARMS:
            for k in ("w_sink", "w_masked", "w_appended", "b_sink", "b_masked", "b_appended", "keys_changed"):
                if c[k] != 0:
                    gate_fail("G4", f"{tag}: {k}={c[k]}")
            if c["passthrough_layers"]:
                gate_fail("G5", f"{tag}: {c['passthrough_layers']} layers took the pass-through path")
            if c["appended_rows_seen"] != args.expect_restore * rec["n_kv_heads"]:
                gate_fail("G4", f"{tag}: wrapper saw {c['appended_rows_seen']} appended rows, expected "
                                f"{args.expect_restore * rec['n_kv_heads']} per layer")
            if c["noevict_heads_touched"]:
                gate_fail("SPEC", f"{tag}: {c['noevict_heads_touched']} heads without eviction were modified")
            if c["scores_missing_layers"] or (rec["evicting_layers"] > 0 and c["evicting_layers_seen"] != rec["evicting_layers"]):
                gate_fail("G5", f"{tag}: scores missing on {c['scores_missing_layers']} layers / evicting layers seen "
                                f"{c['evicting_layers_seen']} vs {rec['evicting_layers']}")
            if kind in ("C", "B") and c["merged"] > 0 and (c["bias_nonzero"] == 0 or c["bias_mask_calls"] == 0):
                gate_fail("G5", f"{tag}: merged={c['merged']} but bias_nonzero={c['bias_nonzero']} calls={c['bias_mask_calls']}")
            if kind == "C0":  # zero-content kernel-path control: no writes, no non-zero bias, the zero-bias path taken
                if c["writes"] or c["bias_nonzero"]:
                    gate_fail("G5", f"{tag}: C0 wrote {c['writes']} values / {c['bias_nonzero']} non-zero bias entries")
                if c["merged"] > 0 and (int(c.get("c0_layers_forced", 0)) == 0 or c["bias_mask_calls"] == 0):
                    gate_fail("G5", f"{tag}: C0 merged={c['merged']} but zero-bias layers {c.get('c0_layers_forced', 0)} "
                                    f"calls={c['bias_mask_calls']}")
            if kind == "B" and float(c.get("b_self_maxrel", 0.0)) > args.b_maxrel_tol:
                gate_fail("G6", f"{tag}: self-fold changed values by up to {c.get('b_self_maxrel'):.3e} (rel) > {args.b_maxrel_tol}")
            gates["G4"]["item_arms_checked"] += 1
            gates["SPEC"]["noevict_heads"] += c["noevict_heads"]
        if kind in ("A", "Z", "AV"):
            if c["bias_mask_calls"] or c["writes"] or c["bias_nonzero"] or bias_left:
                gate_fail("G5", f"{tag}: bias/writes in a no-compensation arm {c} bias_left={bias_left}")
        gates["G5"]["item_arms_checked"] += 1
        # ---- G6 control integrity against arm C of the same item (C0: C's would-be bias before zeroing; B: installed bias)
        if kind in ATTRIB_KINDS:
            refc = C_REF[iid]
            g6 = gates["G6"]["per_arm"].setdefault(arm, {"items_checked": 0, "merged_equal": 0, "bias_nonzero_equal": 0,
                                                         "bias_nonzero_max_rel_diff": 0.0, "bias_sum_max_rel_diff": 0.0})
            if int(c["merged"]) != int(refc["c_merged"]):
                gate_fail("G6", f"{tag}: merged {c['merged']} vs C {refc['c_merged']}")
            bnz_w = int(c.get("c0_bias_nonzero_c", 0)) if kind == "C0" else int(c["bias_nonzero"])
            bsum_w = float(c.get("c0_bias_sum_c", 0.0)) if kind == "C0" else float(c["bias_sum"])
            rel = abs(bnz_w - int(refc["c_bias_nonzero"])) / max(1, int(refc["c_bias_nonzero"]))
            relsum = abs(bsum_w - float(refc["c_bias_sum"])) / max(1e-12, abs(float(refc["c_bias_sum"])))
            for gg in (gates["G6"], g6):
                gg["bias_nonzero_max_rel_diff"] = max(gg["bias_nonzero_max_rel_diff"], rel)
                gg["bias_sum_max_rel_diff"] = max(gg["bias_sum_max_rel_diff"], relsum)
            if rel > args.g6_bias_tol:
                gate_fail("G6", f"{tag}: C-routing bias entries {bnz_w} vs C {refc['c_bias_nonzero']} (rel {rel:.2e})")
            if relsum > 1e-3:
                gate_fail("G6", f"{tag}: C-routing bias sum {bsum_w:.6g} vs C {refc['c_bias_sum']:.6g} (rel {relsum:.2e})")
            if kind == "B":
                gates["G6"]["b_self_maxrel_max"] = max(gates["G6"]["b_self_maxrel_max"], float(c.get("b_self_maxrel", 0.0)))
            for gg in (gates["G6"], g6):
                gg["merged_equal"] += 1
                gg["bias_nonzero_equal"] += int(bnz_w == int(refc["c_bias_nonzero"]))
                gg["items_checked"] += 1
        # ---- G1 / G2 against arm A of the same item (rev 1.4: only where a stored A row exists; always true without --no-bare-arm)
        if kind != "A" and iid in A_REF:
            ref = A_REF[iid]
            if rec["kept"] != ref["kept"] or rec["mask_sha1"] != ref["mask_sha1"]:
                nd = sum(int(x != y) for la, lb_ in zip(rec["kept"], ref["kept"]) for x, y in zip(la, lb_))
                if kind == "AV":
                    gates["AV"]["status"] = "FAIL (cross-sandbox kept-set drift)"
                gate_fail("G1", f"{tag}: {nd} (layer, head) kept counts differ; mask_sha1 {rec['mask_sha1']} vs {ref['mask_sha1']}")
            d = abs(rec["effective_cr"] - ref["effective_cr"])
            gates["G2"]["max_abs_diff"] = max(gates["G2"]["max_abs_diff"], d)
            if d > 1e-6:
                gate_fail("G2", f"{tag}: effective_cr {rec['effective_cr']} vs A {ref['effective_cr']}")
            gates["G1"]["items_checked"] += 1
            gates["G2"]["items_checked"] += 1
            gates["score_sha1_equal_to_A"]["checked"] += 1
            gates["score_sha1_equal_to_A"]["equal"] += int(REC.get("score_sha1") == ref.get("score_sha1"))
            if kind == "AV":  # re-run noise of the bare press across sandboxes (A21's criterion); the kept set is gated above
                pr = gates["AV"]["probes"].setdefault(arm, {"items_checked": 0, "kept_equal": 0, "pred_equal": 0, "score_equal": 0,
                                                            "l1_equal": 0})
                pr["items_checked"] += 1
                pr["kept_equal"] += 1
                pr["pred_equal"] += int(pred == ref["predicted_answer"])
                pr["score_equal"] += int(ruler_score(pred, row["answer"], row["task"]) == float(ref["score"]))
                pr["l1_equal"] += int(fs["l1_sha"] is not None and fs["l1_sha"] == ref.get("l1_sha"))
                if not str(gates["AV"].get("status", "")).startswith("FAIL"):
                    gates["AV"]["status"] = "pass"
            if kind == "Z":
                if pred != ref["predicted_answer"]:
                    gate_fail("SPEC", f"{tag}: W(P, none) prediction differs from A: {pred!r} vs {ref['predicted_answer']!r}")
                gates["SPEC"]["z_items_checked"] += 1
        elif kind != "A":
            gates["G1"]["items_without_reference"] = gates["G1"].get("items_without_reference", 0) + 1
        r = {"claim": args.claim, "press": "RestoreKV_plus", "cr": args.cr, "arm": arm, "item_id": int(iid), "task": row["task"],
             "attn": ATTN, "in_subset": bool(iid in SUBSET_SET), "stage": stage, "kind": kind, **fs,
             "predicted_answer": pred, "score": ruler_score(pred, row["answer"], row["task"]), "answer": str(row["answer"]),
             "max_new_tokens": mnt, "wall_s": wall,
             "lb_predicted_answer": (str(lb["predicted_answer"].iloc[iid]) if lb is not None else None),
             "lb_score": (ruler_score(lb["predicted_answer"].iloc[iid], row["answer"], row["task"]) if lb is not None else None),
             "lb_cr": (float(lb["compression_ratio"].iloc[iid]) if lb is not None else None),
             "score_sha1": REC.get("score_sha1"), "requested_cr_at_compress": REC.get("requested_cr"), "n_restore": REC.get("n_restore"),
             **{k: rec[k] for k in ("ctx_len", "cache_len", "n_layers", "n_kv_heads", "kept_total", "masked_total", "evicting_layers",
                                    "restore_masked", "mask_sha1", "kept_sha1", "effective_cr")},
             **{f"c_{k}": v for k, v in c.items()}}
        if kind == "A":
            r["kept"] = rec["kept"]
            A_REF[iid] = r
        if kind == "C":
            C_REF[iid] = r
        rows.append(r)
        done.add((arm, iid))
        ROWS_FH.write(json.dumps(r, default=str) + "\n")
        ROWS_FH.flush()
        new_item_arms += 1
        flush()
        if kind == "A" and args.g3_mode == "assert" and gates["G3_early"].get("status") != "pass" and len(A_REF) >= N_EARLY:
            ok_early = g3_check("G3_early")
            flush()
            if not ok_early:
                gate_fail("G3_early", f"bare arm does not reproduce the published predictions after {len(A_REF)} items; "
                                      "no further item or treatment arm is run")
        log(f"{arm} {n_stage + 1}/{len(todo)} id={iid} task={row['task']} ctx={rec['ctx_len']} effCR={rec['effective_cr']:.4f} "
            f"score={r['score']:.0f} lb={r['lb_score']} merged={c['merged']} bias_nz={c['bias_nonzero']} "
            f"calls={c['bias_mask_calls']} wall={wall:.2f}s cost~EUR{args.spent_eur + job_cost_eur():.2f}")
    if stop_reason:
        if st.get("status") == "running":
            st["status"] = "incomplete"
        break
    st["status"] = "complete"
    st["wall_mean_s"] = float(np.mean([r["wall_s"] for r in rows if r["arm"] == arm]))
    if stage == "A1" and gates["G3_early"].get("status") != "pass":
        ok = g3_check("G3_early")
        flush()
        if not ok and args.g3_mode == "assert":
            gate_fail("G3_early", "bare arm does not reproduce the published predictions; Z and later stages are not run")
    if stage == "A2":
        ok = g3_check("G3")
        flush()
        if not ok and args.g3_mode == "assert":
            gate_fail("G3", "bare arm does not reproduce the published predictions; no treatment arm is run")

gates["stop_reason"] = stop_reason
gates["complete"] = stop_reason is None
dfp = pd.DataFrame(rows)
if len(dfp):
    log("per-arm pooled means: " + json.dumps(dfp.groupby("arm")["score"].mean().round(3).to_dict()))
    gates["wall_per_item_arm_s"] = {a: float(v) for a, v in dfp.groupby("arm")["wall_s"].mean().items()}
if torch.cuda.is_available():
    gates["max_memory_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
gates["timestamps"]["end_s"] = time.time() - T0
gates["timestamps"]["end_job_s"] = time.time() - JOB_T0
STATE["stop"] = True
flush()
log(f"done: stop_reason={stop_reason} rows={len(rows)} job list-price EUR {job_cost_eur():.3f}")
