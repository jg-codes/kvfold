#!/usr/bin/env python
"""sig_build_stats.py rev 1 -- statistics of the RULER-4096 full-set significance run (wiki claim C141), written and tested BEFORE any
significance-run row was read (tests: synthetic fixture and the C111 smoke rows).

Modes
  final   --rows linear=<jsonl> [mlp=<jsonl>] [kvzip=<jsonl>]  all read-outs -> <prefix>_stats.json, <prefix>_paired_open.csv
  smoke   --rows kvzip=<jsonl> --kill-rule                      KVzip smoke: kill rule, placebo contrasts, gates
Scores: per item with the leaderboard metric (kvpress evaluation/benchmarks/ruler/calculate_metrics.py semantics: control characters
stripped from the prediction, golds = parquet answer list, qa_* max over golds, else mean over golds, x100); copied verbatim from
build_c127_stats.py rev 3 (validated against metrics.json of three leaderboard entries on 13/13 tasks). The runner scorer (row field
'score', golds parsed from str(answer)) is the sensitivity. Published per-item predictions are the 'lb_predicted_answer' field the runner
stored in every row (read from the volume file whose sha256 the job asserted).
Primary (per operating point o): d_i = s_W(i) - s_A(i) over the ids; item bootstrap 95% CI (B=10,000), one-sided item-level sign-flip p
(B=100,000); Holm over the family {linear, mlp, kvzip, kvzip97} (p = 1 for a missing operating point; rev 2 = amendment rev 1.2: the family of the registration had three
points, the fourth is KVzip CR 0.97 and the three-point Holm of the registration is reported as primary_family_original3); pass = CI lower bound > 0 and Holm-rejected.
kvzip97 has no published leaderboard entry: no S1, G3 'not applicable' (replaced by the target-CR check |mean eCR(A) - 0.97| <= 0.01), per-task rows without registered margins.
No read-out uses an outcome to choose another read-out.
"""
import argparse
import hashlib
import json
import re
import sys

import numpy as np
import pandas as pd
from scipy import stats as sps

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["final", "smoke"], required=True)
ap.add_argument("--rows", nargs="+", required=True, help="op=path (linear, mlp, kvzip, kvzip97)")
ap.add_argument("--golds", required=True)
ap.add_argument("--params", required=True)
ap.add_argument("--params-sha256", required=True)
ap.add_argument("--gates", nargs="*", default=[], help="gates.json files (op=path) for the gate summary")
ap.add_argument("--out-prefix", default="sig")
ap.add_argument("--n-expected", type=int, default=6500)
ap.add_argument("--items97", default=None, help="kvzip97: the registered 1,300-id item file (completeness check)")
ap.add_argument("--b-boot", type=int, default=None)
ap.add_argument("--b-flip", type=int, default=None)
args = ap.parse_args()


def fsha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


assert fsha(args.params) == args.params_sha256, "registered parameters file hash mismatch"
P = json.load(open(args.params))
ORDER = P["tasks_report_order"]
B_BOOT = args.b_boot or P["bootstrap"]["B"]
B_FLIP = args.b_flip or P["signflip"]["B_primary"]
B_FLIP2 = min(B_FLIP, P["signflip"]["B_secondary"])
SEED_B, SEED_F = P["bootstrap"]["seed"], P["signflip"]["seed"]
NP = re.compile(r"[\x00-\x1f]")


# ----------------------------------------------------------------------------- scorer (verbatim from build_c127_stats.py rev 3)
def clean(x):
    return NP.sub("", str(x).strip()).strip()


def score_lb(pred, refs, task):
    p = clean(pred).lower()
    h = [1.0 if r.lower() in p else 0.0 for r in refs]
    return 100.0 * (max(h) if task.split("_")[0] == "qa" else sum(h) / len(h))


def lb_task_metric(preds, refs, task):
    preds = [clean(x) for x in preds]
    if task.split("_")[0] == "qa":
        score = sum([max([1.0 if r.lower() in pred.lower() else 0.0 for r in ref]) for pred, ref in zip(preds, refs)]) / len(preds) * 100
    else:
        score = sum([sum([1.0 if r.lower() in pred.lower() else 0.0 for r in ref]) / len(ref) for pred, ref in zip(preds, refs)]) / len(preds) * 100
    return round(score, 2)


# ----------------------------------------------------------------------------- statistics
def boot_ci(d, B=B_BOOT, seed=SEED_B, strata=None):
    d = np.asarray(d, float)
    n = len(d)
    rng = np.random.default_rng(seed)
    res = np.empty(B)
    for b0 in range(0, B, 500):
        bb = min(500, B - b0)
        if strata is None:
            res[b0:b0 + bb] = d[rng.integers(0, n, size=(bb, n))].mean(axis=1)
        else:
            tot = np.zeros(bb)
            for s in np.unique(strata):
                ds = d[strata == s]
                tot += ds[rng.integers(0, len(ds), size=(bb, len(ds)))].sum(axis=1)
            res[b0:b0 + bb] = tot / n
    lo, hi, lb5 = np.percentile(res, [2.5, 97.5, 5.0])
    return float(lo), float(hi), float(lb5)


def signflip_p(d, B, seed=SEED_F, two_sided=False):
    d = np.asarray(d, float)
    n = len(d)
    t0 = d.mean()
    rng = np.random.default_rng(seed)
    cnt, tot = 0, d.sum()
    chunk = max(1, int(4e6 // max(n, 1)))
    for b0 in range(0, B, chunk):
        bb = min(chunk, B - b0)
        bits = rng.integers(0, 2, size=(bb, n), dtype=np.int8).astype(np.float64)
        t = (2.0 * (bits @ d) - tot) / n
        cnt += int((np.abs(t) >= abs(t0) - 1e-12).sum() if two_sided else (t >= t0 - 1e-12).sum())
    return (1 + cnt) / (B + 1)


def holm(pdict, alpha):
    items = sorted(pdict.items(), key=lambda kv: kv[1])
    m = len(items)
    out, stop, running = {}, False, 0.0
    for k, (name, pv) in enumerate(items):
        thr = alpha / (m - k)
        rej = (not stop) and pv <= thr
        stop = stop or not rej
        running = max(running, min(1.0, (m - k) * pv))
        out[name] = {"p": pv, "threshold": thr, "reject": bool(rej), "p_holm_adj": running}
    return out


def cp_ci(k, n, a=0.05):
    if n == 0:
        return [float("nan"), float("nan")]
    lo = 0.0 if k == 0 else float(sps.beta.ppf(a / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(sps.beta.ppf(1 - a / 2, k + 1, n - k))
    return [lo, hi]


def paired(a, b, label="", B=B_BOOT, strata=None, names=("first", "second")):
    """delta = mean(a) - mean(b); mean_<names[0]> and mean_<names[1]> carry the two arm means"""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    lo, hi, lb5 = boot_ci(d, B)
    r = {"n": int(len(d)), f"mean_{names[0]}": float(a.mean()), f"mean_{names[1]}": float(b.mean()), "delta_pp": float(d.mean()), "ci95": [lo, hi],
         "lb95_onesided": lb5, "sd_item": float(d.std(ddof=1)) if len(d) > 1 else None, "better": int((d > 0).sum()),
         "worse": int((d < 0).sum()), "tied": int((d == 0).sum())}
    if strata is not None:
        r["ci95_task_stratified"] = list(boot_ci(d, B, strata=strata)[:2])
    return r


# ----------------------------------------------------------------------------- data
GJ = json.load(open(args.golds))
GOLD, TASKS = GJ["golds"], GJ["task"]
assert len(GOLD) == len(TASKS) == 6500


def load_rows(path):
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


ROW_OP = {"kvzip97": "kvzip"}  # the runner writes op='kvzip' for both KVzip ratios; the operating point is identified by the file / --rows key
TARGET_CR = {"kvzip": 0.875, "kvzip97": 0.97}
ITEMS97 = sorted(json.load(open(args.items97))["item_ids"]) if args.items97 else None


def op_frame(path, op):
    df = pd.DataFrame(load_rows(path))
    if "op" in df.columns:
        df = df[df["op"] == ROW_OP.get(op, op)]
    assert not df.duplicated(["arm", "item_id"]).any(), "duplicate (arm, item_id) rows"
    return df


def arm_table(df, arm):
    t = df[df.arm == arm].copy()
    t["item_id"] = t["item_id"].astype(int)
    t = t.set_index("item_id").sort_index()
    t["s_lb"] = [score_lb(p, GOLD[i], TASKS[i]) for i, p in zip(t.index, t.predicted_answer)]
    return t


def gate_summary(path):
    g = json.load(open(path))
    keys = ("G1", "G2", "G3", "G4", "G5", "SPEC", "G6", "AV", "CV")
    out = {k: ({kk: vv for kk, vv in g[k].items() if kk != "items"} if isinstance(g.get(k), dict) else g.get(k)) for k in keys if k in g}
    out.update({k: g.get(k) for k in ("stop_reason", "complete", "wall_s", "argv", "op", "cr", "score_map")})
    out["cost"] = g.get("cost")
    return out


def analyse_op(op, df, params):
    A, W = arm_table(df, "A"), arm_table(df, "C")
    ids = sorted(set(A.index) & set(W.index))
    res = {"op": op, "n_A": int(len(A)), "n_W": int(len(W)), "n_paired": int(len(ids)),
           "complete": (bool(ITEMS97 is not None and ids == ITEMS97) if (op == "kvzip97" and args.mode == "final") else
                        bool(len(ids) == args.n_expected and ids == list(range(6500)) if args.n_expected == 6500 else len(ids) == args.n_expected))}
    A, W = A.loc[ids], W.loc[ids]
    task = np.array([TASKS[i] for i in ids])
    sA, sW = A.s_lb.to_numpy(), W.s_lb.to_numpy()
    has_ref = bool("lb_predicted_answer" in A.columns and A.lb_predicted_answer.notna().all() and W.lb_predicted_answer.notna().all())
    res["has_published_reference"] = has_ref
    if has_ref:
        pub_pred = A.lb_predicted_answer.astype(str)
        assert (pub_pred.values == W.lb_predicted_answer.astype(str).values).all(), "published prediction differs between A and W rows"
        sP = np.array([score_lb(p, GOLD[i], TASKS[i]) for i, p in zip(ids, pub_pred)])
    else:
        assert op in ("kvzip97",), f"{op}: rows without published predictions"
        pub_pred, sP = None, np.full(len(ids), np.nan)
    d = sW - sA
    # ---- integrity from the rows
    ecr = (W.effective_cr - A.effective_cr).abs()
    integ = {"max_abs_effective_cr_diff": float(ecr.max()), "kept_sha1_equal": int((A.kept_sha1.values == W.kept_sha1.values).sum()),
             "mask_sha1_equal": int((A.mask_sha1.values == W.mask_sha1.values).sum()), "n": int(len(ids)),
             "kept_total_equal": int((A.kept_total.values == W.kept_total.values).sum()),
             "w_sink_nonzero_items": int((W.c_w_sink != 0).sum()), "w_masked_nonzero_items": int((W.c_w_masked != 0).sum()),
             "w_appended_nonzero_items": int((W.c_w_appended != 0).sum()), "b_sink_nonzero_items": int((W.c_b_sink != 0).sum()),
             "b_masked_nonzero_items": int((W.c_b_masked != 0).sum()), "b_appended_nonzero_items": int((W.c_b_appended != 0).sum()),
             "keys_changed_items": int((W.c_keys_changed != 0).sum()), "passthrough_items": int((W.c_passthrough_layers != 0).sum()),
             "noevict_heads_touched_items": int((W.c_noevict_heads_touched != 0).sum()),
             "items_merged_zero": int((W.c_merged == 0).sum()), "merged_mean": float(W.c_merged.mean()),
             "bias_nonzero_mean": float(W.c_bias_nonzero.mean()),
             "merged_without_bias_items": int(((W.c_merged > 0) & ((W.c_bias_nonzero == 0) | (W.c_bias_mask_calls == 0))).sum()),
             "A_bias_or_writes_items": int(((A.c_bias_mask_calls != 0) | (A.c_writes != 0) | (A.c_bias_nonzero != 0)).sum()),
             "effective_cr_mean": float(A.effective_cr.mean()), "ctx_len_mean": float(A.ctx_len.mean())}
    integ["pass"] = bool(integ["max_abs_effective_cr_diff"] <= 1e-6 and integ["kept_sha1_equal"] == len(ids) and integ["mask_sha1_equal"] == len(ids)
                         and integ["w_sink_nonzero_items"] + integ["w_masked_nonzero_items"] + integ["w_appended_nonzero_items"] == 0
                         and integ["b_sink_nonzero_items"] + integ["b_masked_nonzero_items"] + integ["b_appended_nonzero_items"] == 0
                         and integ["keys_changed_items"] == 0 and integ["passthrough_items"] == 0 and integ["noevict_heads_touched_items"] == 0
                         and integ["merged_without_bias_items"] == 0 and integ["A_bias_or_writes_items"] == 0)
    res["integrity"] = integ
    # ---- primary
    prim = paired(sW, sA, B=B_BOOT, strata=task, names=("W", "A"))
    prim["signflip_p_one_sided"] = signflip_p(d, B_FLIP)
    prim["signflip_p_two_sided"] = signflip_p(d, B_FLIP2, two_sided=True)
    prim["leaderboard_style"] = {}
    for nm, arr_pred in (("A", A.predicted_answer.tolist()), ("W", W.predicted_answer.tolist())) + ((("published", pub_pred.tolist()),) if has_ref else ()):
        per = {}
        for t in ORDER:
            m = np.where(task == t)[0]
            if len(m):
                per[t] = lb_task_metric([arr_pred[j] for j in m], [GOLD[ids[j]] for j in m], t)
        prim["leaderboard_style"][nm] = {"per_task": per, "mean": float(np.mean(list(per.values()))), "n_tasks": len(per)}
    prim["sensitivity_runner_scorer"] = paired(W.score.astype(float).to_numpy(), A.score.astype(float).to_numpy(), B=B_BOOT, names=("W", "A"))
    res["primary"] = prim
    # ---- S1: W vs published; G3: A vs published (not available at an operating point without a published entry)
    if has_ref:
        s1 = paired(sW, sP, B=B_BOOT, names=("W", "published"))
        s1["signflip_p_one_sided"] = signflip_p(sW - sP, B_FLIP2)
        res["S1_W_vs_published"] = s1
        g3 = paired(sA, sP, B=B_BOOT, names=("A", "published"))
        g3["exact_string_match"] = float((A.predicted_answer.astype(str).str.strip().values == pub_pred.str.strip().values).mean())
        g3["effective_cr_A_mean"] = float(A.effective_cr.mean())
        g3["published_cr_mean"] = float(A.lb_cr.mean())
        g3["effective_cr_diff"] = g3["effective_cr_A_mean"] - g3["published_cr_mean"]
        g3["per_item_cr_max_abs_diff"] = float((A.effective_cr - A.lb_cr).abs().max())
        g3c = params["g3_full"]
        g3["criterion"] = {"abs_pooled_diff_pp_max": g3c["abs_pooled_diff_pp_max"], "exact_match_min": g3c["exact_match_min"],
                           "abs_effective_cr_diff_max": g3c["abs_effective_cr_diff_max"]}
        g3["pass"] = bool(abs(g3["delta_pp"]) <= g3c["abs_pooled_diff_pp_max"] and g3["exact_string_match"] >= g3c["exact_match_min"]
                          and abs(g3["effective_cr_diff"]) <= g3c["abs_effective_cr_diff_max"])
        g3["complete"] = res["complete"]
        res["G3_full"] = g3

    else:
        res["S1_W_vs_published"] = None
        ecr_a = float(A.effective_cr.mean())
        res["G3_full"] = {"applicable": False, "reason": "no published leaderboard entry at this operating point (KVzip CR 0.97)", "effective_cr_A_mean": ecr_a,
                          "target_cr": TARGET_CR[op], "effective_cr_target_diff": ecr_a - TARGET_CR[op], "per_item_cr_max_abs_diff_to_target": float((A.effective_cr - TARGET_CR[op]).abs().max()),
                          "pass": bool(abs(ecr_a - TARGET_CR[op]) <= 0.01), "criterion": "|mean effective CR of A - target CR| <= 0.01 (amendment rev 1.2; replaces the published-reference G3)",
                          "complete": res["complete"]}
    # ---- per task
    per_task, pvals = [], {}
    reg = op in params["margins_pp_n500"]  # kvzip97: no smoke, no registered margins -> descriptive per-task rows
    m500 = params["margins_pp_n500"][op] if reg else None
    sd_reg = params["registered_sd"]["sd_reg"][op] if reg else None
    for t in ORDER:
        m = np.where(task == t)[0]
        if not len(m):
            continue
        dt = d[m]
        n_t = len(m)
        marg = 1.96 * sd_reg[t] / np.sqrt(n_t) if reg else None
        lo, hi, lb5 = boot_ci(dt)
        p2 = signflip_p(dt, B_FLIP2, two_sided=True)
        pvals[t] = p2
        per_task.append({"task": t, "n": int(n_t), "A": float(sA[m].mean()), "W": float(sW[m].mean()), "published": (float(sP[m].mean()) if has_ref else None),
                         "delta": float(dt.mean()), "ci95": [lo, hi], "lb95_onesided": lb5, "margin_pp": (float(marg) if reg else None),
                         "margin_pp_n500_registered": (float(m500[t]) if reg else None), "registered_sd": (float(sd_reg[t]) if reg else None),
                         "observed_sd": float(dt.std(ddof=1)), "noninferior": (bool(lb5 > -marg) if reg else None), "harm_ci_below_zero": bool(hi < 0),
                         "below_retired_uniform_floor": (bool(lb5 <= -P["noninferiority"]["retired_uniform_floor_pp_n500"]) if reg else None),
                         "better": int((dt > 0).sum()), "worse": int((dt < 0).sum()), "tied": int((dt == 0).sum()),
                         "signflip_p_two_sided": p2, "W_minus_published": (float((sW[m] - sP[m]).mean()) if has_ref else None),
                         "A_minus_published": (float((sA[m] - sP[m]).mean()) if has_ref else None)})
    hh = holm(pvals, 0.05)
    for r in per_task:
        r["signflip_p_holm13"] = hh[r["task"]]["p_holm_adj"]
        r["holm13_reject"] = hh[r["task"]]["reject"]
    res["per_task"] = per_task
    res["noninferiority_all_tasks"] = (bool(all(r["noninferior"] for r in per_task)) and len(per_task) == 13) if reg else None
    res["tasks_not_noninferior"] = [r["task"] for r in per_task if r["noninferior"] is False]
    res["tasks_harm_ci_below_zero"] = [r["task"] for r in per_task if r["harm_ci_below_zero"]]
    # ---- repair / break (W vs A)
    rep = (sA < 100) & (sW > sA)
    brk = (sA == 100) & (sW < 100)
    n_lt, n_eq = int((sA < 100).sum()), int((sA == 100).sum())
    res["repair_break"] = {"repairs": int(rep.sum()), "of_A_below_100": n_lt, "repair_rate": float(rep.sum() / max(1, n_lt)),
                           "repair_ci": cp_ci(int(rep.sum()), n_lt), "breaks": int(brk.sum()), "of_A_at_100": n_eq,
                           "break_rate": float(brk.sum() / max(1, n_eq)), "break_ci": cp_ci(int(brk.sum()), n_eq),
                           "per_task": {t: {"repairs": int((rep & (task == t)).sum()), "breaks": int((brk & (task == t)).sum()),
                                            "A_below_100": int(((sA < 100) & (task == t)).sum()), "A_at_100": int(((sA == 100) & (task == t)).sum())}
                                        for t in ORDER}}
    res["string_identical_W_vs_A"] = int((A.predicted_answer.values == W.predicted_answer.values).sum())
    res["timing_s_per_item"] = {"A": float(A.wall_s.mean()), "W": float(W.wall_s.mean())}
    # ---- placebo arms
    plac = {}
    for arm in ("C0", "Cperm"):
        if (df.arm == arm).any():
            plac[arm] = arm_table(df, arm)
    if plac:
        pids = sorted(set.intersection(*[set(v.index) for v in plac.values()]) & set(ids))
        pa = pd.Series(sA, index=ids).loc[pids].to_numpy()
        pw = pd.Series(sW, index=ids).loc[pids].to_numpy()
        arr = {"A": pa, "C": pw, **{k: v.loc[pids].s_lb.to_numpy() for k, v in plac.items()}}
        ptask = np.array([TASKS[i] for i in pids])
        con = {}
        for a_, b_ in (("C", "A"), ("C", "C0"), ("C", "Cperm"), ("C0", "A"), ("Cperm", "A")):
            if a_ in arr and b_ in arr:
                r = paired(arr[a_], arr[b_], B=B_BOOT, names=(a_, b_))
                r["discordant"] = int((arr[a_] != arr[b_]).sum())
                con[f"{a_} - {b_}"] = r
        att = None
        if "C - C0" in con and "C - Cperm" in con:
            c0, cp_ = con["C - C0"], con["C - Cperm"]
            if c0["ci95"][0] > 0 and cp_["ci95"][0] > 0:
                att = "value content (both CI lower bounds > 0)"
            elif c0["delta_pp"] > 0 and cp_["delta_pp"] > 0:
                att = "direction only: value content (both means > 0, CI includes 0)"
            elif c0["delta_pp"] <= 0:
                att = "bias / kernel path (C - C0 <= 0)"
            else:
                att = "weights and bias rather than value content (C - C0 > 0, C - Cperm <= 0)"
        res["placebo"] = {"n_ids": int(len(pids)), "per_task_n": {t: int((ptask == t).sum()) for t in ORDER}, "contrasts": con, "attribution": att,
                          "means": {k: float(v.mean()) for k, v in arr.items()},
                          "timing_s_per_item": {k: float(v.loc[pids].wall_s.mean()) for k, v in plac.items()}}
    return res, pd.DataFrame({"op": op, "item_id": ids, "task": task, "s_A": sA, "s_W": sW, "s_published": sP, "d_W_minus_A": d,
                              "d_W_minus_published": sW - sP})


def kill_rule(res, params):
    """KVzip smoke kill rule (registered): any task d < -margin_n15, pooled CI upper < 0, pooled mean <= 0."""
    op = res["op"]
    m15 = params["margins_pp_n15"][op]
    sd_reg = params["registered_sd"]["sd_reg"][op]
    fails = [r["task"] for r in res["per_task"] if r["delta"] < -1.96 * sd_reg[r["task"]] / np.sqrt(r["n"])]
    prim = res["primary"]
    reasons = [f"task {t} below its registered n=15 margin" for t in fails]
    if prim["ci95"][1] < 0:
        reasons.append("pooled CI upper < 0")
    if prim["delta_pp"] <= 0:
        reasons.append("pooled mean <= 0")
    return {"fired": bool(reasons), "reasons": reasons, "tasks_below_margin": fails,
            "margins_n15": {t: float(m15[t]) for t in ORDER}, "complete": bool(res["n_paired"] == 195)}


if __name__ == "__main__":
    out = {"mode": args.mode, "params_sha256": args.params_sha256, "golds_sha256": fsha(args.golds), "rows_sha256": {}, "ops": {}, "n_expected": args.n_expected,
           "B_boot": B_BOOT, "B_flip": B_FLIP, "B_flip_secondary": B_FLIP2}
    opened = []
    for spec in args.rows:
        op, path = spec.split("=", 1)
        out["rows_sha256"][op] = fsha(path)
        df = op_frame(path, op)
        if args.mode == "smoke":
            assert args.n_expected == 195
        res, paired_df = analyse_op(op, df, P)
        if args.mode == "smoke":
            res["kill_rule"] = kill_rule(res, P)
        out["ops"][op] = res
        opened.append(paired_df)
    if args.mode == "final" and "kvzip" in out["ops"] and "kvzip97" in out["ops"]:  # S6 (exploratory, amendment rev 1.2): fold gain at CR 0.97 vs 0.875 on the shared ids; no decision depends on it
        pf = {fr["op"].iloc[0]: fr.set_index("item_id") for fr in opened}
        shared = sorted(set(pf["kvzip"].index) & set(pf["kvzip97"].index))
        if shared:
            a97, a875 = pf["kvzip97"].loc[shared], pf["kvzip"].loc[shared]
            s6 = paired(a97.d_W_minus_A.to_numpy(), a875.d_W_minus_A.to_numpy(), B=B_BOOT, names=("d_097", "d_0875"))
            s6.update({"n_shared": int(len(shared)), "bare_A_mean_097": float(a97.s_A.mean()), "bare_A_mean_0875": float(a875.s_A.mean()),
                       "W_mean_097": float(a97.s_W.mean()), "W_mean_0875": float(a875.s_W.mean()),
                       "note": "exploratory: mean over the shared ids of (W - A at 0.97) - (W - A at 0.875); descriptive, no p-value, no decision depends on it"})
            out["S6_dose_response_kvzip"] = s6
    gsum = {}
    for spec in args.gates:
        op, path = spec.split("=", 1)
        gsum.setdefault(op, {})[path] = gate_summary(path)
    out["gates"] = gsum
    if args.mode == "final":
        def family_table(members):
            pdict = {op: out["ops"][op]["primary"]["signflip_p_one_sided"] if op in out["ops"] else 1.0 for op in members}
            hh = holm(pdict, P["holm"]["alpha"])
            fam = {}
            for op in members:
                r = out["ops"].get(op)
                ci_lo = r["primary"]["ci95"][0] if r else None
                fam[op] = {**hh[op], "run": bool(r), "complete": bool(r and r["complete"]), "ci95_lower": ci_lo,
                           "pass": bool(r and ci_lo > 0 and hh[op]["reject"]), "provisional": bool(not (r and r["complete"]))}
            return fam
        FAMILY = list(P["holm"]["family"]) + ([] if "kvzip97" in P["holm"]["family"] else ["kvzip97"])
        out["family_members"] = FAMILY
        out["primary_family"] = family_table(FAMILY)  # amendment rev 1.2: four points
        out["primary_family_original3"] = family_table(list(P["holm"]["family"]))  # the family of the registration rev 1.0/1.1 (for transparency)
    json.dump(out, open(f"{args.out_prefix}_stats.json", "w"), indent=1)
    pd.concat(opened).to_csv(f"{args.out_prefix}_paired_open.csv", index=False)
    print(json.dumps({op: {"n": r["n_paired"], "delta": round(r["primary"]["delta_pp"], 3), "ci": [round(x, 3) for x in r["primary"]["ci95"]],
                           "p1": r["primary"]["signflip_p_one_sided"]} for op, r in out["ops"].items()}))
