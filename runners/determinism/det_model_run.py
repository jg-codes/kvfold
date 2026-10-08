"""det_model_run.py - one end-to-end fold run on a small model on CPU (separate process per run).
usage: python det_model_run.py --tag T --scatter atomic|sorted|orig --threads N --det on|off --out DIR"""
import argparse, glob, hashlib, json, os, sys, time
ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True); ap.add_argument("--scatter", default="atomic"); ap.add_argument("--threads", type=int, default=3)
ap.add_argument("--det", default="off"); ap.add_argument("--out", default="out"); ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
ap.add_argument("--cr", type=float, default=0.9375); ap.add_argument("--max-new", type=int, default=24)
args = ap.parse_args()
import numpy as np, pandas as pd, torch
torch.set_num_threads(args.threads)
if args.det == "on":
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        import torch.utils.deterministic as _d; _d.fill_uninitialized_memory = False
    except Exception:
        pass
import kvpress
from kvpress import SnapKVPress
from kvpress.presses.merging_any_press import MergingAnyPress, MassFold
from transformers import pipeline
store = {}
orig_call = MassFold.__call__
def rec_call(self, ks):
    r = orig_call(self, ks)
    if r is not None and r.values is not None:
        store[int(ks.layer_idx)] = (r.values.detach().float().numpy().copy(), r.logit_bias.detach().float().numpy().copy())
    return r
MassFold.__call__ = rec_call
pq = sorted(glob.glob(os.path.join(os.environ["HF_HOME"], "hub", "datasets--simonjegou--ruler", "snapshots", "*", "4096", "test-00000-of-00001.parquet")))[0]
df = pd.read_parquet(pq)
row = df[df.task == "qa_1"].iloc[0]
pipe = pipeline("kv-press-text-generation", model=args.model, model_kwargs={"dtype": torch.float32, "attn_implementation": "sdpa"}, device="cpu")
calls = []
pipe.model.lm_head.register_forward_hook(lambda m, i, o: calls.append({"shape": list(o.shape), "sha1": hashlib.sha1(o[0, -1].detach().float().numpy().tobytes()).hexdigest()[:16]}))
inner = SnapKVPress(compression_ratio=args.cr, window_size=64, kernel_size=5)
if args.scatter == "orig":
    comp = MassFold(score_map="log")
else:
    comp = MassFold(score_map="log", scatter=args.scatter)
press = MergingAnyPress(inner, compensation=comp, n_sink=4)
t0 = time.time()
out = pipe(row["context"], question=row["question"], answer_prefix=row["answer_prefix"] or "", press=press, max_new_tokens=args.max_new)
answer = out["answer"]
layers = sorted(store)
per_layer = {str(l): {"values_sha1": hashlib.sha1(store[l][0].tobytes()).hexdigest()[:16], "bias_sha1": hashlib.sha1(store[l][1].tobytes()).hexdigest()[:16]} for l in layers}
np.savez_compressed(os.path.join(args.out, f"det_run_{args.tag}.npz"), **{f"v{l}": store[l][0] for l in layers}, **{f"b{l}": store[l][1] for l in layers})
rec = {"tag": args.tag, "scatter": args.scatter, "threads": args.threads, "det": args.det, "model": args.model, "kvpress_file": kvpress.__file__, "torch": torch.__version__,
       "item_task": str(row["task"]), "answer": answer, "answer_sha1": hashlib.sha1(answer.encode()).hexdigest()[:16], "n_layers_folded": len(layers),
       "lm_head_calls": calls[:3], "n_lm_head_calls": len(calls), "first_step_sha1": calls[1]["sha1"] if len(calls) > 1 else None,
       "per_layer": per_layer, "fold_all_sha1": hashlib.sha1("".join(per_layer[str(l)]["values_sha1"] + per_layer[str(l)]["bias_sha1"] for l in layers).encode()).hexdigest()[:16],
       "wall_s": time.time() - t0}
json.dump(rec, open(os.path.join(args.out, f"det_run_{args.tag}.json"), "w"), indent=1)
print(json.dumps({k: rec[k] for k in ("tag", "scatter", "threads", "det", "answer_sha1", "first_step_sha1", "fold_all_sha1", "n_layers_folded", "wall_s")}))
