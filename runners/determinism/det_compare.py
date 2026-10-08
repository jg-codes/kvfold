"""det_compare.py - pairwise comparison of det_run_<tag>.json/.npz.  usage: python det_compare.py DIR tagA:tagB [tagC:tagD ...]"""
import json, sys
import numpy as np
d = sys.argv[1]; out = {}
for pair in sys.argv[2:]:
    a, b = pair.split(":")
    ra = json.load(open(f"{d}/det_run_{a}.json")); rb = json.load(open(f"{d}/det_run_{b}.json"))
    na = np.load(f"{d}/det_run_{a}.npz"); nb = np.load(f"{d}/det_run_{b}.npz")
    keys = sorted(na.files)
    ne_layers_v = sum(int(not np.array_equal(na[k], nb[k])) for k in keys if k.startswith("v"))
    ne_layers_b = sum(int(not np.array_equal(na[k], nb[k])) for k in keys if k.startswith("b"))
    maxv = max(float(np.abs(na[k] - nb[k]).max()) for k in keys if k.startswith("v")); maxb = max(float(np.abs(na[k] - nb[k]).max()) for k in keys if k.startswith("b"))
    out[pair] = {"answer_equal": ra["answer"] == rb["answer"], "first_step_equal": ra["first_step_sha1"] == rb["first_step_sha1"], "fold_all_equal": ra["fold_all_sha1"] == rb["fold_all_sha1"],
                 "layers_values_differ": ne_layers_v, "layers_bias_differ": ne_layers_b, "n_layers": sum(1 for k in keys if k.startswith("v")), "max_abs_values": maxv, "max_abs_bias": maxb,
                 "cfg_a": [ra["scatter"], ra["threads"], ra["det"]], "cfg_b": [rb["scatter"], rb["threads"], rb["det"]]}
print(json.dumps(out, indent=1))
