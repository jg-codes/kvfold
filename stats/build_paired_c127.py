#!/usr/bin/env python
"""build_paired_c127.py - builds the paired 6500-id table D (C127 arm-C rows on 3,900 new ids, C115 arm-C rows on 2,600 ids, published predictions,
bare-arm A and C0 rows on the C115 ids) used by refutation_c127_c135.py. Inputs: rows_c127_job9.jsonl (ve0edc708), rows_c115_final.jsonl (v58135468),
item_ids_c127_fullset.json (502ede63), ruler_golds.json (ee81e9fe), ext_ruler8k_rows.jsonl (v94dc999c)."""
import json, pandas as pd, numpy as np
def load(paths):
    jl = lambda p: [json.loads(l) for l in open(p)]
    R = pd.DataFrame(jl(paths["c127"])); C = pd.DataFrame(jl(paths["c115"])); E = pd.DataFrame(jl(paths["ext"]))
    it = json.load(open(paths["items"])); g = json.load(open(paths["golds"]))
    for d in (R, C): d["iid"] = d.item_id.astype(int); d["sc"] = d.score.astype(float); d["lbs"] = d.lb_score.astype(float)
    Cc = C[C.arm == "C"].set_index("iid"); Ca = C[C.arm == "A"].set_index("iid"); C0 = C[C.arm == "C0"].set_index("iid"); Rc = R[R.arm == "C"].set_index("iid")
    c115 = set(Cc.index); new = set(it["new_ids"]); used = set(it["partition_ids"]["used_1912"]); assert len(c115 | new) == 6500
    rows = []
    for i in sorted(c115 | new):
        r = (Cc if i in c115 else Rc).loc[i]
        rows.append(dict(iid=i, task=g["task"][i], cpred=r.predicted_answer, cs=r.sc, pub=r.lb_predicted_answer, pubs=r.lbs,
                         part=("c115" if i in c115 else ("used" if i in used else "never")), ctx=int(r.ctx_len),
                         apred=(Ca.loc[i].predicted_answer if i in c115 else None), as_=(Ca.loc[i].sc if i in c115 else np.nan),
                         c0pred=(C0.loc[i].predicted_answer if i in C0.index else None), c0s=(C0.loc[i].sc if i in C0.index else np.nan)))
    D = pd.DataFrame(rows)
    OV = pd.DataFrame([dict(iid=i, task=g["task"][i], p115=Cc.loc[i].predicted_answer, p127=Rc.loc[i].predicted_answer) for i in sorted(it["overlap_ids"])])
    return D, g, E, OV
