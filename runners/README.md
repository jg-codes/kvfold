# Runners and statistics scripts (reference copies)

These scripts produced the per-item rows behind the preprint. They are the record of how the numbers were made, not a packaged tool.

- The GPU runs used a single NVIDIA A100 (40 GB), bf16, `attn_implementation="sdpa"`, greedy decoding, on a research build of kvpress 0.5.5. That build adds the RestoreKV press and a fitted-compensation module to upstream kvpress 0.5.5. Neither is part of upstream kvpress or of this repository, so the RestoreKV and fitted arms of the runners do not import against upstream kvpress alone. The wrapper itself (`kvfold.merging_any_press`) is the file the runs used, plus the optional `scatter` setting and a compatibility import for `get_query_states`, which upstream kvpress 0.5.5 does not define (the fallback computes the same expression from `get_prerope_query_states`).
- RULER-4096 contexts are loaded at run time from the Hugging Face dataset `simonjegou/ruler`. They are not in this repository.
- `--leaderboard` takes the `predictions.csv` of the leaderboard entry that the run compares with. Download it from the leaderboard Space (file names as in the defaults, for example `leaderboard/RestoreKV_plus__0.94__predictions.csv`).
- The runners import the wrapper as `kvpress.presses.merging_any_press`, as in the research build. With kvfold, alias it first: `import sys, kvfold.merging_any_press as m; sys.modules["kvpress.presses.merging_any_press"] = m`.
- Differences between the copies and the files as used: two path defaults that pointed at a storage volume of the original compute host now point at `leaderboard/`; the cost-guard defaults (a budget cap in EUR) are `inf`; one comment and one docstring line no longer name the host or the research repository. Nothing else changed. The table lists the first 16 hex digits of the SHA-256 of the file as used and of the copy.

| File | SHA-256 as used | SHA-256 of copy | Status | Purpose (first docstring line) |
|:--|:--|:--|:--|:--|
| `runners/ruler/control_folds.py` | `946da09af5be1149` | `d6699b3b6bf47dcf` | changed | Control compensations for the C86 mass fold (pre-registered in hypothesis_C113.json and in revision 1.1 of |
| `runners/ruler/item_ids_c127_fullset.json` | `29ae443ce0e65a7b` | `29ae443ce0e65a7b` | same |  |
| `runners/ruler/run_kvzap_smoke.py` | `774fbba88b955000` | `fa5610029f7c33e9` | changed | KVzap smoke (wiki claim C111, pre-registration hypothesis_C111.json): MergingAnyPress over the leaderboard KVz |
| `runners/ruler/run_restorekv_confirm.py` | `3451e8620a09ed31` | `7193dc084db12705` | changed | RestoreKV_plus mass-fold CONFIRMATION of C113 (wiki claim C115, filed first; pre-registration hypothesis_C115. |
| `runners/ruler/sig_launcher.py` | `7a039e4da634fb99` | `7a039e4da634fb99` | same | sig_launcher.py -- runs the frozen runner run_kvzap_smoke.py UNCHANGED, optionally under deterministic CUDA al |
| `runners/ruler/sig_launcher_kvzip.py` | `d6e31dd125197802` | `d6e31dd125197802` | same | sig_launcher_kvzip.py -- runs the KVzip derivative of the frozen runner sig_run_kvzip.py UNCHANGED, optionally |
| `runners/determinism/det_compare.py` | `64b4d02261a2649e` | `64b4d02261a2649e` | same | det_compare.py - pairwise comparison of det_run_<tag>.json/.npz.  usage: python det_compare.py DIR tagA:tagB [ |
| `runners/determinism/det_cpu_test.py` | `019c8be97cd98707` | `019c8be97cd98707` | same | det_cpu_test.py - operator-level bit-identity tests of the fold's scatter (CPU).  usage: python det_cpu_test.p |
| `runners/determinism/det_gpu_microbench.py` | `3332b26a5d76b5c9` | `3332b26a5d76b5c9` | same | det_gpu_microbench.py - the fold's scatter variants on the GPU: run-to-run identity and cost. usage: python de |
| `runners/determinism/det_launcher.py` | `cb2deadea0b1bcd3` | `cb2deadea0b1bcd3` | same | det_launcher.py - runs a frozen runner UNCHANGED under deterministic CUDA algorithms and records the setting. |
| `runners/determinism/det_model_run.py` | `0b520c37343bfa6f` | `0b520c37343bfa6f` | same | det_model_run.py - one end-to-end fold run on a small model on CPU (separate process per run). |
| `runners/determinism/det_scatter_lib.py` | `84dcb1b863a0f8ee` | `84dcb1b863a0f8ee` | same | Deterministic scatter for the mass fold: fixed-order segment sums. |
| `runners/determinism/item_ids_det104.json` | `6112c78ebbef2e0b` | `6112c78ebbef2e0b` | same |  |
| `stats/build_paired_c127.py` | `c1bdadd26abf8bb2` | `c1bdadd26abf8bb2` | same | build_paired_c127.py - builds the paired 6500-id table D (C127 arm-C rows on 3,900 new ids, C115 arm-C rows on |
| `stats/sig_build_stats.py` | `d844053ba6cace6b` | `d844053ba6cace6b` | same | sig_build_stats.py rev 1 -- statistics of the RULER-4096 full-set significance run (wiki claim C141), written  |
| `stats/sig_registered_params.json` | `02754c2544535be1` | `02754c2544535be1` | same |  |

Statistics scripts: `stats/sig_build_stats.py` computes the registered statistics of the KVzap and KVzip rows (parameters in `stats/sig_registered_params.json`); `stats/build_paired_c127.py` builds the paired file of the RestoreKV_plus full-set run. `scripts/reproduce_table1.py` is the short path from the stored per-item records to Table 1.
