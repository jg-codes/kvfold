# Per-item records

RULER-4096 test split of `simonjegou/ruler` (13 tasks x 500 items, in dataset order), Qwen3-8B (bf16, sdpa), greedy decoding at the dataset's answer length.

Columns: `item_id` (index into the 6,500 items), `task`, `arm`, `score` (containment score 0-100: for qa_1 and qa_2 the share of 0 or 1 for any gold string contained in the answer, for the other tasks the share of gold strings contained, case-insensitive, as in the leaderboard's string-match metric), `predicted_answer`.

Arms: `bare` (the press, this stack), `fold` (the same press inside `MergingAnyPress`, same kept entries), `published` (the leaderboard entry's predictions, rescored; score only), `full_cache` (the leaderboard's no_press predictions, rescored; score only). Gold answers and contexts are not included. Row order within an arm is the order in which the paired bootstrap resamples; `scripts/reproduce_table1.py` relies on it for the RestoreKV_plus row (seed 0).

| File | Rows | Rows per arm | Bytes | SHA-256 (16 hex) |
|:--|--:|:--|--:|:--|
| `ruler4096_qwen3-8b__full_cache.csv.gz` | 6,500 | full_cache 6,500 | 17,250 | `d755f31236f630af` |
| `ruler4096_qwen3-8b__kvzap_linear_thr-3.csv.gz` | 19,500 | bare 6,500, fold 6,500, published 6,500 | 439,768 | `2f6b30bd28e8cd1d` |
| `ruler4096_qwen3-8b__kvzap_mlp_thr-3.csv.gz` | 19,500 | bare 6,500, fold 6,500, published 6,500 | 382,661 | `7d416a32db9171da` |
| `ruler4096_qwen3-8b__kvzip_cr0.875.csv.gz` | 19,500 | bare 6,500, fold 6,500, published 6,500 | 367,714 | `555a6e4905db3650` |
| `ruler4096_qwen3-8b__kvzip_cr0.97.csv.gz` | 3,900 | bare 1,300, fold 1,300, published 1,300 | 104,953 | `322bac58459aeb52` |
| `ruler4096_qwen3-8b__restorekv_plus_cr0.9375.csv.gz` | 7,800 | bare 2,600, fold 2,600, published 2,600 | 195,390 | `b05edf862523cafd` |
| `ruler4096_qwen3-8b__restorekv_plus_cr0.9375_all6500.csv.gz` | 13,000 | fold 6,500, published 6,500 | 215,899 | `ee90e9807a438b8a` |

Sources of the rows: `restorekv_plus_cr0.9375` holds the 2,600 fresh items of the registered RestoreKV_plus run (record C115); `restorekv_plus_cr0.9375_all6500` holds the fold arm of all 6,500 items (C115 rows for the 2,600 fresh items, C127 rows for the other 3,900) and the published scores; the KVzap and KVzip files hold the paired arms of the registered runs of the same names (records C141, KVzip CR 0.875 completed to 6,500 items after the registered stop). `published` in `kvzip_cr0.97` is the leaderboard entry labelled 0.97 (effective CR 0.96875), scored on the 1,300 items of the run. The scorer rescored every `predicted_answer` and reproduced the stored scores exactly; for the RestoreKV_plus rows it applies the corrected gold-string parsing described in the preprint.

`../expected_table1.json` holds the numbers printed in Table 1 of the preprint and the two full-set numbers; `../leaderboard_reported_scores.csv` holds the leaderboard's reported pooled scores for the entries used (main revision e81fc403a5).
