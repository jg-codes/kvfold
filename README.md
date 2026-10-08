# kvfold

Fold the entries that a KV-cache press discards back into the entries it keeps. `kvfold` is a training-free wrapper for any [kvpress](https://github.com/NVIDIA/kvpress) press.

This repository holds the wrapper, the runners and statistics scripts, and the per-item records behind Table 1 of the preprint *Folding discarded entries back into a compressed KV cache: evidence from reconstruction-scored presses on Qwen3-8B and Llama-3.1-8B-Instruct* (Johannes Gast, 2026).

## Use case

A press such as KVzip, KVzap or RestoreKV keeps a small share of the context entries and discards the rest. `MergingAnyPress` runs the press unchanged, so the kept entries and the compression ratio stay the same. It then sends each discarded entry to the kept entry with the most similar key (`MassFold`): the kept value becomes the press-score-weighted mean of the two, and the kept entry gets a log-mass bias on its attention logit. No training is needed and the question is not read at compression time. The bias goes through the attention mask, so the model has to run with `attn_implementation="sdpa"`.

Use it if you already run a kvpress press at high compression and want to know whether folding raises your task score at equal compression. The preprint tests effective compression ratios from 0.83 to 0.97 on RULER-4096 with Qwen3-8B (five operating points of three presses) and, with a smaller result, Llama-3.1-8B-Instruct. The limits below say where the evidence stops.

## Install

```
pip install "kvpress==0.5.5" "transformers>=4.56,<5"
pip install "kvfold @ git+https://github.com/jg-codes/kvfold"
```

kvfold is not on PyPI yet. Tested with kvpress 0.5.5, Python 3.12, PyTorch 2.14 (CPU) and transformers 4.57 (unit tests and the Table 1 rebuild); the exact CI versions are in `constraints/ci.txt`. transformers 5 is not supported yet: under transformers 5.2 one unit test that runs the kvpress KVComposePress fails during prefill.

## Use

```python
from transformers import pipeline
from kvpress import KVzipPress
from kvfold import MassFold, MergingAnyPress

pipe = pipeline("kv-press-text-generation", model="Qwen/Qwen3-8B", device="cuda",
                model_kwargs={"attn_implementation": "sdpa"})
inner = KVzipPress(compression_ratio=0.875)  # any kvpress press that scores every cache entry
press = MergingAnyPress(inner, compensation=MassFold(score_map="log"), n_sink=4)
print(pipe(context, question=question, press=press)["answer"])
```

`compensation="none"` reproduces the inner press bit for bit. The preprint uses `score_map="log"` for RestoreKV_plus and KVzip and `score_map="identity"` for KVzap; the exact constructor calls are in `runners/ruler/`.

## Before and after in one command

Table 1 of the preprint, rebuilt from the stored per-item records on a CPU in a few seconds (numpy and pandas only). "Bare press" is the before, "Fold" the after.

```
git clone https://github.com/jg-codes/kvfold && cd kvfold
python -m pip install numpy pandas
python scripts/reproduce_table1.py
```

Output:

```
Table 1. RULER-4096 | Qwen3-8B (bf16, sdpa) | containment score 0-100, pooled over 13 task means | higher is better | paired items at equal effective CR
| Press, operating point | Items | Published bare press | Bare press (this stack) | Fold | Fold - bare [95% CI] | Full cache | Gap closed |
|:--|--:|--:|--:|--:|--:|--:|--:|
| RestoreKV_plus, CR 0.9375 | 2,600 | 86.39 | 85.83 | 90.24 | +4.41 [+3.57, +5.30] | 95.21 | 0.47 |
| KVzap linear, threshold −3 | 6,500 | 82.47 | 82.53 | 87.46 | +4.93 [+4.24, +5.60] | 95.32 | 0.39 |
| KVzap MLP, threshold −3 | 6,500 | 93.54 | 93.62 | 94.01 | +0.40 [+0.03, +0.76] | 95.32 | 0.23 |
| KVzip, CR 0.875 | 6,500 | 92.24 | 92.21 | 93.92 | +1.71 [+1.32, +2.11] | 95.32 | 0.55 |
| KVzip, CR 0.97 | 1,300 | 19.58† | 18.83 | 25.59 | +6.76 [+5.47, +8.16] | 95.11 | 0.09 |
† published KVzip entry at CR 0.96875 (leaderboard label 0.97), scored on the same 1,300 items; no published row exists at CR 0.97.
All 6,500 items, RestoreKV_plus CR 0.9375: published 86.38, fold 89.97, difference +3.59
check RestoreKV_plus, CR 0.9375 OK
check KVzap linear, threshold −3 OK
check KVzap MLP, threshold −3 OK
check KVzip, CR 0.875 OK
check KVzip, CR 0.97 OK
check all 6,500 items OK
```

The script exits with status 1 if a printed number differs from Table 1 of the preprint at two decimals. The interval is a paired item bootstrap (B = 10,000, numpy `default_rng`; seeds and chunking are in the script header).

## Comparison with leaderboard entries

Benchmark RULER-4096 · model Qwen3-8B (bf16) · metric containment score 0-100 (the leaderboard's string-match score), pooled over 13 task means · higher is better. The leaderboard entries are bare presses (kvpress leaderboard, Hugging Face Space `nvidia/kvpress-leaderboard`, main revision e81fc403a5, read 8 October 2026). The fold is not a leaderboard entry.

| Leaderboard entry (label) | Reported score, all 6,500 items | Items in the paired run | Reported score, same items | Bare press, this stack | Fold | Fold − bare [95% CI] |
|:--|--:|--:|--:|--:|--:|--:|
| RestoreKV_plus (0.94) | 86.38 | 2,600 | 86.39 | 85.83 | 90.24 | +4.41 [+3.57, +5.30] |
| kvzap_linear (−3) | 82.47 | 6,500 | 82.47 | 82.53 | 87.46 | +4.93 [+4.24, +5.60] |
| kvzap_mlp (−3) | 93.54 | 6,500 | 93.54 | 93.62 | 94.01 | +0.40 [+0.03, +0.76] |
| kvzip (0.88) | 92.24 | 6,500 | 92.24 | 92.21 | 93.92 | +1.71 [+1.32, +2.11] |
| kvzip (0.97) | 19.85 | 1,300 | 19.58† | 18.83 | 25.59 | +6.76 [+5.47, +8.16] |
| no_press (full cache) | 95.32 | | | | | |

Reported = the published entry; "same items" rescored with the leaderboard's string-match rule on the items of the paired run. Bare press = the same press re-run here on sdpa, because the bias needs sdpa and the leaderboard runs used flash_attention_2. Fold = the same press inside `MergingAnyPress` with the same kept entries. † The leaderboard row labelled 0.97 has an effective ratio of 0.96875 and keeps 4% more entries than the run here (CR 0.97); no row exists at CR 0.97. The full cache is the leaderboard's no_press run.

On the same items, the bare re-run differed from the published predictions by +0.06, +0.08, −0.03 and −0.56 points (KVzap linear, KVzap MLP, KVzip 0.875, RestoreKV_plus). On all 6,500 items the fold over RestoreKV_plus scores 89.97 against 86.38 reported; the difference is +3.59 points [+3.04, +4.13] (paper, Section 4.1) and contains the offset between the two attention kernels.

## Stated limits

These come from Section 6 and the abstract of the preprint.

- **Containment score.** The RULER gains use a containment score. Exact match on question answering fell at three of the four KVzap and KVzip operating points (unchanged at KVzip CR 0.97) and against the published RestoreKV_plus row (Section 4.2). The read-out on the demotion stack of Section 4.6 (containment, exact match, token F1 and answer length) is in Note D8. Together with KVzip at CR 0.875 (−6.40 [−8.60, −4.20]), this is a consistent loss on strict question answering. Direction X (Section 7) targets it.
- **LongBench.** Folding every item gained +5.66 points on fresh RULER-8192 items but lost 3.03 points on fresh LongBench items, and exact match on RULER question answering fell. The fold lost on LongBench. The registered gate removed the mean loss but acts per task, and a coin flip would also have met its registered criteria (Section 4.4).
- **The gate.** A gate that reads only the context, registered before the blind test, gave +0.37 points [−0.46, +1.20] on LongBench, carried by one task (trec). It kept about half of the RULER-8192 gain (+3.06). It acts as a task-level switch (provisional): a task-blind random gate that folds 11% to 14.5% of the items would also meet its registered criteria.
- **Reproducibility.** The fold accumulates in float32 with a CUDA scatter-add whose order is not fixed, so repeated runs of the default path give different strings. A deterministic mode gave identical output in the tested setting only (Table D2, Note A3). The rows of this paper come from the default path. The `MassFold(scatter="sorted")` option sums in a fixed order and is bit-reproducible on CPU; it has not been run end to end on a GPU. The rows of the preprint come from the default path.
- **Scope.** The results use two 8B instruction-tuned models (Qwen3-8B and Llama-3.1-8B-Instruct) and presses that share one kind of score. On Llama, the pooled RULER gain comes from cwe and fwe, LongBench shows no net gain, the gate threshold (fitted on Qwen3-8B) does not transfer, and the mechanism is untested (Section 4.6). A smoke test with an offset of 0.01 in the scores fired its kill rule; a second one at offset 0 was not confirmed (Section 2). Other model families, sparse-attention models and other press families, for example EchoPress, remain untested.

## Per-item records

`data/per_item/*.csv.gz`: one file per operating point, columns `item_id, task, arm, score, predicted_answer`. `item_id` is the index into the 6,500 items of the RULER-4096 test split of `simonjegou/ruler` (13 tasks, 500 items each). Arms: `bare` and `fold` (this repository's runs; score and predicted answer), `published` and `full_cache` (leaderboard predictions rescored; score only). `data/per_item/README.md` lists files, row counts and checksums.

## Not included

- Benchmark contexts. The data card of `simonjegou/ruler` states no licence, so the contexts and gold answers are not redistributed. The runners load the dataset from the Hub at run time.
- RestoreKV checkpoints. They are licensed CC BY-NC 4.0 according to their model cards. Download them from the Hub: `higokri/RestoreKV-Qwen3-8B_plus` (used for RestoreKV_plus on Qwen3-8B) and `higokri/RestoreKV-Llama-3.1-8B-Instruct_plus`. Non-commercial use only.
- Leaderboard prediction files. The runners take them through `--leaderboard` (download the entry's `predictions.csv` from the leaderboard Space).

## Layout

```
src/kvfold/              the wrapper (MergingAnyPress, MassFold, scatter options)
tests/                   unit tests (pytest)
scripts/reproduce_table1.py
data/per_item/           per-item records (csv.gz)
data/expected_table1.json, data/leaderboard_reported_scores.csv
runners/ruler/           reference copies of the RULER-4096 runners
runners/determinism/     CPU and GPU determinism checks
stats/                   statistics scripts of the registered tests
```

`runners/README.md` says what the runners need and what was changed in the copies.

## Cite

```
@misc{gast2026fold,
  title  = {Folding discarded entries back into a compressed KV cache: evidence from reconstruction-scored presses on Qwen3-8B and Llama-3.1-8B-Instruct},
  author = {Gast, Johannes},
  year   = {2026},
  note   = {Preprint, 8 October 2026. The arXiv identifier is added after submission.}
}
```

`CITATION.cff` has the same entry. If you use kvpress, cite it through its own `CITATION.cff`.

The code is archived on Zenodo. [10.5281/zenodo.23232929](https://doi.org/10.5281/zenodo.23232929) always resolves to the latest release; [10.5281/zenodo.23232930](https://doi.org/10.5281/zenodo.23232930) is v0.1.1.

## Licence

Apache-2.0 (see `LICENSE` and `NOTICE`). The preprint text is CC BY 4.0. RestoreKV checkpoints and the RULER data keep their own terms.
