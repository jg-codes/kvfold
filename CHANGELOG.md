# Changelog

## 0.1.1 (2026-10-08)

- Pin transformers below 5 in `pyproject.toml`, and pin the tested versions in CI (`constraints/ci.txt`). Under transformers 5.2.0 the unit test `test_stale_bias_does_not_leak_into_a_later_run`, which runs the kvpress KVComposePress, failed during prefill on the GitHub runner.
- Unit tests now also run on every push to `main` and on pull requests.
- `.zenodo.json` gives the license as `apache-2.0`, the lower-case SPDX id that Zenodo's license list uses.
- No change to the fold wrapper, the runners, the statistics scripts or the per-item records.

## 0.1.0 (2026-10-08)

- First release: fold wrapper, reference runners, statistics scripts and per-item records of the preprint. It was published before the Zenodo integration was switched on, so it has no Zenodo DOI; use 0.1.1.
