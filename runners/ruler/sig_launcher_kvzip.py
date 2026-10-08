#!/usr/bin/env python
"""sig_launcher_kvzip.py -- runs the KVzip derivative of the frozen runner sig_run_kvzip.py UNCHANGED, optionally under deterministic CUDA algorithms.

usage: python sig_launcher.py --det on|off -- <runner arguments>
  --det on : CUBLAS_WORKSPACE_CONFIG=:4096:8 (set before CUDA initialisation) and torch.use_deterministic_algorithms(True, warn_only=True)
  --det off: nothing is changed (the runner behaves exactly as in the C111 smoke)
The runner file is executed with runpy.run_path(run_name='__main__'), so its own hash bookkeeping (SRC_SHA, argv) is unchanged.
Reason (conventions s6): MassFold uses float32 scatter_add_ on CUDA, which is not bit-reproducible (A24 violated, P33).
"""
import os
import runpy
import sys

args = sys.argv[1:]
assert len(args) >= 3 and args[0] == "--det" and args[1] in ("on", "off") and args[2] == "--", f"usage error: {args[:3]}"
det = args[1] == "on"
rest = args[3:]
if det:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch  # noqa: E402

if det:
    torch.use_deterministic_algorithms(True, warn_only=True)
    assert torch.are_deterministic_algorithms_enabled()
    print(f"launcher: deterministic algorithms ON (warn_only=True), CUBLAS_WORKSPACE_CONFIG={os.environ['CUBLAS_WORKSPACE_CONFIG']}", flush=True)
else:
    print("launcher: deterministic algorithms OFF", flush=True)
sys.argv = ["sig_run_kvzip.py"] + rest
runpy.run_path("sig_run_kvzip.py", run_name="__main__")
