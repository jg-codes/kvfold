#!/usr/bin/env python
"""det_launcher.py - runs a frozen runner UNCHANGED under deterministic CUDA algorithms and records the setting.

usage: python det_launcher.py --det on|off -- <runner.py> <runner arguments>      (env DET_META = path of the meta json, optional)
  --det on : CUBLAS_WORKSPACE_CONFIG=:4096:8 (set before CUDA initialisation), torch.use_deterministic_algorithms(True, warn_only=True),
             cudnn.deterministic=True, cudnn.benchmark=False.
  --det off: nothing is changed.
The runner is executed with runpy.run_path(run_name='__main__'), so its own source-hash bookkeeping is unchanged.
Reason (conventions section 6): MassFold uses float32 scatter_add_ on CUDA, which is not bit-reproducible (A24 violated, P33)."""
import atexit, json, os, runpy, sys, warnings
args = sys.argv[1:]
assert len(args) >= 4 and args[0] == "--det" and args[1] in ("on", "off") and args[2] == "--", f"usage error: {args[:3]}"
det = args[1] == "on"; runner = args[3]; rest = args[4:]
if det:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
msgs = set()
_orig_show = warnings.showwarning
def _show(message, category, filename, lineno, file=None, line=None):
    m = str(message)
    if "determin" in m.lower():
        msgs.add(m[:240])
    return _orig_show(message, category, filename, lineno, file, line)
warnings.showwarning = _show
if det:
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    assert torch.are_deterministic_algorithms_enabled()
    print(f"launcher: deterministic algorithms ON (warn_only=True), CUBLAS_WORKSPACE_CONFIG={os.environ['CUBLAS_WORKSPACE_CONFIG']}", flush=True)
else:
    print("launcher: deterministic algorithms OFF", flush=True)
def write_meta():
    p = os.environ.get("DET_META")
    if not p:
        return
    meta = {"det": det, "deterministic_enabled": bool(torch.are_deterministic_algorithms_enabled()), "warn_only": bool(torch.is_deterministic_algorithms_warn_only_enabled()),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"), "torch": torch.__version__, "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "runner": runner, "runner_args": rest, "deterministic_warnings": sorted(msgs)}
    json.dump(meta, open(p, "w"), indent=1)
atexit.register(write_meta)
sys.argv = [runner] + rest
runpy.run_path(runner, run_name="__main__")
