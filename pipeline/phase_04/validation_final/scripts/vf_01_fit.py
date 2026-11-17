"""Phase 4B: run one approved validation fit (config/vf_settings.yaml). Usage: vf_01_fit.py --tag TAG"""
import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import vf_lib as V  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--tag", required=True); a = ap.parse_args()
assert a.tag in V.CFG["fit_caps_cpu_h"], "not an approved fit"
res, draws = V.run_fit(a.tag)
print(json.dumps({k: res.get(k) for k in ["tag", "cpu_h", "wall_s", "divergences", "hyper_max_rhat", "hyper_argmax_rhat", "hyper_min_ess_bulk",
                                          "hyper_argmin_ess_bulk", "ebfmi", "stepsize_by_chain", "mean_leapfrog", "latent_max_rhat", "log_scale_diagnostics"]},
                 indent=1, default=str), flush=True)
