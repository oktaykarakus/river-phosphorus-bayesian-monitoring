"""Phase 3C-F: run the R3 program with the prepared pilot metric and over-dispersed inits (p3f_02_prepare_inputs.py).
Usage: p3f_run_r3.py --tag TAG --iter_warmup W --iter_sampling S [--save_warmup] [--no_summary]"""
import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import p3f_lib as L  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True); ap.add_argument("--iter_warmup", type=int, required=True); ap.add_argument("--iter_sampling", type=int, required=True)
ap.add_argument("--adapt_delta", type=float, default=0.98); ap.add_argument("--save_warmup", action="store_true"); ap.add_argument("--no_summary", action="store_true")
a = ap.parse_args()
inp = L.scratch_dir() / "inputs"
data = json.load(open(inp / "data.json"))
info = json.load(open(L.ROOT / "diagnostics" / "main_inputs.json"))
inits = [str(inp / f"init_{k}.json") for k in range(1, 5)]
metric = json.load(open(inp / "inv_metric.json"))["inv_metric"]
mc = L.R.model_config("R3")
out = L.run_fit(a.tag, data, mc, iter_warmup=a.iter_warmup, iter_sampling=a.iter_sampling, chains=4, adapt_delta=a.adapt_delta, inits=inits,
                inv_metric=metric, step_size=info["initial_step"], save_warmup=a.save_warmup, summarise=not a.no_summary,
                extra=dict(inits="over-dispersed along (share_u, share_B, log rho_tu); see diagnostics/main_inputs.json",
                           metric="pilot: geometric mean of the 4 Phase 3C-R A5 adapted metrics; full windowed adaptation engaged"))
if not a.no_summary:
    res = out[0]
    print(json.dumps({k: res[k] for k in ["cpu_h", "wall_s", "divergences", "hyper_max_rhat", "hyper_argmax_rhat", "hyper_min_ess_bulk", "hyper_argmin_ess_bulk",
                                          "hyper_min_ess_tail", "ebfmi", "stepsize_by_chain", "mean_leapfrog", "latent_max_rhat"]}, indent=1))
