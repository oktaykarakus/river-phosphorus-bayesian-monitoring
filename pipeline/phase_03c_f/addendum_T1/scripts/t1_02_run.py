"""T1 step 3: the new sampling segment (settings: addendum_T1/config/t1_settings.yaml).
Per chain k: init = main_R3 final draw of chain k, chain k's adapted inverse metric and step size, warmup 0, adaptation off,
2,500 draws, seed 20261123. Then summarised with the Phase 3C-F summariser, redirected to the addendum
(tables/T1_segment_posterior_summary.csv, diagnostics/T1_segment_sampling_diagnostics.json, scratch T1_segment_draws.npz).
"""
import json
import resource
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import t1_lib as T  # noqa: E402

L = T.redirect()                      # every Phase 3C-F writer now writes into addendum_T1
C = T.T1CFG["continuation"]


def main():
    sc = T.scratch(); inp = sc / "inputs_T1"; od = sc / "fit_T1_segment"
    if od.exists():
        raise SystemExit(f"{od} exists: refusing to overwrite")
    od.mkdir()
    data = json.load(open(sc / "inputs" / "data.json"))
    steps = json.load(open(inp / "stepsize.json"))["stepsize"]
    model = L.get_model()
    r0 = resource.getrusage(resource.RUSAGE_CHILDREN); t0 = time.time()
    status = "completed"
    try:
        fit = model.sample(data=data, chains=4, parallel_chains=4, seed=C["seed"], iter_warmup=0, iter_sampling=C["iter_sampling"], adapt_engaged=False,
                           metric="diag_e", inv_metric=[str(inp / f"inv_metric_{k}.json") for k in range(1, 5)], step_size=steps,
                           inits=[str(inp / f"init_{k}.json") for k in range(1, 5)], max_treedepth=C["max_treedepth"], sig_figs=C["sig_figs"],
                           output_dir=str(od), show_progress=False, refresh=25)
    except Exception as e:  # stopped by the monitor (budget) or failed
        fit = None; status = f"not completed: {type(e).__name__}: {str(e)[:300]}"
    wall = time.time() - t0; r1 = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu = (r1.ru_utime - r0.ru_utime) + (r1.ru_stime - r0.ru_stime)
    T.save_json(dict(status=status, cpu_s=cpu, cpu_h=cpu / 3600, wall_s=wall), T.ADD / "diagnostics" / "T1_segment_run_status.json")
    print(status, round(cpu / 3600, 3), "CPU-h", round(wall / 60, 1), "min wall", flush=True)
    if fit is None:
        return
    mc = L.R.model_config("R3")
    res, _, _ = L.summarise_fit(fit, T.SEG_TAG, data, mc, cpu, wall, 0, C["iter_sampling"], 4, None, C["max_treedepth"], C["seed"],
                                extra=dict(kind=C["kind"], adapt_engaged=False, inits="main_R3 chain endpoints (inputs_T1)",
                                           metric="chain-specific adapted main_R3 metrics, fixed", step_size=steps))
    print(json.dumps({k: res[k] for k in ["cpu_h", "divergences", "hyper_max_rhat", "hyper_argmax_rhat", "hyper_min_ess_bulk", "hyper_min_ess_tail",
                                          "ebfmi", "stepsize_by_chain", "mean_leapfrog", "latent_max_rhat"]}, indent=1))


if __name__ == "__main__":
    main()
