"""Phase 3C-F: inputs for the feasibility benchmark and the main R3 fit (no sampling here).

* Pilot metric: geometric mean over the 4 Phase 3C-R A5 chains of their ADAPTED diagonal inverse metrics (same program, same data,
  same unconstrained order; 300 adaptive warmup iterations each). Initial step size = median adapted A5 step size.
* Over-dispersed initial values along the weakly identified directions: from the Phase 3C-R R1 full-data screen (same model, 4,000
  draws), the draw nearest (in rank space) to each of four target corners of (share_u, share_B, log rho_tu):
  (.05, .95, .05), (.95, .05, .95), (.05, .05, .95), (.95, .95, .05). Mapped to R3 coordinates exactly as in Phase 3C-R (p3r_07).
Outputs (scratch): inputs/data.json, inputs/init_{1..4}.json, inputs/inv_metric.json; diagnostics/main_inputs.json
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import p3f_lib as L  # noqa: E402

A5DIR = Path(L.CFG["phase3cr_scratch"]).expanduser() / "fit_full_R3w"
SHARED = ["alpha", "beta", "gamma", "harm", "delta_bar", "z_seas", "tau_s", "theta", "sigma_om", "z_om", "sigma", "lambda", "tau_sig", "z_sig",
          "sigma_A", "sigma_C", "z_A", "z_C", "sigma_w", "rho", "z_w", "sigma_e", "z_e", "rho_tu"]
TARGETS = [(.05, .95, .05), (.95, .05, .95), (.05, .05, .95), (.95, .95, .05)]


def rank01(x):
    return (np.argsort(np.argsort(x)) + 0.5) / len(x)


def main():
    import cmdstanpy
    out = L.scratch_dir() / "inputs"; out.mkdir(exist_ok=True)
    meas, sites, sy, mc, data = L.r3_data()
    cmdstanpy.write_stan_json(str(out / "data.json"), data)
    # pilot metric
    mets, steps = [], []
    for f in sorted(A5DIR.glob("track_a_m2rn_r3-*_[1-4].csv")):
        lines = open(f).read().splitlines()
        for i, ln in enumerate(lines):
            if ln.startswith("# Step size ="):
                steps.append(float(ln.split("=")[1]))
            if ln.startswith("# Diagonal elements of inverse mass matrix:"):
                mets.append(np.array([float(x) for x in lines[i + 1].lstrip("# ").split(",")]))
                break
    M = np.vstack(mets); gm = np.exp(np.log(M).mean(axis=0))
    json.dump({"inv_metric": gm.tolist()}, open(out / "inv_metric.json", "w"))
    # dispersed inits
    D = L.R.load_draws("screen_R1")
    su, sb, stu = L.B.get1(D, "sigma_u"), L.B.get1(D, "sigma_B"), L.B.get1(D, "sigma_tu")
    st = np.sqrt(su ** 2 + sb ** 2 + stu ** 2); sh = np.vstack([su ** 2, sb ** 2, stu ** 2]).T / st[:, None] ** 2
    rho_tu = L.B.get1(D, "rho_tu")
    Rk = np.vstack([rank01(sh[:, 0]), rank01(sh[:, 1]), rank01(np.log(rho_tu))]).T
    picks, rec = [], []
    for k, t in enumerate(TARGETS):
        s = int(np.argmin(((Rk - np.array(t)) ** 2).sum(axis=1)))
        picks.append(s)
        it = {key: (np.asarray(D[key])[s].tolist() if np.ndim(np.asarray(D[key])[s]) else float(np.asarray(D[key])[s])) for key in SHARED}
        it.update(sigma_site=float(st[s]), share=sh[s].tolist())
        json.dump(it, open(out / f"init_{k + 1}.json", "w"))
        rec.append(dict(chain=k + 1, target_rank=t, achieved_rank=Rk[s].round(3).tolist(), draw=s, sigma_site=float(st[s]), share=sh[s].round(4).tolist(),
                        sigma_u=float(su[s]), sigma_B=float(sb[s]), sigma_tu=float(stu[s]), rho_tu=float(rho_tu[s]), lp_rank=float(rank01(D["lp__"])[s])))
    info = dict(n_unconstrained=int(len(gm)), pilot_metric_from=[str(f.name) for f in sorted(A5DIR.glob("track_a_m2rn_r3-*_[1-4].csv"))],
                a5_adapted_steps=steps, initial_step=float(np.median(steps)), inits=rec,
                posterior_quantiles_R1_screen=dict(share_u=np.quantile(sh[:, 0], [.05, .5, .95]).round(4).tolist(),
                                                   share_B=np.quantile(sh[:, 1], [.05, .5, .95]).round(4).tolist(),
                                                   rho_tu=np.quantile(rho_tu, [.05, .5, .95]).round(1).tolist()))
    L.save_json(info, L.ROOT / "diagnostics" / "main_inputs.json")
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
