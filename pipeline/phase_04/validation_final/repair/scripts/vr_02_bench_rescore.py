"""Track A validation repair, VR_02: cost benchmark and exactness check for a LEAN site-joint rescoring (no fitting).

Recomputes lp[i, s] = site-joint log predictive density for every held-out site of one fold at a few of the frozen scoring
draws, using the frozen Phase 3C functions exactly as p3c_folds.heldout_scores does (only the site-joint part; pointwise,
PIT and site-year replicate scores are not needed for the ELPD), compares with the stored per-draw values and times it.
Usage: vr_02_bench_rescore.py FOLD [n_bench_draws=16]. Writes repair/diagnostics/VR_bench_{FOLD}.json."""
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
VR = Path(__file__).resolve().parents[1]
VFDIR = VR.parent
sys.path.insert(0, str(VFDIR / "scripts"))
import vf_lib as V  # noqa: E402

F, C, R, B = V.F, V.C, V.R, V.B
ORIG = Path(os.path.expanduser("~/track_a_scratch/phase_04_vf"))


def site_lp(D, meas, sites, sy, mc, held, pick, rn=None):
    """Lines of p3c_folds.heldout_scores that produce the per-draw site-joint log density, unchanged.
    rn (river-network structures) may be precomputed once by the caller; it depends on the sites only."""
    S0 = B.ndraws(D)
    Ds = {k: (np.asarray(v)[pick] if np.ndim(v) >= 1 and len(v) == S0 else v) for k, v in D.items()}
    held_idx = np.flatnonzero(sites.site.isin(held).to_numpy())
    if rn is None and mc.get("tu", 0):
        rn = C.rn_structures(sites)
    if not mc.get("tu", 0):
        rn = None
    mu, sd = F.heldout_theta_conditional(Ds, sites, mc, held_idx, rn)
    hm = meas[meas.site.isin(held)].copy()
    eta_all = B.eta_obs(Ds, hm, sites, sy, mc)
    th = np.asarray(Ds["theta"], float); om = B.get1(Ds, "sigma_om")[:, None] * np.asarray(Ds["z_om"], float)
    seas_all = eta_all - th[:, hm.site_idx.to_numpy() - 1] - om[:, hm.siteyear_idx.to_numpy() - 1]
    s_om = B.get1(Ds, "sigma_om"); lam = B.get1(Ds, "lambda"); tsg = B.get1(Ds, "tau_sig"); lsig = np.log(B.get1(Ds, "sigma"))
    zt = sites.z_logarea.to_numpy()
    out, names = [], []
    for k, i in enumerate(held_idx):
        kk = (hm.site_idx.to_numpy() - 1) == i
        if not kk.any():
            continue
        sub = hm[kk]
        yr = pd.factorize(sub.siteyear_idx.to_numpy())[0]
        out.append(F.site_joint_logpred(mu[:, k], sd[:, k], s_om, lsig + lam * zt[i], tsg, sub.log_y.to_numpy(),
                                        sub.censored.to_numpy(), sub.log_limit.to_numpy(), seas_all[:, kk], yr))
        names.append(sites.site.iloc[i])
    return np.vstack(out), np.array(names)


def cpu():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


def main(fold, nb):
    meas, sites, sy = V.L.load_tables()
    held = V.heldout_sites(fold, sites)
    res = dict(fold=fold, n_bench_draws=nb)
    for model in ["m2rn", "m1s2"]:
        D = V.comparator_draws(fold) if model == "m1s2" else V.load_draws(f"vf_{fold}_R3")
        mc = C.model_config(tu=0) if model == "m1s2" else R.model_config("R3")
        S0 = B.ndraws(D)
        pick400 = np.linspace(0, S0 - 1, 400).astype(int)
        j = np.linspace(0, 399, nb).astype(int)                       # spread over all four chains
        z = np.load(ORIG / f"score_lp_{fold}_{model}.npz")
        c0, w0 = cpu(), time.time()
        lp, names = site_lp(D, meas, sites, sy, mc, held, pick400[j])
        c1, w1 = cpu(), time.time()
        assert list(names) == list(z["site"].astype(str))
        dev = float(np.abs(lp - z["lp"][:, j]).max())
        per_draw = (c1 - c0) / nb
        res[model] = dict(S_all=int(S0), max_abs_dev_vs_stored=dev, cpu_s=c1 - c0, wall_s=w1 - w0, cpu_s_per_draw=per_draw,
                          projected_cpu_h_all_draws=per_draw * S0 / 3600, projected_cpu_h_1600=per_draw * min(1600, S0) / 3600)
        print(model, json.dumps(res[model]), flush=True)
    V.save_json(res, VR / "diagnostics" / f"VR_bench_{fold}.json")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 16)
