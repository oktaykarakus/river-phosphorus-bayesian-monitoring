"""Correction stage, amendment AM1 (VC_00 §4; a protocol amendment, NOT pre-declared): the Rao-Blackwell consistency statistic of
the frozen scorer (max over held-out sites of |mean conditional predictive mean - mean direct posterior draw of theta|) computed
on ALL posterior draws, threshold 0.05 unchanged. The z-summary of vf_02c is reported alongside (descriptive).
Usage: vc_03_rb_amended.py FOLD [original|refit] [N_DRAWS]   (N_DRAWS=400 gives the pre-declared 400-draw statistic)"""
import json
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import vc_lib as K  # noqa: E402

V = K.V
L = V.L
F, C, R, B = V.F, V.C, V.R, V.B


def main(fold, which="original", n_draws=None):
    import arviz as az
    meas, sites, sy = L.load_tables()
    held = V.heldout_sites(fold, sites)
    D = K.new_draws("vc_W1_R3") if which == "refit" else K.original_draws(f"vf_{fold}_R3")
    mc = R.model_config("R3")
    S0 = B.ndraws(D); n = S0 if n_draws is None else min(n_draws, S0)
    pk = np.linspace(0, S0 - 1, n).astype(int)
    Ds = {k: (np.asarray(v)[pk] if np.ndim(v) >= 1 and len(v) == S0 else v) for k, v in D.items()}
    hidx = np.flatnonzero(sites.site.isin(held).to_numpy())
    mu, sd = F.heldout_theta_conditional(Ds, sites, mc, hidx, C.rn_structures(sites))
    th = np.asarray(Ds["theta"], float)[:, hidx]
    d = mu - th; S = d.shape[0]
    ess = np.array([float(az.ess(d[:, i].reshape(4, S // 4))) for i in range(d.shape[1])])
    mcse = d.std(0, ddof=1) / np.sqrt(np.clip(ess, 10, S)); z = d.mean(0) / mcse
    mx = float(np.abs(d.mean(0)).max())
    res = dict(fold=fold, posterior=("vc_W1_R3 (refit)" if which == "refit" else f"vf_{fold}_R3 (Phase 4B)"), n_sites=int(d.shape[1]),
               n_draws=int(S), max_abs_mean_diff=mx, threshold=0.05, passes_threshold=bool(mx <= 0.05),
               median_abs_mean_diff=float(np.median(np.abs(d.mean(0)))), median_mcse=float(np.median(mcse)), max_mcse=float(mcse.max()),
               mean_z=float(z.mean()), sd_z=float(z.std(ddof=1)), share_abs_z_gt3=float((np.abs(z) > 3).mean()), max_abs_z=float(np.abs(z).max()),
               site_of_max_diff=str(np.asarray(sites.site)[hidx][np.abs(d.mean(0)).argmax()]))
    if S == 400 or n_draws == 400:
        res["status"] = "pre-declared statistic (400 draws)"; suffix = "_400draws_PREDECLARED"
    else:
        res["status"] = "AMENDED check AM1 (all draws; protocol amendment, not pre-declared)"; suffix = "_alldraws_AMENDED"
    tag = f"{fold}{'_refit' if which == 'refit' else ''}"
    K.save_json(res, K.VC / "diagnostics" / f"VC_rb_{tag}{suffix}.json")
    K.ledger(f"RB {res['status'].split(' (')[0]} {tag} ({S} draws)")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "original", int(sys.argv[3]) if len(sys.argv) > 3 else None)
