"""Post-hoc follow-up of the Rao-Blackwell (RB) check for R3 fold fits (VF_01 §4).

The pre-declared RB check (VF_00 §5.1) compares, for every held-out site, the posterior mean of its conditional predictive
mean with the mean of its direct posterior draws of theta; a disagreement > 0.05 was to stop scoring. The rule did not allow
for Monte Carlo error. With 400 draws and held-out posterior SDs of ~0.9, the MC SE of a site mean is ~0.05-0.07, so the
maximum over ~360 sites is expected near 0.15-0.2 even when the code is correct. This script separates a code error from
Monte Carlo noise:
  z_i = (mean_s mu_is - mean_s theta_is) / MCSE_i,  MCSE_i = sd of the per-draw differences / sqrt(ESS of that difference)
(ESS by arviz over the 4 chains, chain-major thinning as in the frozen scorer).
Declared reading (before computing): a correct conditional gives z approximately N(0,1): |mean z| < 0.2, sd(z) in [0.8, 1.25],
and no more than ~1% of |z| > 3. A systematic bias (|mean z| >= 0.2 or sd(z) > 1.25) is a code problem and stops the fold.
Output: diagnostics/VF_rb_check_{fold}.json (400 draws) or VF_rb_check_{fold}_{S}draws_EXPLORATORY.json
Usage: vf_02c_rb_check.py FOLD [N_DRAWS]
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import vf_lib as V  # noqa: E402

L = V.redirect()
F, C, R, B = V.F, V.C, V.R, V.B


def main(fold, n_draws=400):
    import arviz as az
    meas, sites, sy = L.load_tables()
    held = V.heldout_sites(fold, sites)
    D = V.load_draws(f"vf_{fold}_R3"); mc = R.model_config("R3")
    S0 = B.ndraws(D); pick = np.linspace(0, S0 - 1, min(n_draws, S0)).astype(int)
    Ds = {k: (np.asarray(v)[pick] if np.ndim(v) >= 1 and len(v) == S0 else v) for k, v in D.items()}
    hidx = np.flatnonzero(sites.site.isin(held).to_numpy())
    mu, sd = F.heldout_theta_conditional(Ds, sites, mc, hidx, C.rn_structures(sites))
    th = np.asarray(Ds["theta"], float)[:, hidx]
    d = mu - th                                                    # S x n
    S = d.shape[0]
    ess = np.array([float(az.ess(d[:, i].reshape(4, S // 4))) for i in range(d.shape[1])])
    mcse = d.std(0, ddof=1) / np.sqrt(np.clip(ess, 10, S))
    z = d.mean(0) / mcse
    res = dict(fold=fold, n_sites=int(d.shape[1]), n_draws=int(S), max_abs_mean_diff=float(np.abs(d.mean(0)).max()),
               median_mcse=float(np.median(mcse)), max_mcse=float(mcse.max()), mean_z=float(z.mean()), sd_z=float(z.std(ddof=1)),
               share_abs_z_gt3=float((np.abs(z) > 3).mean()), max_abs_z=float(np.abs(z).max()),
               site_of_max_diff=str(np.asarray(sites.site)[hidx][np.abs(d.mean(0)).argmax()]),
               z_of_max_diff=float(z[np.abs(d.mean(0)).argmax()]))
    res["code_problem"] = bool(abs(res["mean_z"]) >= 0.2 or res["sd_z"] > 1.25)
    suffix = "" if n_draws == 400 else f"_{S}draws_EXPLORATORY"
    res["status"] = "exploratory (not a pre-declared gate)"
    json.dump(res, open(V.VF / f"diagnostics/VF_rb_check_{fold}{suffix}.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 400)
