"""Phase 4 A2: selection sensitivity on the authoritative posterior (no refit), as frozen in P2B_04 §3 and P4-V0 §3 A2.

For each posterior draw and each T unit, the log level L (fixed + location terms incl. river term + gamma_t + log seasonal
factor) is computed exactly as in the frozen Delta_T code (phase_03c/scripts/p3c_lib.py::delta_T); new site and site-year
effects are integrated exactly, the heteroscedastic SD by 15-node Gauss-Hermite. Then
  pi(shift) = E_gh Phi((L + shift + s^2/2 - log c) / sd_re)
  Delta_T             = mean_T [pi(delta) - pi(0)]                                  (unchanged by psi)
  P^R_T(psi)          = mean_T pi(-psi)                                              (RSN-level exceedance of comparable reaches)
  Delta^S_T(psi,kappa)= mean_T [pi(-psi + kappa (delta + psi)) - pi(-psi)]          (siting contrast net of measurement share)
  s^M = kappa (delta + psi); s^M / sigma_site
  A0: Delta_T and P^k_T without the short-term variance term (s^2/2 dropped).
Outputs: tables/A2_selection_grid.csv, tables/A2_PR_by_psi.csv, tables/A2_A0.csv, diagnostics/A2_selection.json
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtr

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import p4_lib as P  # noqa: E402

L = P.redirect()
B, C = L.B, L.C
A = B.A if hasattr(B, "A") else None


def setup(D, sites, sy, mc, n_gh):
    comp = C.site_components(D, sites, mc)
    Tm = sy.in_target_T.to_numpy() == 1
    sidx = sy.site_idx.to_numpy()[Tm] - 1; yrt = (sy.year_idx.to_numpy()[Tm] == 2).astype(float)
    w = sy.target_weight.to_numpy()[Tm]; w = w / w.sum()
    loc = comp["vA"] + comp["vC"] + comp["vB"] + comp["w"] + comp["rn"]
    base = comp["fixed"][:, sidx] + loc[:, sidx] + B.get1(D, "gamma")[:, None] * yrt[None, :]
    lsf = B.log_sf_siteyear(D, sites, sy, mc)[:, Tm]
    sd_re = np.sqrt(B.get1(D, "sigma_u") ** 2 + B.get1(D, "sigma_om") ** 2)[:, None]
    ghx, ghw = B.A.gh_nodes(n_gh)
    sg = B.get1(D, "sigma"); lam = B.get1(D, "lambda"); tsg = B.get1(D, "tau_sig"); zt = sites.z_logarea.to_numpy()[sidx]
    h = [(wg, 0.5 * (sg[:, None] * np.exp(lam[:, None] * zt[None, :] + tsg[:, None] * xg)) ** 2) for xg, wg in zip(ghx, ghw)]
    return dict(L=base + lsf, sd=sd_re, h=h, w=w)


def pi_T(S, shift, c, a0=False):
    """mean over T of pi at log level + shift; shift: (S,) or scalar. Returns (S,)."""
    sh = np.asarray(shift, float)
    sh = sh[:, None] if sh.ndim == 1 else sh
    out = 0.0
    for wg, hh in S["h"]:
        out = out + wg * ndtr((S["L"] + sh + (0.0 if a0 else hh) - np.log(c)) / S["sd"])
    return out @ S["w"]


def main():
    t0 = time.time()
    cfg = P.CFG["A2_selection"]
    meas, sites, sy = L.load_tables()
    mc = L.R.model_config("R3")
    D = P.load_post(cfg["draws"])
    S = setup(D, sites, sy, mc, cfg["gh_nodes"])
    dl = B.get1(D, "delta_bar"); ss = B.get1(D, "sigma_site")
    psis = sorted(set(cfg["psi_primary"] + cfg["psi_extreme"])); kap = cfg["kappa"]
    # verification against the stored authoritative Delta_T draws (same thinning)
    dT_store = dict(np.load(L.scratch_dir() / f"{P.TAG}_delta_T_draws.npz"))
    n_all = len(dT_store["0.1"]); idx = np.round(np.linspace(0, n_all - 1, cfg["draws"])).astype(int)
    rows, prow, a0rows, ver = [], [], [], {}
    for c in P.THRESH:
        piM = pi_T(S, dl, c); pi0 = pi_T(S, 0.0, c)
        dT = piM - pi0
        ver[str(c)] = float(np.max(np.abs(dT - np.asarray(dT_store[str(c)])[idx])))
        for psi in psis:
            pref = pi_T(S, -psi, c)
            prow.append(dict(threshold=c, psi=psi, **{f"PR_{k}": v for k, v in P.band(pref).items()},
                             **{f"PM_{k}": v for k, v in P.band(piM).items()}))
            for k in kap:
                if k == 1.0:
                    ds = piM - pref
                elif k == 0.0:
                    ds = np.zeros_like(pref)
                else:
                    ds = pi_T(S, -psi + k * (dl + psi), c) - pref
                sM = k * (dl + psi)
                rows.append(dict(threshold=c, psi=psi, kappa=k, grid="primary" if psi in cfg["psi_primary"] else "extreme",
                                 **{f"DS_{q}": v for q, v in P.band(ds).items()}, DS_p_gt0=float((ds > 0).mean()),
                                 sM_median=float(np.median(sM)), sM_over_sigma_site_median=float(np.median(sM / ss))))
        a0M = pi_T(S, dl, c, a0=True); a00 = pi_T(S, 0.0, c, a0=True)
        a0rows.append(dict(threshold=c, **{f"DT_A0_{q}": v for q, v in P.band(a0M - a00).items()}, DT_A0_p_gt0=float(((a0M - a00) > 0).mean()),
                           **{f"PR_A0_{q}": v for q, v in P.band(a00).items()}, **{f"DT_{q}": v for q, v in P.band(dT).items()},
                           **{f"PR_{q}": v for q, v in P.band(pi0).items()}))
    G = pd.DataFrame(rows); G.to_csv(P.P4 / "tables/A2_selection_grid.csv", index=False)
    PR = pd.DataFrame(prow); PR.to_csv(P.P4 / "tables/A2_PR_by_psi.csv", index=False)
    A0 = pd.DataFrame(a0rows); A0.to_csv(P.P4 / "tables/A2_A0.csv", index=False)
    # tipping points: smallest |psi| (primary + extreme) where P(Delta^S > 0) < 0.9, by kappa
    tip = {}
    for c in P.THRESH:
        for k in kap:
            g = G[(G.threshold == c) & (G.kappa == k)].sort_values("psi")
            bad = g[g.DS_p_gt0 < 0.9]
            tip[f"{c}_kappa{k}"] = sorted(bad.psi.tolist())
    out = dict(draws=cfg["draws"], verification_max_abs_diff_vs_stored_DeltaT=ver, delta_bar=P.band(dl), sigma_site=P.band(ss),
               psi_where_P_DS_gt0_below_0_9=tip, seconds=time.time() - t0)
    P.save_json(out, P.P4 / "diagnostics/A2_selection.json")
    pd.set_option("display.width", 250)
    print(out)
    print(G[G.kappa.isin([1.0, 0.5])][["threshold", "psi", "kappa", "DS_median", "DS_lo95", "DS_hi95", "DS_p_gt0", "sM_over_sigma_site_median"]].round(4).to_string())
    print(PR[["threshold", "psi", "PR_median", "PR_lo95", "PR_hi95", "PM_median"]].round(4).to_string())
    print(A0.round(4).to_string())


if __name__ == "__main__":
    main()
