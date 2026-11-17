"""Phase 3C-F: practical importance of the light single-measurement upper tail (Phase 3B TC4) for the annual arithmetic-mean
exceedance estimands. Existing posterior draws only; no new model.

For n_draws posterior draws:
 1. Standardised measurement residuals z = (log y - eta) / sigma_i (uncensored); tail frequencies P(z > 2, 2.5, 3) by network vs N(0,1).
 2. Arithmetic-mean correction factor (smearing) by network: kappa = mean over measurements of exp(sigma z - sigma^2/2) (uncensored)
    or Phi(a - sigma) / Phi(a) (censored at standardised limit a). Under the model E[kappa] = 1: kappa > 1 means the lognormal
    formula exp(eta + sigma^2/2) under-states arithmetic means. Also the share of the arithmetic total carried by z > 2 measurements.
 3. Delta_T sensitivity: the frozen Phase 3C Delta_T with the log annual mean shifted by log kappa_RSN (RSN) and log kappa_MS (MS),
    per draw; plus a worst-case asymmetric shift (kappa_MS at its 95th, kappa_RSN at its 5th percentile).
Usage: p3f_07_upper_tail.py --tag main_R3 [--where 3cf|3cr]
Outputs: diagnostics/upper_tail_<tag>.json, tables/upper_tail_<tag>.csv
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtr, log_ndtr

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import p3f_lib as L  # noqa: E402

B = L.B; C = L.C; A = C.A


def delta_T_shift(D, sites, sy, mc, shift_R=0.0, shift_M=0.0, n_gh=15):
    """Copy of the frozen p3c_lib.delta_T (include_rn=True, include_location=True) with an additive shift of the log annual
    arithmetic mean by network (arrays of length S or scalars). shift 0 reproduces the frozen function exactly (checked)."""
    comp = C.site_components(D, sites, mc)
    Tm = sy.in_target_T.to_numpy() == 1
    sidx = sy.site_idx.to_numpy()[Tm] - 1; yrt = (sy.year_idx.to_numpy()[Tm] == 2).astype(float)
    w = sy.target_weight.to_numpy()[Tm]; w = w / w.sum()
    loc = comp["vA"] + comp["vC"] + comp["vB"] + comp["w"] + comp["rn"]
    base = comp["fixed"][:, sidx] + loc[:, sidx] + B.get1(D, "gamma")[:, None] * yrt[None, :]
    dT = comp["delta_r"][:, sites.region_idx.to_numpy()[sidx] - 1]
    lsf = B.log_sf_siteyear(D, sites, sy, mc)[:, Tm]
    sd_re = np.sqrt(B.get1(D, "sigma_u") ** 2 + B.get1(D, "sigma_om") ** 2)[:, None]
    sg = B.get1(D, "sigma"); ghx, ghw = A.gh_nodes(n_gh); lam = B.get1(D, "lambda"); tsg = B.get1(D, "tau_sig"); zt = sites.z_logarea.to_numpy()[sidx]
    sR = np.broadcast_to(np.asarray(shift_R, float), (len(sg),))[:, None]; sM = np.broadcast_to(np.asarray(shift_M, float), (len(sg),))[:, None]
    res = {}
    for c in C.THRESH:
        pR = np.zeros_like(base); pM = np.zeros_like(base)
        for xg, wg in zip(ghx, ghw):
            h = 0.5 * (sg[:, None] * np.exp(lam[:, None] * zt[None, :] + tsg[:, None] * xg)) ** 2
            pR += wg * ndtr((base + lsf + h + sR - np.log(c)) / sd_re)
            pM += wg * ndtr((base + lsf + dT + h + sM - np.log(c)) / sd_re)
        res[c] = dict(PR=pR @ w, PM=pM @ w, D=pM @ w - pR @ w)
    return res


def summ(x):
    return dict(median=float(np.median(x)), lo95=float(np.quantile(x, .025)), hi95=float(np.quantile(x, .975)))


def main(tag, where, n_draws):
    meas, sites, sy = L.load_tables()
    mc = L.R.model_config("R3" if tag.startswith(("main_R3", "bench")) else "R1")
    D0 = L.load_draws(tag, where=where)
    S = B.ndraws(D0); pick = np.linspace(0, S - 1, n_draws).astype(int)
    D = {k: (np.asarray(v)[pick] if (np.ndim(v) >= 1 and len(v) == S) else v) for k, v in D0.items()}
    y = meas.value.to_numpy(); cen = meas.censored.to_numpy(); lim = meas.limit.to_numpy()
    net = meas.network.to_numpy(); si = meas.site_idx.to_numpy() - 1
    sgs = B.sig_site(D, sites, mc)                       # (S', I)
    rows, kap = [], {"RSN": [], "MS": []}
    tails = {n: {t: [] for t in (2, 2.5, 3)} for n in ("RSN", "MS")}
    top_share = {n: [] for n in ("RSN", "MS")}
    for b0 in range(0, n_draws, 25):
        idx = np.arange(b0, min(n_draws, b0 + 25))
        eta = B.eta_obs(D, meas, sites, sy, mc, draws=idx)
        for k, s in enumerate(idx):
            sd = sgs[s][si]
            z = (np.log(np.where(cen, lim, y)) - eta[k]) / sd
            r = np.where(cen, np.exp(log_ndtr(z - sd) - log_ndtr(z)), np.exp(sd * z - 0.5 * sd ** 2))
            for n in ("RSN", "MS"):
                kk = net == n
                kap[n].append(float(r[kk].mean()))
                for t in (2, 2.5, 3):
                    tails[n][t].append(float(((z > t) & ~cen & kk).sum() / kk.sum()))
                ex = np.exp(eta[k][kk] + 0.5 * sd[kk] ** 2)        # model-expected measurement means
                obs_contrib = np.where(cen[kk], 0.0, y[kk])
                top_share[n].append(float(obs_contrib[(z[kk] > 2) & ~cen[kk]].sum() / np.maximum(1e-12, ex.sum())))
    from scipy.stats import norm
    out = dict(tag=tag, where=where, n_draws=n_draws)
    for n in ("RSN", "MS"):
        out[f"kappa_{n}"] = summ(np.array(kap[n]))
        out[f"tail_{n}"] = {str(t): dict(observed=summ(np.array(tails[n][t])), normal=float(1 - norm.cdf(t))) for t in (2, 2.5, 3)}
        out[f"share_of_expected_total_from_z_gt_2_{n}"] = summ(np.array(top_share[n]))
    kR, kM = np.log(np.array(kap["RSN"])), np.log(np.array(kap["MS"]))
    base = delta_T_shift(D, sites, sy, mc)
    chk = C.delta_T(D, sites, sy, mc)
    out["delta_T_reproduces_frozen_max_abs"] = float(max(np.abs(base[c]["D"] - chk[c]["D"]).max() for c in C.THRESH))
    sh = delta_T_shift(D, sites, sy, mc, shift_R=kR, shift_M=kM)
    wc = delta_T_shift(D, sites, sy, mc, shift_R=np.quantile(kR, .05), shift_M=np.quantile(kM, .95))
    wc2 = delta_T_shift(D, sites, sy, mc, shift_R=np.quantile(kR, .95), shift_M=np.quantile(kM, .05))
    tab = []
    for c in C.THRESH:
        tab.append(dict(threshold=c, base=summ(base[c]["D"]), kappa_adjusted=summ(sh[c]["D"]), worst_case_up=summ(wc[c]["D"]), worst_case_down=summ(wc2[c]["D"]),
                        PR_base=float(np.median(base[c]["PR"])), PR_adj=float(np.median(sh[c]["PR"])), PM_base=float(np.median(base[c]["PM"])),
                        PM_adj=float(np.median(sh[c]["PM"])), shift_median=float(np.median(sh[c]["D"] - base[c]["D"]))))
    out["delta_T"] = tab
    L.save_json(out, L.ROOT / "diagnostics" / f"upper_tail_{tag}.json")
    pd.DataFrame([dict(threshold=t["threshold"], base_median=t["base"]["median"], adj_median=t["kappa_adjusted"]["median"], shift=t["shift_median"],
                       worst_up=t["worst_case_up"]["median"], worst_down=t["worst_case_down"]["median"]) for t in tab]).to_csv(
        L.ROOT / "tables" / f"upper_tail_{tag}.csv", index=False)
    print(json.dumps({k: v for k, v in out.items() if k != "delta_T"}, indent=1))
    for t in tab:
        print(t["threshold"], {k: (round(v["median"], 4) if isinstance(v, dict) else round(v, 4)) for k, v in t.items() if k != "threshold"})


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="main_R3"); ap.add_argument("--where", default="3cf"); ap.add_argument("--n_draws", type=int, default=200)
    a = ap.parse_args(); main(a.tag, a.where, a.n_draws)
