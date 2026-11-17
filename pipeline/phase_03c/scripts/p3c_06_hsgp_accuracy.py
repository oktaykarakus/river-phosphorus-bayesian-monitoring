"""Phase 3C Stage 5: HSGP approximation accuracy over the POSTERIOR-supported range, and numerical vs model uncertainty.

For each fit tag (default: the frozen reference m1s2_full and the M2-RN pilot m2rn_full):
  * posterior quantiles of rho (and sigma_w);
  * exact squared-exponential correlation vs the HSGP approximation (m = 48 x 59, c_b = 1.25) at all site pairs, for rho at
    the posterior 1/5/50/95/99% quantiles and at the prior support edges (12, 70 km): max |error|, RMS error over pairs < 100 km,
    max error on the diagonal (variance), error for sites within 10% of the bounding-box edge;
  * numerical vs model uncertainty: approximation error of the field covariance (sigma_w^2 x correlation error) compared with
    the posterior variance of w(s_i) at the sites;
  * an enlarged basis (c_b = 1.5, m scaled by 1.5/1.25 -> 58 x 71) evaluated at the same rho, to show whether a corrected basis
    is needed.
Output: diagnostics/hsgp_accuracy_posterior.json, tables/hsgp_accuracy_posterior.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p3c_lib as L  # noqa: E402

B = L.B


def approx_corr(x, y, rho, m=None, c_b=None):
    PhiE, slE, PhiN, slN = B.hsgp_basis(x, y, m=m, c_b=c_b)
    LAM2 = slE[:, None] ** 2 + slN[None, :] ** 2
    S = 2 * np.pi * rho ** 2 * np.exp(-0.5 * rho ** 2 * LAM2)                 # spectral density, sigma_w = 1
    Psi = (PhiE[:, :, None] * PhiN[:, None, :]).reshape(len(x), -1)
    return (Psi * S.ravel()[None, :]) @ Psi.T


def main(tags):
    meas, sites, sy = L.load_tables()
    x = sites.x_km.to_numpy(); y = sites.y_km.to_numpy()
    d = np.sqrt((x[:, None] - x[None, :]) ** 2 + (y[:, None] - y[None, :]) ** 2)
    sc = B.hsgp_constants(); cE, cN = sc["hsgp_centre_km"]; SE, SN = sc["hsgp_half_range_km"]
    edge = (np.abs(x - cE) > 0.9 * SE) | (np.abs(y - cN) > 0.9 * SN)
    near = d < 100
    out = {"n_sites_near_box_edge": int(edge.sum())}
    rows = []
    for tag in tags:
        ref = tag == "m1s2_full"
        try:
            D = L.load_draws(tag, ref=ref)
        except FileNotFoundError:
            print("missing", tag); continue
        rho = B.get1(D, "rho"); sw = B.get1(D, "sigma_w")
        q = {p: float(np.quantile(rho, p)) for p in [0.01, 0.05, 0.5, 0.95, 0.99]}
        wvar = np.asarray(D["w_site"], float).var(0)
        out[tag] = dict(rho_quantiles=q, sigma_w_median=float(np.median(sw)), rho95_rule_ok=q[0.95] <= L.CFG["hsgp"]["rho95_max_km"],
                        rho05_ge_rho_min=q[0.05] >= L.CFG["hsgp"]["rho_min_km"], posterior_var_w_median=float(np.median(wvar)),
                        posterior_var_w_min=float(wvar.min()))
        for lab, r in [("q01", q[0.01]), ("q05", q[0.05]), ("q50", q[0.5]), ("q95", q[0.95]), ("q99", q[0.99]), ("12km", 12.0), ("70km", 70.0)]:
            for basis, m, cb in [("primary_1.25", None, None), ("enlarged_1.5", [58, 71], 1.5)]:
                Ka = approx_corr(x, y, r, m=m, c_b=cb); Ke = np.exp(-0.5 * d ** 2 / r ** 2)
                E = Ka - Ke
                rows.append(dict(tag=tag, at=lab, rho_km=r, basis=basis, max_abs_err=float(np.abs(E).max()),
                                 rms_err_pairs_lt100km=float(np.sqrt((E[near] ** 2).mean())), max_abs_err_diag=float(np.abs(np.diag(E)).max()),
                                 max_abs_err_edge_sites=float(np.abs(E[edge][:, edge]).max()) if edge.any() else np.nan,
                                 field_cov_err_over_post_var_median=float(np.median(np.median(sw) ** 2 * np.abs(np.diag(E)) / wvar))))
                print(rows[-1], flush=True)
    T = pd.DataFrame(rows); T.to_csv(L.ROOT / "tables" / "hsgp_accuracy_posterior.csv", index=False)
    out["table"] = "tables/hsgp_accuracy_posterior.csv"
    L.save_json(out, L.ROOT / "diagnostics" / "hsgp_accuracy_posterior.json")
    print(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--tags", nargs="+", default=["m1s2_full", "m2rn_full"])
    main(ap.parse_args().tags)
