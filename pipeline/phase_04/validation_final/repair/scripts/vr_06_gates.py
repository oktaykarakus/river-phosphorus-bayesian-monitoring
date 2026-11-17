"""VR_06: pre-declared gates of the W1 final attempt (VR_03 protocol). No sampling, no scoring.

S1  W2 precision:  MCSE(diff) / clustered SE < 0.10 by BOTH the correlation-aware estimator (vr_mcse, primary) and the frozen
    per-site formula (VC_00 S2 rule: with the larger value as well; both must pass). Uses the lean all-draw (or AM2) scores.
S4  W1 precision:  the same rule for the W1 refit.
S5  C4:            clustered z >= 2 over W1 (refit) + W2 and calibration not worse (PIT max-decile deviation and 90% coverage
                   within 0.02 of M1-S2), pre-declared; z with MC error added reported (amended reporting, not a gate).
Usage: vr_06_gates.py S1|S4|S5 [LABEL]   (LABEL = all or 1600; default all). Exit code 0 = pass, 3 = fail.
Writes repair/diagnostics/VR_gate_{stage}.json."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import vr_final_lib as K
import vr_mcse as M

V = K.V
AD = K.CFG["adequacy_fold"]


def frozen_site_mcse(lp):
    """vf_02_score.site_mcse reproduced (arviz bulk ESS of the weights, floor 10, cap S)."""
    import arviz as az
    out = []
    for row in lp:
        w = np.exp(row - row.max()); S = len(w)
        try:
            e = float(az.ess(w.reshape(4, S // 4)))
        except Exception:
            e = float(S)
        e = max(min(e, S), 10.0)
        out.append(w.std(ddof=1) / (np.sqrt(e) * w.mean()))
    return np.array(out)


def clustered(d, c):
    g = pd.DataFrame({"d": d, "c": c}).groupby("c").agg(d=("d", "sum"), n=("d", "size"))
    Kc = len(g); tot = g.d.sum(); n = g.n.sum()
    return float(np.sqrt(Kc / (Kc - 1) * ((g.d - g.n * tot / n) ** 2).sum())), int(Kc), int(n), float(tot)


def fold_scores(fold, lab):
    src = "vr" if fold == "W1" else "vf"
    za = np.load(K.SCR / f"score_lp_{lab}_{fold}_m2rn_{src}.npz"); zb = np.load(K.SCR / f"score_lp_{lab}_{fold}_m1s2_vf.npz")
    assert list(za["site"]) == list(zb["site"])
    return za["lp"], zb["lp"], za["site"].astype(str)


def precision_gate(fold, lab):
    meas, sites, sy = V.L.load_tables()
    cat = sites.set_index("site").mancat.fillna("__none_" + sites.set_index("site").index.to_series())
    la, lb, site = fold_scores(fold, lab)
    sa, sb = M.mc_summary(la, 4), M.mc_summary(lb, 4)
    d = sa["sites"]["elpd"] - sb["sites"]["elpd"]
    se, Kc, n, tot = clustered(d, cat.loc[site].to_numpy())
    mc_corr = M.mcse_independent_fits(sa["mcse"], sb["mcse"])
    mc_frozen = float(np.sqrt((frozen_site_mcse(la) ** 2).sum() + (frozen_site_mcse(lb) ** 2).sum()))
    thr = M.khat_threshold(la.shape[1])
    res = dict(fold=fold, scoring_draws=dict(m2rn=int(la.shape[1]), m1s2=int(lb.shape[1])), n_sites=n, n_catchments=Kc, diff=tot,
               per_site=tot / n, se_clustered=se, z_clustered=tot / se, mcse_corrected=mc_corr, mcse_frozen_formula=mc_frozen,
               ratio_corrected=mc_corr / se, ratio_frozen_formula=mc_frozen / se,
               pass_corrected=bool(mc_corr / se < AD["mcse_share_of_clustered_se"]),
               pass_frozen_formula=bool(mc_frozen / se < AD["mcse_share_of_clustered_se"]),
               ess_T=dict(m2rn=sa["ess_T"], m1s2=sb["ess_T"]), k_T=dict(m2rn=sa["k_T"], m1s2=sb["k_T"]),
               jackknife_mcse=M.mcse_independent_fits(M.block_jackknife_elpd(la, 4, 5), M.block_jackknife_elpd(lb, 4, 5)),
               khat_threshold=thr,
               n_sites_khat_ge_thr=dict(m2rn=int((sa["sites"]["khat"] >= thr).sum()), m1s2=int((sb["sites"]["khat"] >= thr).sum())),
               max_khat=dict(m2rn=float(sa["sites"]["khat"].max()), m1s2=float(sb["sites"]["khat"].max())),
               z_with_mc=tot / np.hypot(se, mc_corr))
    res["pass"] = bool(res["pass_corrected"] and res["pass_frozen_formula"])
    pd.DataFrame(dict(fold=fold, site=site, catchment=cat.loc[site].to_numpy(), elpd_m2rn=sa["sites"]["elpd"], elpd_m1s2=sb["sites"]["elpd"],
                      khat_m2rn=sa["sites"]["khat"], khat_m1s2=sb["sites"]["khat"])).to_csv(K.TAB / f"VR_final_sites_{fold}_{lab}.csv", index=False)
    return res


def calib(u):
    h = np.histogram(u, bins=10, range=(0, 1))[0] / len(u)
    return float(np.abs(h - .1).max()), float(((u > .05) & (u < .95)).mean())


def c4(lab):
    meas, sites, sy = V.L.load_tables()
    cat = sites.set_index("site").mancat.fillna("__none_" + sites.set_index("site").index.to_series())
    w2lab = "all" if (K.TAB / "VR_final_sites_W2_all.csv").exists() and lab != "400" else lab   # W2 scored with all draws in S1
    S = pd.concat([pd.read_csv(K.TAB / f"VR_final_sites_W1_{lab}.csv"), pd.read_csv(K.TAB / f"VR_final_sites_W2_{w2lab}.csv")])
    d = (S.elpd_m2rn - S.elpd_m1s2).to_numpy()
    se, Kc, n, tot = clustered(d, S.catchment.to_numpy())
    g1 = json.load(open(K.DIAG / "VR_gate_S1.json")); g4 = json.load(open(K.DIAG / "VR_gate_S4.json"))
    mc = float(np.hypot(g1["mcse_corrected"], g4["mcse_corrected"]))
    # calibration (frozen definition, 400-draw frozen scorer): W1 refit from S3, W1 M1-S2 / W2 both models stored in Phase 4B
    P1 = pd.read_csv(K.FOLDS_VR / "W1_pointwise.csv")
    Pst = pd.concat([pd.read_csv(K.VFDIR / "tables/folds" / f"{f}_pointwise.csv") for f in ["W1", "W2"]])
    Pst = Pst[~((Pst.fold == "W1") & (Pst.model == "m2rn"))]
    P = pd.concat([P1, Pst])
    cr = calib(P[P.model == "m2rn"].pit.to_numpy()); cm = calib(P[P.model == "m1s2"].pit.to_numpy())
    res = dict(n_sites=n, n_catchments=Kc, diff=tot, per_site=tot / n, se_clustered=se, z_clustered=tot / se,
               pit_max_dev=dict(m2rn=cr[0], m1s2=cm[0]), pit_cov90=dict(m2rn=cr[1], m1s2=cm[1]),
               z_ge_2=bool(tot / se >= 2), calibration_not_worse=bool(cr[0] - cm[0] <= 0.02 and abs(cr[1] - cm[1]) <= 0.02),
               mcse_corrected=mc, z_with_mc_AMENDED_REPORTING=tot / np.hypot(se, mc),
               per_site_95_with_mc_AMENDED_REPORTING=[(tot - 1.96 * np.hypot(se, mc)) / n, (tot + 1.96 * np.hypot(se, mc)) / n])
    res["pass"] = bool(res["z_ge_2"] and res["calibration_not_worse"])
    return res


def main(stage, lab="all"):
    if stage == "S1":
        res = precision_gate("W2", lab)
    elif stage == "S4":
        res = precision_gate("W1", lab)
    else:
        res = c4(lab)
    res["stage"] = stage; res["scoring_label"] = lab
    K.save_json(res, K.DIAG / f"VR_gate_{stage}.json")
    K.ledger(f"gate {stage}")
    print(json.dumps(res, indent=1, default=float))
    sys.exit(0 if res["pass"] else 3)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "all")
