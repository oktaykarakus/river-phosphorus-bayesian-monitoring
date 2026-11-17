"""Correction stage, VC_00 addendum S1-S2: checks before the W2 rescoring (stored Phase 4B outputs only, no scoring).
S1 score-definition consistency (both models), S2 appropriateness of the frozen MCSE calculation.
Writes correction/diagnostics/VC_score_checks.json and tables/VC_mcse_method_sites.csv."""
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import vc_lib as K  # noqa: E402

V = K.V; L = V.L; F, C, R, B = V.F, V.C, V.R, V.B
TA = V.TA


def ess(x, method="bulk"):
    import arviz as az
    x = np.asarray(x, float); S = len(x)
    return float(az.ess(x.reshape(4, S // 4), method=method))


def s1(fold, sites):
    out = {}
    held = sorted(V.heldout_sites(fold, sites))
    dr = json.load(open(K.VFDIR / "diagnostics" / f"vf_{fold}_R3_sampling_diagnostics.json"))
    dc = json.load(open(TA / "phase_03c/diagnostics" / f"cv_{fold}_m1s2_sampling_diagnostics.json"))
    out["a_heldout_identical"] = bool(sorted(dr["held_out_sites"]) == held == sorted(dc["held_out_sites"]))
    out["b_sizes"] = {k: [dr.get(k), dc.get(k)] for k in ["N_train", "I", "J"]}
    out["b_sizes_identical"] = bool(all(a == b for a, b in out["b_sizes"].values()))
    mr, mcfg = dr["model_config"], dc["model_config"]
    diff = sorted(k for k in set(mr) | set(mcfg) if mr.get(k) != mcfg.get(k))
    out["c_config_differences"] = {k: [mr.get(k), mcfg.get(k)] for k in diff}
    out["c_only_network_keys"] = bool(all(("tu" in k.lower()) or ("rn" in k.lower()) or k in ("r3", "param", "parameterisation", "variant", "name")
                                          for k in diff))
    S_ = pd.read_csv(K.VFDIR / "tables/folds" / f"{fold}_site.csv"); P_ = pd.read_csv(K.VFDIR / "tables/folds" / f"{fold}_pointwise.csv")
    a = S_[S_.model == "m2rn"].set_index("site"); b = S_[S_.model == "m1s2"].set_index("site")
    out["d_same_sites"] = bool(set(a.index) == set(b.index)); out["d_n_sites"] = int(len(a))
    out["d_same_n_meas"] = bool((a.n_meas.sort_index() == b.n_meas.sort_index()).all())
    out["d_pointwise_counts"] = [int((P_.model == "m2rn").sum()), int((P_.model == "m1s2").sum())]
    meas, _, _ = L.load_tables()
    out["d_heldout_measurements_in_data"] = int(meas.site.isin(held).sum())
    out["d_counts_match_data"] = bool(out["d_pointwise_counts"][0] == out["d_pointwise_counts"][1] == out["d_heldout_measurements_in_data"]
                                      == int(a.n_meas.sum()))
    # (f) comparator conditional vs its direct draws (all draws), same statistic as the RB check
    D = V.comparator_draws(fold); mc = C.model_config(tu=0)
    hidx = np.flatnonzero(sites.site.isin(held).to_numpy())
    mu, sd = F.heldout_theta_conditional(D, sites, mc, hidx, None)
    th = np.asarray(D["theta"], float)[:, hidx]; d = mu - th; S = d.shape[0]
    e = np.array([ess(d[:, i]) for i in range(d.shape[1])]); z = d.mean(0) / (d.std(0, ddof=1) / np.sqrt(np.clip(e, 10, S)))
    out["f_comparator_rb_all_draws"] = dict(n_draws=int(S), max_abs_mean_diff=float(np.abs(d.mean(0)).max()), mean_z=float(z.mean()),
                                            sd_z=float(z.std(ddof=1)), max_abs_z=float(np.abs(z).max()))
    out["f_pass"] = bool(out["f_comparator_rb_all_draws"]["max_abs_mean_diff"] <= 0.05)
    out["pass"] = bool(out["a_heldout_identical"] and out["b_sizes_identical"] and out["c_only_network_keys"] and out["d_same_sites"]
                       and out["d_same_n_meas"] and out["d_counts_match_data"] and out["f_pass"])
    return out


def s2(fold):
    import arviz as az
    VV = json.load(open(K.VFDIR / "diagnostics/VF_validation.json")); se = {g["group"]: g for g in VV["groups"]}[fold]["se_clustered"]
    rows, tot = [], {}
    for model in ["m2rn", "m1s2"]:
        z = np.load(K.ORIG_SCRATCH / f"score_lp_{fold}_{model}.npz"); lp = z["lp"]; site = z["site"].astype(str); S = lp.shape[1]
        W = np.exp(lp - lp.max(1, keepdims=True))
        for i in range(len(site)):
            w = W[i]; cv = w.std(ddof=1) / w.mean()
            eb = max(min(ess(w, "bulk"), S), 10.0); em = max(min(ess(w, "mean"), S), 10.0)
            _, k = az.psislw(lp[i].copy())
            rows.append(dict(fold=fold, model=model, site=site[i], cv=cv, ess_bulk=eb, ess_mean=em, mcse_frozen=cv / np.sqrt(eb),
                             mcse_ess_mean=cv / np.sqrt(em), pareto_k=float(np.asarray(k).ravel()[0])))
        T = (W / W.mean(1, keepdims=True)).sum(0)
        tot[model] = float(T.std(ddof=1) / np.sqrt(max(min(ess(T, "mean"), S), 10.0)))
    D = pd.DataFrame(rows)
    res = dict(clustered_se=se, ratio_frozen=float(np.sqrt((D.mcse_frozen ** 2).sum()) / se),
               ratio_ess_mean=float(np.sqrt((D.mcse_ess_mean ** 2).sum()) / se),
               ratio_cross_site_correlated=float(np.sqrt(sum(v ** 2 for v in tot.values())) / se),
               mcse_total_correlated_by_model=tot,
               mcse_total_independent_by_model={m: float(np.sqrt((D[D.model == m].mcse_frozen ** 2).sum())) for m in ["m2rn", "m1s2"]},
               n_sites_k_ge_07=int((D.pareto_k >= 0.7).sum()), n_sites_k_ge_05=int((D.pareto_k >= 0.5).sum()), max_pareto_k=float(D.pareto_k.max()),
               median_ess_bulk_over_mean=float(np.median(D.ess_bulk / D.ess_mean)))
    res["rel_change_ess_mean"] = res["ratio_ess_mean"] / res["ratio_frozen"] - 1
    res["rel_change_correlated"] = res["ratio_cross_site_correlated"] / res["ratio_frozen"] - 1
    res["appropriate"] = bool(abs(res["rel_change_ess_mean"]) < 0.2 and abs(res["rel_change_correlated"]) < 0.2 and res["n_sites_k_ge_07"] == 0)
    res["frozen_understates"] = bool(max(res["ratio_ess_mean"], res["ratio_cross_site_correlated"]) > res["ratio_frozen"])
    return res, D


def main():
    meas, sites, sy = L.load_tables()
    out = dict(code_check=dict(
        e_same_function="both models scored by p3c_folds.heldout_scores (same pick = linspace(0,S0-1,n).astype(int), same held-out measurements, "
                        "same quadrature in site_joint_logpred, same censoring likelihood); the only model-specific inputs are the draws and mc "
                        "(mc['tu'] selects the river-network conditional)",
        heldout_scores_source=inspect.getsourcefile(F.heldout_scores)))
    tabs = []
    for fold in ["W2", "W1", "C1", "C2"]:
        o = dict(S1=s1(fold, sites) if fold in ("W1", "W2") else "not applicable (no rescoring)")
        o["S2"], D = s2(fold); tabs.append(D)
        out[fold] = o
        print(fold, json.dumps({k: v for k, v in o.items()}, indent=1, default=str)[:3000], flush=True)
    pd.concat(tabs).to_csv(K.VC / "tables" / "VC_mcse_method_sites.csv", index=False)
    K.save_json(out, K.VC / "diagnostics" / "VC_score_checks.json")
    K.ledger("score-definition and MCSE-method checks (S1-S2)")


if __name__ == "__main__":
    main()
