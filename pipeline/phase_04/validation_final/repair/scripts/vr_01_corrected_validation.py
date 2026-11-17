"""Track A validation repair, VR_01: corrected Monte Carlo error and validation tables from EXISTING outputs only.

Inputs (read-only): the Phase 4B per-draw site log densities (~/track_a_scratch/phase_04_vf/score_lp_{fold}_{model}.npz,
400 thinned draws), the stored site tables (tables/folds), VF_validation.json, the fold-fit diagnostics and the posterior draw
archives (scalar hyperparameters only, for the precision projection). No fitting and no scoring.

Per fold: point estimates unchanged (frozen); catchment-clustered SE recomputed with the frozen formula and checked against the
stored value; Monte Carlo error of the difference by the correlation-aware estimator (repair/scripts/vr_mcse.py) and, for
comparison, the frozen formula; block-jackknife check; Pareto k_hat site screen; Jensen bias; sensitivity without
MC-unreliable sites; projection of the MCSE ratio if all existing posterior draws were scored.
Outputs: repair/tables/VR_*.csv, repair/diagnostics/VR_results.json."""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
VR = Path(__file__).resolve().parents[1]
VFDIR = VR.parent
sys.path.insert(0, str(VR / "scripts"))
sys.path.insert(0, str(VFDIR / "scripts"))
import vr_mcse as M  # noqa: E402
import vf_lib as V  # noqa: E402   (read-only use: data tables, fold definitions, draw paths)

TA = V.TA
ORIG = Path(os.path.expanduser("~/track_a_scratch/phase_04_vf"))
FOLDS = ["W1", "W2", "C1", "C2"]
MODELS = ["m2rn", "m1s2"]
CRIT = V.CFG["adequacy"]["fold"]["mcse_share_of_clustered_se"]          # 0.10, frozen (P4-V0)
CHAINS = 4


def diag_path(fold, model):
    if model == "m2rn":
        return VFDIR / "diagnostics" / f"vf_{fold}_R3_sampling_diagnostics.json"
    if fold in ("W1", "W2"):
        return TA / "phase_03c/diagnostics" / f"cv_{fold}_m1s2_sampling_diagnostics.json"
    if fold == "C1":
        return TA / "phase_03c_r/diagnostics/cv_C1_m1s2_sampling_diagnostics.json"
    return VFDIR / "diagnostics" / f"vf_{fold}_m1s2_sampling_diagnostics.json"


def draws_path(fold, model):
    if model == "m2rn":
        return ORIG / f"vf_{fold}_R3_draws.npz"
    return Path(os.path.expanduser(V.CFG["existing_comparators"][fold])) if fold in V.CFG["existing_comparators"] \
        else ORIG / f"vf_{fold}_m1s2_draws.npz"


def clustered_se(d, clus):
    """Frozen vf_02_score.clustered: ratio-estimator SE of the total over clusters (deterministic part)."""
    g = pd.DataFrame({"d": np.asarray(d), "c": np.asarray(clus)}).groupby("c").agg(d=("d", "sum"), n=("d", "size"))
    K = len(g); tot = g.d.sum(); n = g.n.sum()
    return float(np.sqrt(K / (K - 1) * ((g.d - g.n * tot / n) ** 2).sum())), int(K), int(n)


def heldout_checks(fold, sites, meas):
    """Same held-out observations and same predictive quantity for both models."""
    held = sorted(V.heldout_sites(fold, sites))
    dr = json.load(open(diag_path(fold, "m2rn"))); dc = json.load(open(diag_path(fold, "m1s2")))
    st = pd.read_csv(VFDIR / "tables/folds" / f"{fold}_site.csv"); pw = pd.read_csv(VFDIR / "tables/folds" / f"{fold}_pointwise.csv")
    a = st[st.model == "m2rn"].set_index("site").sort_index(); b = st[st.model == "m1s2"].set_index("site").sort_index()
    za = np.load(ORIG / f"score_lp_{fold}_m2rn.npz")["site"].astype(str); zb = np.load(ORIG / f"score_lp_{fold}_m1s2.npz")["site"].astype(str)
    n_data = int(meas.site.isin(held).sum())
    out = dict(
        heldout_sites_R3_fit=sorted(dr.get("held_out_sites") or []) == held,
        heldout_sites_m1s2_fit=(sorted(dc["held_out_sites"]) == held) if dc.get("held_out_sites") is not None else "not recorded in diagnostics",
        N_train_equal=dr.get("N_train") == (dc.get("N_train") or dc.get("N")),
        N_train=[dr.get("N_train"), dc.get("N_train") or dc.get("N")],
        scored_sites_identical=bool(list(a.index) == list(b.index) and list(za) == list(zb)),
        n_scored_sites=int(len(a)), n_heldout_sites=len(held),
        heldout_sites_without_measurements=int(len(held) - len(a)),
        n_meas_identical=bool((a.n_meas == b.n_meas).all()),
        heldout_measurements_scored=[int(a.n_meas.sum()), int(b.n_meas.sum())], heldout_measurements_in_data=n_data,
        pointwise_rows=[int((pw.model == "m2rn").sum()), int((pw.model == "m1s2").sum())])
    out["pass"] = bool(out["heldout_sites_R3_fit"] and out["heldout_sites_m1s2_fit"] in (True, "not recorded in diagnostics")
                       and out["N_train_equal"] and out["scored_sites_identical"] and out["n_meas_identical"]
                       and out["heldout_measurements_scored"][0] == out["heldout_measurements_scored"][1] == n_data
                       and out["pointwise_rows"][0] == out["pointwise_rows"][1] == n_data)
    return out


def proxy_gain(fold, model, T, S_used):
    """ESS gain of T if ALL stored posterior draws were scored, from a surrogate: OLS of T on the scalar hyperparameters
    (log scale) and lp__ at the scored draws, evaluated on all draws. Central = surrogate gain; conservative = smallest
    single-hyperparameter gain; ideal = S_all / S_used with ESS capped at S_all."""
    z = np.load(draws_path(fold, model))
    keys = [k for k in ["sigma_u", "sigma_om", "sigma", "sigma_site", "alpha", "gamma", "lp__"] if k in z.files]
    S_all = len(z["lp__"])
    pick = np.linspace(0, S_all - 1, S_used).astype(int)
    X = np.column_stack([np.log(z[k]) if k.startswith("sigma") else z[k] for k in keys])
    Xs = (X - X.mean(0)) / X.std(0)
    A = np.column_stack([np.ones(S_used), Xs[pick]])
    beta, *_ = np.linalg.lstsq(A, T, rcond=None)
    r2 = 1 - ((T - A @ beta) ** 2).sum() / ((T - T.mean()) ** 2).sum()
    sur = np.column_stack([np.ones(S_all), Xs]) @ beta
    g_sur = M.ess_mean(sur, CHAINS) / M.ess_mean(sur[pick], CHAINS) if r2 > 0.05 else np.nan
    g_each = {k: M.ess_mean(Xs[:, j], CHAINS) / M.ess_mean(Xs[pick, j], CHAINS) for j, k in enumerate(keys)}
    return dict(S_all=int(S_all), keys=keys, r2_surrogate=float(r2), gain_surrogate=float(g_sur),
                gain_min_hyper=float(min(g_each.values())), gain_by_hyper={k: float(v) for k, v in g_each.items()},
                gain_ideal=float(S_all / S_used))


def main():
    t0 = time.time()
    meas, sites, sy = V.L.load_tables()
    cat = sites.set_index("site").mancat.fillna("__none_" + sites.set_index("site").index.to_series())
    VV = json.load(open(VFDIR / "diagnostics/VF_validation.json"))
    G = {g["group"]: g for g in VV["groups"]}
    res, site_rows, mf_rows, fold_rows = dict(folds={}), [], [], []
    per_fold = {}
    for fold in FOLDS:
        st = pd.read_csv(VFDIR / "tables/folds" / f"{fold}_site.csv")
        F_ = dict(heldout_checks=heldout_checks(fold, sites, meas))
        lps, sm = {}, {}
        for m in MODELS:
            z = np.load(ORIG / f"score_lp_{fold}_{m}.npz")
            lp, site = z["lp"], z["site"].astype(str)
            s = M.mc_summary(lp, CHAINS)
            s["jackknife_mcse_5blocks_per_chain"] = M.block_jackknife_elpd(lp, CHAINS, 5)
            s["jackknife_mcse_10blocks_per_chain"] = M.block_jackknife_elpd(lp, CHAINS, 10)
            _, psi = M.site_influence(lp)
            s["T"] = psi.sum(0)
            dg = json.load(open(diag_path(fold, m)))
            s["fit"] = dict(divergences=dg.get("divergences"), divergence_share=dg.get("divergence_share"),
                            min_ebfmi=float(min(dg.get("ebfmi") or [np.nan])), hyper_max_rhat=dg.get("hyper_max_rhat") or dg.get("top_max_rhat"))
            s["projection"] = proxy_gain(fold, m, s["T"], lp.shape[1])
            stored = st[st.model == m].set_index("site").loc[site]
            assert np.allclose(s["sites"]["elpd"], stored.elpd_site.to_numpy(), atol=1e-9, rtol=0)
            s["frozen_site_mcse"] = stored.mcse_elpd.to_numpy()
            lps[m], sm[m] = (lp, site), s
            thr = M.khat_threshold(lp.shape[1])
            for j, si in enumerate(site):
                site_rows.append(dict(fold=fold, model=m, site=si, catchment=cat[si], n_meas=int(stored.n_meas.iloc[j]),
                                      elpd_site=s["sites"]["elpd"][j], sd_psi=s["sites"]["sd_psi"][j], ess=s["sites"]["ess"][j],
                                      mcse_site=s["sites"]["mcse"][j], mcse_site_frozen=s["frozen_site_mcse"][j],
                                      khat=s["sites"]["khat"][j], mc_unreliable=bool(s["sites"]["khat"][j] >= thr)))
            mf_rows.append(dict(fold=fold, model=m, n_sites=s["n_sites"], n_draws=s["n_draws"], elpd=s["elpd"], mcse_corrected=s["mcse"],
                                mcse_split=s["mcse_split"], mcse_whole=s["mcse_whole"], ess_T=s["ess_T"], ess_T_split=s["ess_T_split"],
                                ess_T_whole=s["ess_T_whole"], sd_T=s["sd_T"], mcse_indep_sites=s["mcse_indep"],
                                mcse_frozen=float(np.sqrt((s["frozen_site_mcse"] ** 2).sum())), cov_inflation=s["cov_inflation"],
                                jackknife5=s["jackknife_mcse_5blocks_per_chain"], jackknife10=s["jackknife_mcse_10blocks_per_chain"],
                                jensen_bias=s["jensen_bias"], k_T=s["k_T"], khat_threshold=thr,
                                n_sites_khat_ge_0p5=int((s["sites"]["khat"] >= 0.5).sum()), n_sites_mc_unreliable=int((s["sites"]["khat"] >= thr).sum()),
                                max_khat=float(s["sites"]["khat"].max()), S_all=s["projection"]["S_all"], r2_surrogate=s["projection"]["r2_surrogate"],
                                gain_surrogate=s["projection"]["gain_surrogate"], gain_min_hyper=s["projection"]["gain_min_hyper"],
                                gain_ideal=s["projection"]["gain_ideal"], **{f"fit_{k}": v for k, v in s["fit"].items()}))
        # ---- fold difference
        site = lps["m2rn"][1]
        assert list(site) == list(lps["m1s2"][1])
        d = sm["m2rn"]["sites"]["elpd"] - sm["m1s2"]["sites"]["elpd"]
        se, K, n = clustered_se(d, cat.loc[site].to_numpy())
        assert abs(se - G[fold]["se_clustered"]) < 1e-9 and abs(d.sum() - G[fold]["diff"]) < 1e-9
        thr = M.khat_threshold(400)
        bad = (sm["m2rn"]["sites"]["khat"] >= thr) | (sm["m1s2"]["sites"]["khat"] >= thr)
        mc_c = M.mcse_independent_fits(sm["m2rn"]["mcse"], sm["m1s2"]["mcse"])
        mc_split = M.mcse_independent_fits(sm["m2rn"]["mcse_split"], sm["m1s2"]["mcse_split"])
        mc_f = G[fold]["mcse_diff"]
        mc_jk = M.mcse_independent_fits(sm["m2rn"]["jackknife_mcse_5blocks_per_chain"], sm["m1s2"]["jackknife_mcse_5blocks_per_chain"])
        # sensitivity: MC-reliable sites only (both models), same estimator
        okidx = np.flatnonzero(~bad)
        sr = {m: M.mc_summary(lps[m][0], CHAINS, idx=okidx, with_sites=False) for m in MODELS}
        se_r, _, _ = clustered_se(d[~bad], cat.loc[site[~bad]].to_numpy())
        # projection if all existing draws were scored (no fitting)
        proj = {}
        for k in ["ideal", "surrogate", "min_hyper"]:
            v = 0.0
            for m in MODELS:
                p = sm[m]["projection"]
                g = p["gain_ideal"] if k == "ideal" else p[f"gain_{k}"]
                g = g if np.isfinite(g) else p["gain_min_hyper"]
                ess_new = min(sm[m]["ess_T"] * g, p["S_all"]) if k != "ideal" else p["S_all"]
                v += sm[m]["sd_T"] ** 2 / ess_new
            proj[k] = float(np.sqrt(v) / se)
        jb = sm["m2rn"]["jensen_bias"] - sm["m1s2"]["jensen_bias"]
        row = dict(fold=fold, setting="A" if fold.startswith("W") else "B", n_sites=n, n_catchments=K, diff=float(d.sum()),
                   per_site=float(d.sum() / n), se_clustered=se, z_clustered=float(d.sum() / se),
                   mcse_diff_frozen=mc_f, ratio_frozen=mc_f / se, mcse_diff_corrected=mc_c, ratio_corrected=mc_c / se,
                   ratio_corrected_split_only=mc_split / se, mcse_diff_jackknife=mc_jk, ratio_jackknife=mc_jk / se,
                   understatement_pct=100 * (mc_c / mc_f - 1),
                   pass_predeclared_frozen_calc=bool(mc_f / se < CRIT), pass_predeclared_corrected_calc=bool(mc_c / se < CRIT),
                   se_total=float(np.hypot(se, mc_c)), z_total=float(d.sum() / np.hypot(se, mc_c)),
                   per_site_lo95_total=float((d.sum() - 1.96 * np.hypot(se, mc_c)) / n), per_site_hi95_total=float((d.sum() + 1.96 * np.hypot(se, mc_c)) / n),
                   jensen_bias_diff=float(jb), diff_jensen_corrected=float(d.sum() - jb),
                   n_sites_mc_unreliable=int(bad.sum()), diff_unreliable_sites=float(d[bad].sum()),
                   sens_reliable_only_diff=float(d[~bad].sum()), sens_reliable_only_se_clustered=se_r,
                   sens_reliable_only_mcse=M.mcse_independent_fits(sr["m2rn"]["mcse"], sr["m1s2"]["mcse"]),
                   sens_reliable_only_ratio=M.mcse_independent_fits(sr["m2rn"]["mcse"], sr["m1s2"]["mcse"]) / se_r,
                   proj_ratio_all_draws_ideal=proj["ideal"], proj_ratio_all_draws_surrogate=proj["surrogate"],
                   proj_ratio_all_draws_conservative=proj["min_hyper"],
                   required_variance_reduction=float((mc_c / (CRIT * se)) ** 2),
                   corr_T_between_models_same_index=float(np.corrcoef(sm["m2rn"]["T"], sm["m1s2"]["T"])[0, 1]))
        fold_rows.append(row)
        per_fold[fold] = dict(d=d, site=site, cat=cat.loc[site].to_numpy(), mc2={m: sm[m]["mcse"] ** 2 for m in MODELS}, bad=bad)
        res["folds"][fold] = dict(row, heldout_checks=F_["heldout_checks"],
                                  models={m: {k: v for k, v in sm[m].items() if k not in ("sites", "T", "frozen_site_mcse")} for m in MODELS})
    # ---- pooled settings (folds are separate fits: Monte Carlo errors independent across folds and models)
    pooled = {}
    for lab, fl in [("A: W1+W2", ["W1", "W2"]), ("B: C1+C2", ["C1", "C2"])]:
        d = np.concatenate([per_fold[f]["d"] for f in fl]); c = np.concatenate([per_fold[f]["cat"] for f in fl])
        se, K, n = clustered_se(d, c)
        assert abs(se - G[lab]["se_clustered"]) < 1e-9
        mc = float(np.sqrt(sum(v for f in fl for v in per_fold[f]["mc2"].values())))
        bad = np.concatenate([per_fold[f]["bad"] for f in fl])
        pooled[lab] = dict(group=lab, n_sites=n, n_catchments=K, diff=float(d.sum()), per_site=float(d.sum() / n), se_clustered=se,
                           z_clustered=float(d.sum() / se), mcse_diff_frozen=G[lab]["mcse_diff"], mcse_diff_corrected=mc,
                           ratio_frozen=G[lab]["mcse_over_se"], ratio_corrected=mc / se, se_total=float(np.hypot(se, mc)),
                           z_total=float(d.sum() / np.hypot(se, mc)), per_site_lo95_clustered=float((d.sum() - 1.96 * se) / n),
                           per_site_hi95_clustered=float((d.sum() + 1.96 * se) / n),
                           per_site_lo95_total=float((d.sum() - 1.96 * np.hypot(se, mc)) / n), per_site_hi95_total=float((d.sum() + 1.96 * np.hypot(se, mc)) / n),
                           n_sites_mc_unreliable=int(bad.sum()), diff_unreliable_sites=float(d[bad].sum()), sens_reliable_only_diff=float(d[~bad].sum()),
                           sens_reliable_only_z_clustered=float(d[~bad].sum() / clustered_se(d[~bad], c[~bad])[0]))
    A_, B_ = pooled["A: W1+W2"], pooled["B: C1+C2"]
    margin = A_["per_site"] / 2
    res["criteria"] = dict(
        C4_predeclared=dict(z_clustered=A_["z_clustered"], z_ge_2=bool(A_["z_clustered"] >= 2), note="calibration part unchanged (VF_validation.json C4)"),
        C4_with_corrected_mc_error=dict(z_total=A_["z_total"], z_ge_2=bool(A_["z_total"] >= 2), status="AMENDED reporting (MC error added to the clustered SE)"),
        C4_without_W1=dict(z=G["W2"]["z"], z_total=[r for r in fold_rows if r["fold"] == "W2"][0]["z_total"]),
        C5_predeclared=dict(margin=margin, hi95=B_["per_site_hi95_clustered"], supported=bool(B_["per_site_hi95_clustered"] < margin)),
        C5_with_corrected_mc_error=dict(margin=margin, hi95=B_["per_site_hi95_total"], supported=bool(B_["per_site_hi95_total"] < margin),
                                        material_loss=bool(B_["per_site_hi95_total"] < 0), status="AMENDED reporting (MC error added)"),
        C5_margin_with_W2_only=dict(margin=G["W2"]["per_site"] / 2, supported=bool(B_["per_site_hi95_total"] < G["W2"]["per_site"] / 2)))
    res["pooled"] = pooled
    res["khat_threshold_400"] = M.khat_threshold(400)
    res["seconds"] = time.time() - t0
    pd.DataFrame(site_rows).to_csv(VR / "tables/VR_sites.csv", index=False)
    pd.DataFrame(mf_rows).to_csv(VR / "tables/VR_model_fold_mc.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(VR / "tables/VR_fold_results.csv", index=False)
    pd.DataFrame(list(pooled.values())).to_csv(VR / "tables/VR_pooled_results.csv", index=False)
    V.save_json(res, VR / "diagnostics/VR_results.json")
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 60)
    print(pd.DataFrame(mf_rows)[["fold", "model", "elpd", "mcse_frozen", "mcse_indep_sites", "mcse_corrected", "mcse_split", "mcse_whole", "jackknife5",
                                 "jackknife10", "ess_T", "cov_inflation", "jensen_bias", "k_T", "n_sites_khat_ge_0p5", "n_sites_mc_unreliable", "max_khat",
                                 "r2_surrogate", "gain_surrogate", "gain_min_hyper", "gain_ideal"]].round(3).to_string())
    print(pd.DataFrame(fold_rows).T.to_string())
    print(pd.DataFrame(list(pooled.values())).T.to_string())
    print(json.dumps(res["criteria"], indent=1, default=str))
    print(json.dumps({f: res["folds"][f]["heldout_checks"] for f in FOLDS}, indent=1, default=str))


if __name__ == "__main__":
    main()
