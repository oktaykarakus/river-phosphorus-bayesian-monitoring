"""Phase 4B A2/A3: held-out scoring of the converged R3 fold fits against the M1-S2 comparators (P4-V0 §3 A1, frozen).

Scores: frozen phase_03c/scripts/p3c_folds.heldout_scores (unchanged), 400 evenly thinned draws per fit: site-joint ELPD
(primary), pointwise LPD, PIT, site-year sample-mean exceedance Brier / log scores and 90% coverage.
The site-joint log predictive density per draw is captured (wrapper around the frozen function, values unchanged) to give the
Monte Carlo SE of every site's ELPD (ratio estimator, ESS of the importance weights by arviz over the 4 chains).
Uncertainty: SEs clustered by management catchment (ratio estimator over catchments, as P4_01), catchment bootstrap (2,000)
and leave-one-catchment-out range. Pooled settings: A = W1 + W2 (clusters = catchments across both folds), B = C1 + C2.
Adequacy (P4-V0, frozen): 0 divergences (<= 0.1%), E-BFMI >= 0.3, hyperparameter R-hat <= 1.05, MCSE of the fold ELPD
difference < 10% of its clustered SE. Criteria: C4 clustered z >= 2 with calibration not worse (PIT max-decile deviation and
90% coverage within +-0.02 of M1-S2); C5 95% upper bound of the per-site difference over catchment folds < half the setting-A
per-site gain. Equivalence is reported separately: both 95% bounds inside +-margin.
Usage: vf_02_score.py [--folds W1 W2 C1 C2]
Outputs: tables/VF_scores_{site,pointwise,siteyear}.csv, tables/VF_validation_{folds,strata,calibration}.csv,
         diagnostics/VF_validation.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import vf_lib as V  # noqa: E402

L = V.redirect()
F, C, R, B = V.F, V.C, V.R, V.B
TA = V.TA
RNG = np.random.default_rng(V.CFG["seeds"]["scoring"])
CAPTURE = []
_orig = F.site_joint_logpred


def _wrapped(*a, **k):
    lp = _orig(*a, **k)
    CAPTURE.append(np.asarray(lp, float))
    return lp


F.site_joint_logpred = _wrapped


def site_mcse(lp, chains=4):
    """MCSE of log(mean exp(lp)) with lp (S,) chain-major; ESS of the weights by arviz (rank-normalised bulk)."""
    import arviz as az
    w = np.exp(lp - lp.max())
    S = len(w)
    try:
        ess = float(az.ess(w.reshape(chains, S // chains)))
    except Exception:
        ess = float(S)
    ess = max(min(ess, S), 10.0)
    return float(w.std(ddof=1) / (np.sqrt(ess) * w.mean())), ess


def fit_diag(fold, model):
    if model == "m2rn":
        p = V.VF / "diagnostics" / f"vf_{fold}_R3_sampling_diagnostics.json"
    elif fold in ("W1", "W2"):
        p = TA / "phase_03c/diagnostics" / f"cv_{fold}_m1s2_sampling_diagnostics.json"
    elif fold == "C1":
        p = TA / "phase_03c_r/diagnostics/cv_C1_m1s2_sampling_diagnostics.json"
    else:
        p = V.VF / "diagnostics" / f"vf_{fold}_m1s2_sampling_diagnostics.json"
    d = json.load(open(p))
    if d.get("hyper_max_rhat") is None:                       # Phase 3C fits recorded top-level parameters only
        d["hyper_max_rhat"] = d.get("top_max_rhat"); d["hyper_argmax_rhat"] = d.get("top_argmax_rhat")
    lr = (d.get("log_scale_diagnostics") or d.get("log_range_diagnostics") or {})
    worst_log = max([v["r_hat"] for v in lr.values()] or [1.0])
    return dict(fold=fold, model=model, source=str(p.relative_to(TA)), cpu_h=d.get("cpu_h"), divergences=d.get("divergences"),
                divergence_share=d.get("divergence_share", d.get("divergences", 0) / (4 * d.get("iter_sampling", 500))),
                min_ebfmi=float(min(d.get("ebfmi") or [np.nan])), hyper_max_rhat=d.get("hyper_max_rhat"), hyper_argmax_rhat=d.get("hyper_argmax_rhat"),
                worst_log_scale_rhat=worst_log, hyper_min_ess_bulk=d.get("hyper_min_ess_bulk", d.get("top_min_ess_bulk")),
                hyper_argmin_ess_bulk=d.get("hyper_argmin_ess_bulk", d.get("top_argmin_ess_bulk")),
                hyper_min_ess_tail=d.get("hyper_min_ess_tail", d.get("top_min_ess_tail")),
                latent_max_rhat=d.get("latent_max_rhat"), latent_min_ess_bulk=d.get("latent_min_ess_bulk"),
                latent_min_ess_tail=d.get("latent_min_ess_tail"), treedepth_hits=d.get("treedepth_hits"), N_train=d.get("N_train") or d.get("N"),
                held_out_sites=d.get("held_out_sites"))


def score_fold(fold, sites, meas, sy, n_draws):
    held = V.heldout_sites(fold, sites)
    out = {}
    for model in ["m1s2", "m2rn"]:
        dg = fit_diag(fold, model)
        if dg["held_out_sites"] is not None:
            assert sorted(dg["held_out_sites"]) == sorted(held), f"fold definition changed: {fold} {model}"
        D = V.comparator_draws(fold) if model == "m1s2" else V.load_draws(f"vf_{fold}_R3")
        mc = C.model_config(tu=0) if model == "m1s2" else R.model_config("R3")
        CAPTURE.clear()
        s, p, y = F.heldout_scores(D, meas, sites, sy, mc, held, n_draws=n_draws)
        assert len(CAPTURE) == len(s)
        mcse = [site_mcse(lp) for lp in CAPTURE]
        s["mcse_elpd"] = [m for m, _ in mcse]; s["ess_weights"] = [e for _, e in mcse]
        # per-draw site log densities (values unchanged), kept for the post-hoc sampler-sensitivity check (VF_01 §3)
        np.savez_compressed(V.scratch() / f"score_lp_{fold}_{model}.npz", lp=np.vstack(CAPTURE), site=s.site.to_numpy().astype(str))
        # Rao-Blackwell sanity check (R3 only): conditional mean of held-out theta vs its direct posterior draws
        if model == "m2rn":
            S0 = B.ndraws(D); pick = np.linspace(0, S0 - 1, min(n_draws, S0)).astype(int)
            Ds = {k: (np.asarray(v)[pick] if np.ndim(v) >= 1 and len(v) == S0 else v) for k, v in D.items()}
            hidx = np.flatnonzero(sites.site.isin(held).to_numpy())
            mu, sd = F.heldout_theta_conditional(Ds, sites, mc, hidx, C.rn_structures(sites))
            th = np.asarray(Ds["theta"], float)[:, hidx]
            out["rb_check"] = dict(max_abs_mean_diff=float(np.abs(mu.mean(0) - th.mean(0)).max()),
                                   median_abs_mean_diff=float(np.median(np.abs(mu.mean(0) - th.mean(0)))),
                                   median_var_ratio=float(np.median((sd ** 2).mean(0) + mu.var(0)) / np.median(th.var(0))),
                                   median_posterior_sd_theta=float(np.median(th.std(0))))
        for t in (s, p, y):
            t.insert(0, "model", model); t.insert(0, "fold", fold)
        out[model] = (s, p, y, dg)
        print(fold, model, "sites", len(s), "elpd", round(s.elpd_site.sum(), 2), "max site mcse", round(s.mcse_elpd.max(), 4), flush=True)
    return out


def clustered(d, clus):
    """Total, ratio-estimator clustered SE, per-site and its SE; catchment bootstrap and leave-one-out."""
    g = pd.DataFrame({"d": d.values, "c": clus.values}).groupby("c").agg(d=("d", "sum"), n=("d", "size"))
    K = len(g); tot = g.d.sum(); n = g.n.sum()
    se = float(np.sqrt(K / (K - 1) * ((g.d - g.n * tot / n) ** 2).sum()))
    bs = []
    for _ in range(2000):
        k = RNG.integers(0, K, K); bs.append(g.d.to_numpy()[k].sum() / g.n.to_numpy()[k].sum())
    loo = [(tot - g.d.iloc[j]) / (n - g.n.iloc[j]) for j in range(K)]
    return dict(n_sites=int(n), n_catchments=int(K), diff=float(tot), se_clustered=se, z=float(tot / se), per_site=float(tot / n),
                per_site_se=se / n, per_site_lo95=float(tot / n - 1.96 * se / n), per_site_hi95=float(tot / n + 1.96 * se / n),
                boot_per_site_lo95=float(np.quantile(bs, .025)), boot_per_site_hi95=float(np.quantile(bs, .975)),
                loo_per_site_min=float(min(loo)), loo_per_site_max=float(max(loo)))


def calib(P_, Y_):
    res = {}
    for m in ["m1s2", "m2rn"]:
        u = P_[P_.model == m].pit.to_numpy(); h = np.histogram(u, bins=10, range=(0, 1))[0] / len(u)
        ym = Y_[Y_.model == m]
        res[m] = dict(pit_deciles=h.round(4).tolist(), pit_max_dev=float(np.abs(h - .1).max()), pit_cov90=float(((u > .05) & (u < .95)).mean()),
                      siteyear_cov90=float(((ym.obs_mean >= ym.lo90) & (ym.obs_mean <= ym.hi90)).mean()), pointwise_lpd=float(P_[P_.model == m].lpd.sum()),
                      **{f"brier_{c}": float(ym[f"brier_{c}"].mean()) for c in V.THRESH},
                      **{f"pred_rate_{c}": float(ym[f"p_{c}"].mean()) for c in V.THRESH}, **{f"obs_rate_{c}": float(ym[f"obs_{c}"].mean()) for c in V.THRESH})
    return res


def main(folds, n_draws):
    t0 = time.time()
    meas, sites, sy = L.load_tables()
    cat = sites.set_index("site").mancat.fillna("__none_" + sites.set_index("site").index.to_series())
    reg = sites.set_index("site").region; net = sites.set_index("site").network
    Cp = pd.read_csv(C.DP / "rn_pairs_connected.csv"); Cp = Cp[Cp.h_km <= V.CFG["scoring"]["neighbour_km"]]
    nb = {}
    for a, b in zip(Cp.up_site, Cp.down_site):
        nb.setdefault(a, set()).add(b); nb.setdefault(b, set()).add(a)
    SS, PP, YY, FD, RB = [], [], [], [], {}
    FO = V.VF / "tables/folds"; FO.mkdir(exist_ok=True)
    for fold in folds:
        if (FO / f"{fold}_site.csv").exists():               # each fold is scored once; later runs reuse the stored scores
            tabs = [pd.read_csv(FO / f"{fold}_{k}.csv") for k in ("site", "pointwise", "siteyear")]
            o = {m: tuple(tb[tb.model == m].copy() for tb in tabs) + (fit_diag(fold, m),) for m in ["m1s2", "m2rn"]}
            o["rb_check"] = json.load(open(FO / f"{fold}_rb.json"))
            print(fold, "reused stored scores", flush=True)
        else:
            o = score_fold(fold, sites, meas, sy, n_draws)
            for j, k in enumerate(("site", "pointwise", "siteyear")):
                pd.concat([o["m1s2"][j], o["m2rn"][j]]).to_csv(FO / f"{fold}_{k}.csv", index=False)
            json.dump(o.get("rb_check"), open(FO / f"{fold}_rb.json", "w"), indent=1)
        hs = set(V.heldout_sites(fold, sites))
        for model in ["m1s2", "m2rn"]:
            s, p, y, dg = o[model]
            s["has_flow_training_nb"] = s.site.map(lambda x: bool(nb.get(x, set()) - hs))
            SS.append(s); PP.append(p); YY.append(y); FD.append({k: v for k, v in dg.items() if k != "held_out_sites"})
        RB[fold] = o.get("rb_check")
    S = pd.concat(SS); P_ = pd.concat(PP); Y = pd.concat(YY); FDt = pd.DataFrame(FD)
    for nm, t in [("site", S), ("pointwise", P_), ("siteyear", Y)]:
        t.to_csv(V.VF / "tables" / f"VF_scores_{nm}.csv", index=False)
    FDt.to_csv(V.VF / "tables/VF_validation_fits.csv", index=False)
    w = S.pivot_table(index=["fold", "site"], columns="model", values="elpd_site").dropna()
    mc_ = S.pivot_table(index=["fold", "site"], columns="model", values="mcse_elpd").reindex(w.index)
    d = (w.m2rn - w.m1s2)
    dfi = d.reset_index(); dfi.columns = ["fold", "site", "d"]
    dfi["catchment"] = dfi.site.map(cat); dfi["region"] = dfi.site.map(reg); dfi["network"] = dfi.site.map(net)
    dfi["nb"] = dfi.set_index(["fold", "site"]).index.map(S[S.model == "m2rn"].set_index(["fold", "site"]).has_flow_training_nb)
    dfi["mcse2"] = (mc_.m2rn ** 2 + mc_.m1s2 ** 2).to_numpy()
    rows, strata, cal = [], [], {}
    groups = [(f, dfi[dfi.fold == f]) for f in folds]
    if {"W1", "W2"} <= set(folds):
        groups.append(("A: W1+W2", dfi[dfi.fold.isin(["W1", "W2"])]))
    cf = [f for f in folds if f.startswith("C")]
    if len(cf) > 1:
        groups.append(("B: " + "+".join(cf), dfi[dfi.fold.isin(cf)]))
    for lab, g in groups:
        r = dict(group=lab, setting="A" if lab.startswith(("W", "A")) else "B", regions=int(g.region.nunique()), **clustered(g.d, g.catchment))
        r["mcse_diff"] = float(np.sqrt(g.mcse2.sum())); r["mcse_over_se"] = r["mcse_diff"] / r["se_clustered"]
        for k, gg in [("with_nb", g[g.nb]), ("without_nb", g[~g.nb])]:
            if len(gg) > 1:
                cc = clustered(gg.d, gg.catchment)
                strata.append(dict(group=lab, stratum=k, **cc))
        for k in ["RSN", "MS"]:
            gg = g[g.network == k]
            if len(gg) > 1:
                strata.append(dict(group=lab, stratum=f"network_{k}", **clustered(gg.d, gg.catchment)))
        fl = [lab] if lab in folds else ([f for f in folds if f.startswith("W")] if lab.startswith("A") else cf)
        cal[lab] = calib(P_[P_.fold.isin(fl)], Y[Y.fold.isin(fl)])
        yy = Y[Y.fold.isin(fl)].pivot_table(index=["fold", "siteyear_idx", "site"], columns="model", values=[f"brier_{c}" for c in V.THRESH])
        for c in V.THRESH:
            db = (yy[(f"brier_{c}", "m2rn")] - yy[(f"brier_{c}", "m1s2")]).dropna().reset_index()
            db.columns = ["fold", "sy", "site", "d"]
            cc = clustered(db.d, db.site.map(cat))
            r[f"brier_diff_{c}"] = cc["per_site"]; r[f"brier_diff_se_{c}"] = cc["per_site_se"]
        r.update({f"{k}_{m}": v for m in ["m1s2", "m2rn"] for k, v in cal[lab][m].items() if k != "pit_deciles"})
        rows.append(r)
    T = pd.DataFrame(rows); T.to_csv(V.VF / "tables/VF_validation_folds.csv", index=False)
    pd.DataFrame(strata).to_csv(V.VF / "tables/VF_validation_strata.csv", index=False)
    # adequacy (frozen)
    ad = V.CFG["adequacy"]["fold"]
    adequacy = {}
    for _, f in FDt.iterrows():
        ok = dict(divergences=(f.divergence_share or 0) <= 0.001, ebfmi=f.min_ebfmi >= ad["min_ebfmi"], rhat=max(f.hyper_max_rhat or 1, f.worst_log_scale_rhat) <= ad["max_rhat"])
        adequacy[f"{f.fold}_{f.model}"] = dict(checks=ok, pass_all=bool(all(ok.values())))
    for _, r in T[T.group.isin(folds)].iterrows():
        adequacy[f"{r.group}_mcse"] = dict(mcse_over_se=r.mcse_over_se, pass_all=bool(r.mcse_over_se < ad["mcse_share_of_clustered_se"]))
    out = dict(folds=folds, n_draws=n_draws, adequacy=adequacy, rao_blackwell_check=RB, groups=T.to_dict("records"), calibration=cal, seconds=time.time() - t0)
    # pre-declared criteria
    if "A: W1+W2" in set(T.group):
        a = T[T.group == "A: W1+W2"].iloc[0]
        calm = cal["A: W1+W2"]
        out["C4"] = dict(z=a.z, per_site=a.per_site, z_ge_2=bool(a.z >= 2),
                         pit_dev_diff=calm["m2rn"]["pit_max_dev"] - calm["m1s2"]["pit_max_dev"],
                         cov90_diff=calm["m2rn"]["pit_cov90"] - calm["m1s2"]["pit_cov90"],
                         calibration_not_worse=bool(calm["m2rn"]["pit_max_dev"] - calm["m1s2"]["pit_max_dev"] <= 0.02 and abs(calm["m2rn"]["pit_cov90"] - calm["m1s2"]["pit_cov90"]) <= 0.02))
        out["C4"]["supported"] = bool(out["C4"]["z_ge_2"] and out["C4"]["calibration_not_worse"])
        margin = a.per_site / 2
        bl = [g for g in T.group if g.startswith("B:")] or [g for g in T.group if g.startswith("C")]
        if bl:
            b = T[T.group == bl[0]].iloc[0]
            out["C5"] = dict(group=bl[0], margin=margin, per_site=b.per_site, lo95=b.per_site_lo95, hi95=b.per_site_hi95,
                             boot_lo95=b.boot_per_site_lo95, boot_hi95=b.boot_per_site_hi95,
                             no_material_advantage=bool(b.per_site_hi95 < margin), equivalence_within_margin=bool(b.per_site_lo95 > -margin and b.per_site_hi95 < margin),
                             material_loss=bool(b.per_site_hi95 < 0))
    V.save_json(out, V.VF / "diagnostics/VF_validation.json")
    pd.set_option("display.width", 250)
    print(T[["group", "n_sites", "n_catchments", "regions", "diff", "se_clustered", "z", "per_site", "per_site_lo95", "per_site_hi95", "mcse_over_se"]].round(4).to_string())
    print(pd.DataFrame(strata)[["group", "stratum", "n_sites", "diff", "se_clustered", "z", "per_site"]].round(3).to_string())
    print(json.dumps({k: out.get(k) for k in ["adequacy", "rao_blackwell_check", "C4", "C5"]}, indent=1, default=str))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--folds", nargs="+", default=["W1", "W2", "C1", "C2"]); ap.add_argument("--n_draws", type=int, default=V.CFG["scoring"]["n_draws"])
    a = ap.parse_args(); main(a.folds, a.n_draws)
