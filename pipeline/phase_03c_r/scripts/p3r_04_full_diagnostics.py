"""Phase 3C-R: diagnostics of the corrected full-data fit (P3CR_03, P3CR_04, P3CR_06).

  * pre-specified full-data gates (config full_gates): R-hat / bulk / tail ESS of key hyperparameters (log rho_tu) AND of Delta_T at
    0.05 / 0.10 / 0.20 (computed for every posterior draw); divergences; E-BFMI per chain; chain-wise Delta_T medians vs posterior SD;
  * posterior summaries, correlations, site-level variance shares, prior-to-posterior contraction (sigma_tu, log rho_tu);
  * Delta_T (frozen definition, river term as location term) vs the Phase 3C pilot (m2rn_full) and the frozen M1-S2;
  * PPC (phase_03b p3b_ppc.ppc, read-only), residual structure by region x network;
  * catchment-stratified flow-connectivity decomposition (raw / catchment-attributable / residual) for M1-S2, the Phase 3C pilot and
    the corrected fit;
  * HSGP: posterior range vs the corrected 55-km rule, approximation error at the posterior quantiles (primary and enlarged basis);
  * rho_tu: prior sensitivity by Pareto-smoothed importance reweighting to two alternative priors.
Usage: p3r_04_full_diagnostics.py --tag full_R1 --cand R1
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as st

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import p3r_lib as L  # noqa: E402

B = L.B; A = L.A; C = L.C
sys.path.insert(0, str(L.TA / "phase_03b" / "scripts"))
import p3b_ppc as P  # noqa: E402
import p3c_06_hsgp_accuracy as H  # noqa: E402  (pure function approx_corr; phase_03c scripts on path via p3r_lib)

KEY = ["alpha", "gamma", "delta_bar", "sigma_u", "sigma_B", "sigma_A", "sigma_C", "sigma_w", "rho", "sigma_tu", "sigma_om", "sigma", "lambda", "tau_sig",
       "sigma_e", "tau_s"]


def rhat_ess(x, chains=4):
    import arviz as az
    a = np.asarray(x).reshape(chains, -1)
    return dict(r_hat=float(az.rhat(a)), ess_bulk=float(az.ess(a, method="bulk")), ess_tail=float(az.ess(a, method="tail")))


def delta_draws(D, sites, sy, mc, chunk=750):
    S = B.ndraws(D); out = {c: [] for c in L.THRESH}
    for s0 in range(0, S, chunk):
        Ds = {k: (np.asarray(v)[s0:s0 + chunk] if np.ndim(v) >= 1 and len(v) == S else v) for k, v in D.items()}
        r = C.delta_T(Ds, sites, sy, mc)
        for c in L.THRESH:
            out[c].append(r[c]["D"])
    return {c: np.concatenate(v) for c, v in out.items()}


def flow_decomp(PS, R, same_c):
    k = PS.k25; fc = PS.fc
    w_same = (k & fc & same_c).sum() / (k & fc).sum()
    rows = {"unstratified": [], "within_same_catchment": [], "across_catchments": [], "stratified": [], "catchment_composition": []}
    for r in R:
        z = (r - r.mean()) / r.std(); pr = z[PS.a] * z[PS.b]
        un = pr[k & fc].mean() - pr[k & ~fc].mean()
        wi = pr[k & fc & same_c].mean() - pr[k & ~fc & same_c].mean()
        ac = pr[k & fc & ~same_c].mean() - pr[k & ~fc & ~same_c].mean()
        stt = w_same * wi + (1 - w_same) * ac
        for nm, v in [("unstratified", un), ("within_same_catchment", wi), ("across_catchments", ac), ("stratified", stt), ("catchment_composition", un - stt)]:
            rows[nm].append(v)
    out = {k_: B.band(v) for k_, v in rows.items()}
    out["p_stratified_gt_0_10"] = float((np.array(rows["stratified"]) > 0.10).mean())
    out["share_connected_pairs_same_catchment"] = float(w_same)
    out["share_unconnected_pairs_same_catchment"] = float((k & ~fc & same_c).sum() / (k & ~fc).sum())
    return out


def invgamma_from_quantiles(lo, plo, hi, phi):
    from scipy.optimize import brentq
    def f(a):
        b = lo / st.invgamma.ppf(plo, a)
        return st.invgamma.cdf(hi, a, scale=b) - (1 - phi)
    a = brentq(f, 0.05, 50); b = lo / st.invgamma.ppf(plo, a)
    return a, b


def main(tag, cand):
    meas, sites, sy = L.load_tables()
    mc = L.model_config(cand, tu=1)
    D = L.load_draws(tag)
    S = B.ndraws(D); nch = json.load(open(L.ROOT / "diagnostics" / f"{tag}_sampling_diagnostics.json"))["chains"]
    sd = json.load(open(L.ROOT / "diagnostics" / f"{tag}_sampling_diagnostics.json"))
    res = dict(tag=tag, candidate=cand, draws=S, chains=nch, sampling={k: sd[k] for k in ["cpu_h", "wall_s", "divergences", "divergences_by_chain", "ebfmi",
                                                                                         "stepsize_by_chain", "mean_treedepth", "treedepth_hits", "mean_leapfrog",
                                                                                         "leapfrog_by_chain", "peak_child_rss_mb", "latent_max_rhat",
                                                                                         "latent_argmax_rhat", "latent_min_ess_bulk", "latent_min_ess_tail"]})
    # ---------------------------------------------------------------- gates
    g = L.CFG["full_gates"]
    summ = pd.read_csv(L.ROOT / "tables" / f"{tag}_posterior_summary.csv", index_col=0)
    keyrows = summ.loc[[i for i in summ.index if i.split("[")[0] in KEY + ["beta", "harm"] and not (cand == "R2" and i.startswith("sigma_B"))]]
    diag = {i: dict(r_hat=float(r.r_hat), ess_bulk=float(r.ess_bulk), ess_tail=float(r.ess_tail)) for i, r in keyrows.iterrows()}
    diag["log_rho_tu"] = rhat_ess(np.log(B.get1(D, "rho_tu")), nch)
    dT = delta_draws(D, sites, sy, mc)
    for c in L.THRESH:
        diag[f"Delta_T_{c}"] = rhat_ess(dT[c], nch)
    res["key_diagnostics"] = diag
    ch = L.chain_index(D, nch)
    spread = {}
    for c in L.THRESH:
        med = [float(np.median(dT[c][ch == k])) for k in range(nch)]
        spread[str(c)] = dict(chain_medians=med, range=float(max(med) - min(med)), posterior_sd=float(dT[c].std()),
                              ok=bool(max(med) - min(med) <= 0.5 * dT[c].std()))
    res["delta_T_chain_spread"] = spread
    worst_r = max(v["r_hat"] for v in diag.values()); worst_b = min(v["ess_bulk"] for v in diag.values()); worst_t = min(v["ess_tail"] for v in diag.values())
    res["gates"] = dict(rhat=dict(value=worst_r, at=max(diag, key=lambda k: diag[k]["r_hat"]), ok=worst_r < g["max_rhat_key"]),
                        ess_bulk=dict(value=worst_b, at=min(diag, key=lambda k: diag[k]["ess_bulk"]), ok=worst_b >= g["min_bulk_ess_key"]),
                        ess_tail=dict(value=worst_t, at=min(diag, key=lambda k: diag[k]["ess_tail"]), ok=worst_t >= g["min_tail_ess_key"]),
                        divergences=dict(value=sd["divergences"], share=sd["divergence_share"], ok=sd["divergence_share"] <= g["max_divergence_share"]),
                        ebfmi=dict(value=min(sd["ebfmi"]), ok=min(sd["ebfmi"]) >= g["min_ebfmi_each_chain"]),
                        delta_T_chain_spread=dict(ok=all(v["ok"] for v in spread.values())))
    res["gates"]["all_pass"] = bool(all(v["ok"] for v in res["gates"].values() if isinstance(v, dict)))
    # ---------------------------------------------------------------- posterior summaries / identifiability
    vc = {}
    for v in ["sigma_u", "sigma_B", "sigma_tu", "rho_tu", "sigma_A", "sigma_C", "sigma_w", "rho", "sigma_om", "sigma", "tau_sig", "lambda", "delta_bar",
              "sigma_e", "tau_s", "alpha", "gamma"]:
        x = B.get1(D, v)
        if x is not None and (v != "sigma_B" or cand != "R2"):
            vc[v] = dict(median=float(np.median(x)), lo90=float(np.quantile(x, .05)), hi90=float(np.quantile(x, .95)),
                         chain_medians=[float(np.median(x[ch == k])) for k in range(nch)])
    res["posterior"] = vc
    names = [v for v in ["sigma_A", "sigma_C", "sigma_B", "sigma_w", "sigma_tu", "sigma_u", "sigma_om", "rho", "rho_tu", "delta_bar"]
             if B.get1(D, v) is not None and (v != "sigma_B" or cand != "R2")]
    M = np.column_stack([np.log(B.get1(D, v)) if v.startswith(("sigma", "rho")) else B.get1(D, v) for v in names])
    res["posterior_correlation_log_scale"] = pd.DataFrame(np.corrcoef(M.T), index=names, columns=names).round(3).to_dict()
    sdn = [v for v in ["sigma_A", "sigma_C", "sigma_B", "sigma_w", "sigma_tu", "sigma_u"] if v in names]
    Ms = np.column_stack([B.get1(D, v) for v in sdn]); tot = (Ms ** 2).sum(1)
    res["site_level_variance_share"] = {n: B.band(Ms[:, k] ** 2 / tot) for k, n in enumerate(sdn)}
    exch = [v for v in ["sigma_u", "sigma_B", "sigma_tu"] if v in names]
    res["exchangeable_total_sd"] = B.band(np.sqrt(sum(B.get1(D, v) ** 2 for v in exch)))
    rcfg = C.CFG["river_network"]
    pr_s = np.abs(np.random.default_rng(1).normal(0, rcfg["sigma_sd"], 200000))
    pr_r = st.invgamma.rvs(rcfg["rho_a"], scale=rcfg["rho_b"], size=200000, random_state=2)
    res["contraction"] = {"sigma_tu": 1 - float(np.var(B.get1(D, "sigma_tu")) / np.var(pr_s)),
                          "log_rho_tu": 1 - float(np.var(np.log(B.get1(D, "rho_tu"))) / np.var(np.log(pr_r)))}
    res["rho_tu_quantiles"] = {str(q): float(np.quantile(B.get1(D, "rho_tu"), q)) for q in [.01, .05, .25, .5, .75, .95, .99]}
    res["rho_tu_prior_quantiles"] = {str(q): float(st.invgamma.ppf(q, rcfg["rho_a"], scale=rcfg["rho_b"])) for q in [.05, .5, .95]}
    # ---------------------------------------------------------------- Delta_T vs Phase 3C pilot and M1-S2
    p3c = L.thin(L.load_draws("m2rn_full", "3c"), 1000); ref = L.thin(L.load_draws("m1s2_full", "3b"), 1000)
    d3c = C.delta_T(p3c, sites, sy, C.model_config(tu=1)); dref = C.delta_T(ref, sites, sy, C.model_config())
    Dt = L.thin(D, 1000)
    dl0 = C.delta_T(Dt, sites, sy, mc, include_location=False); dlnr = C.delta_T(Dt, sites, sy, mc, include_rn=False)
    dlp = C.delta_T(Dt, sites, sy, mc)
    rows = []
    res["delta_T"] = {}
    for c in L.THRESH:
        x = dT[c]; xp = d3c[c]["D"]; xr = dref[c]["D"]
        r = dict(threshold=c, median=float(np.median(x)), lo95=float(np.quantile(x, .025)), hi95=float(np.quantile(x, .975)), p_gt0=float((x > 0).mean()),
                 sd=float(x.std()), width95=float(np.quantile(x, .975) - np.quantile(x, .025)), PR=float(np.median(dlp[c]["PR"])), PM=float(np.median(dlp[c]["PM"])),
                 p3c_median=float(np.median(xp)), p3c_lo95=float(np.quantile(xp, .025)), p3c_hi95=float(np.quantile(xp, .975)), p3c_p_gt0=float((xp > 0).mean()),
                 p3c_width95=float(np.quantile(xp, .975) - np.quantile(xp, .025)), diff_median_vs_p3c=float(np.median(x) - np.median(xp)),
                 m1s2_median=float(np.median(xr)), m1s2_lo95=float(np.quantile(xr, .025)), m1s2_hi95=float(np.quantile(xr, .975)),
                 diff_median_vs_m1s2=float(np.median(x) - np.median(xr)), without_location_terms=float(np.median(dl0[c]["D"])),
                 river_term_as_new_site_effect=float(np.median(dlnr[c]["D"])), RR=float(np.median(dlp[c]["RR"])))
        rows.append(r); res["delta_T"][str(c)] = r
    pd.DataFrame(rows).to_csv(L.ROOT / "tables" / f"{tag}_delta_T.csv", index=False)
    res["delta"] = dict(this=B.band(B.get1(D, "delta_bar")), p3c=B.band(B.get1(p3c, "delta_bar")), m1s2=B.band(B.get1(ref, "delta_bar")),
                        p_gt0=float((B.get1(D, "delta_bar") > 0).mean()))
    # ---------------------------------------------------------------- PPC and residual structure
    out, mt, et = P.ppc(D, meas, sites, sy, mc, n_draws=200)
    res["ppc"] = out
    for k, t in et.items():
        t.to_csv(L.ROOT / "tables" / f"{tag}_ppc_samplemean_exceed_{k}.csv", index=False)
    for k, t in mt.items():
        t.to_csv(L.ROOT / "tables" / f"{tag}_ppc_monthly_{k}.csv")
    comp = C.site_components(Dt, sites, mc)
    res["empirical_sd_of_site_terms"] = {k: B.band(comp[k].std(axis=1)) for k in ["vA", "vC", "vB", "w", "rn", "u"]}
    # ---------------------------------------------------------------- flow-connectivity decomposition (catchment-stratified)
    PS = B.PairSets(sites)
    mancat = sites.mancat.astype(object).to_numpy()
    same_c = np.array([isinstance(mancat[i], str) and mancat[i] == mancat[j] for i, j in zip(PS.a, PS.b)])
    c200 = C.site_components(L.thin(D, 200), sites, mc)
    p200 = C.site_components(L.thin(p3c, 200), sites, C.model_config(tu=1))
    r200 = C.site_components(L.thin(ref, 200), sites, C.model_config())
    sets = {}
    for lab, cc in [("M1-S2", r200), ("P3C M2-RN pilot (R0)", p200), (f"corrected {cand}", c200)]:
        sets[f"{lab}: raw q = vC+vB+u+v_tu"] = cc["vC"] + cc["vB"] + cc["u"] + cc["rn"]
        sets[f"{lab}: r = vC+vB+u"] = cc["vC"] + cc["vB"] + cc["u"]
        sets[f"{lab}: vB+u"] = cc["vB"] + cc["u"]
    res["flow_decomposition"] = {k: flow_decomp(PS, v, same_c) for k, v in sets.items()}
    rows = [dict(quantity=k, **{kk: (vv[1] if isinstance(vv, list) else vv) for kk, vv in v.items()},
                 stratified_lo90=v["stratified"][0], stratified_hi90=v["stratified"][2]) for k, v in res["flow_decomposition"].items()]
    pd.DataFrame(rows).to_csv(L.ROOT / "tables" / f"{tag}_flow_decomposition.csv", index=False)
    res["flow_by_river_distance_corrected_q"] = {}
    q = sets[f"corrected {cand}: raw q = vC+vB+u+v_tu"]
    for lo, hi in [(0, 10), (10, 25), (25, 60), (60, 300)]:
        kk = PS.k25 & PS.fc & (PS.net_km >= lo) & (PS.net_km < hi)
        vals = []
        for r in q:
            z = (r - r.mean()) / r.std(); pr = z[PS.a] * z[PS.b]
            vals.append(pr[kk].mean() - pr[PS.k25 & ~PS.fc].mean())
        res["flow_by_river_distance_corrected_q"][f"{lo}-{hi}"] = dict(n_pairs=int(kk.sum()), excess=B.band(vals))
    # ---------------------------------------------------------------- HSGP range rule and accuracy
    hcfg = L.CFG["hsgp"]
    rho = B.get1(D, "rho"); qq = {p: float(np.quantile(rho, p)) for p in [.01, .05, .5, .95, .99]}
    x = sites.x_km.to_numpy(); y = sites.y_km.to_numpy(); dd = np.sqrt((x[:, None] - x[None]) ** 2 + (y[:, None] - y[None]) ** 2)
    acc = []
    for lab, r in [("q50", qq[.5]), ("q95", qq[.95]), ("q99", qq[.99]), ("rule_55km", 55.0)]:
        for basis, m, cb in [("primary_1.25", None, None), ("enlarged_1.5", hcfg["enlarged_basis"]["m"], hcfg["enlarged_basis"]["boundary_factor"])]:
            E = H.approx_corr(x, y, r, m=m, c_b=cb) - np.exp(-0.5 * dd ** 2 / r ** 2)
            acc.append(dict(at=lab, rho_km=r, basis=basis, max_abs_err=float(np.abs(E).max()), max_abs_err_diag=float(np.abs(np.diag(E)).max())))
    wvar = np.asarray(Dt["w_site"]).var(0)
    res["hsgp"] = dict(rho_quantiles=qq, rho05_ge_12=qq[.05] >= hcfg["rho_min_km"], rho95_le_55=qq[.95] <= hcfg["rho95_max_km"],
                       enlarged_trigger_rho95_gt_50=qq[.95] > 50, accuracy=acc, posterior_var_w_min=float(wvar.min()),
                       max_field_cov_err_at_q99_over_min_post_var=float(np.median(B.get1(D, "sigma_w")) ** 2 *
                                                                         [a["max_abs_err"] for a in acc if a["at"] == "q99" and a["basis"] == "primary_1.25"][0] / wvar.min()))
    pd.DataFrame(acc).to_csv(L.ROOT / "tables" / f"{tag}_hsgp_accuracy.csv", index=False)
    # ---------------------------------------------------------------- rho_tu prior sensitivity (PSIS)
    import arviz as az
    a0, b0 = rcfg["rho_a"], rcfg["rho_b"]
    a1, b1 = invgamma_from_quantiles(5.0, 0.02, 300.0, 0.10)
    rt = B.get1(D, "rho_tu")
    lp0 = st.invgamma.logpdf(rt, a0, scale=b0)
    alts = {"InvGamma P(rho>300)=0.10": st.invgamma.logpdf(rt, a1, scale=b1),
            "log-uniform [2, 5000] km": np.where((rt >= 2) & (rt <= 5000), -np.log(rt), -np.inf)}
    sens = {"alt_invgamma_params": [a1, b1], "alt_invgamma_median_km": float(st.invgamma.median(a1, scale=b1))}
    for nm, lp1 in alts.items():
        lw = lp1 - lp0
        lw_s, khat = az.psislw(lw.reshape(nch, -1).T.reshape(-1) if False else lw)
        w = np.exp(lw_s - lw_s.max()); w /= w.sum()
        def wq(v, p):
            o = np.argsort(v); cw = np.cumsum(w[o]); return float(v[o][np.searchsorted(cw, p)])
        sens[nm] = dict(pareto_k=float(khat), ess_is=float(1 / (w ** 2).sum()),
                        rho_tu_median=wq(rt, .5), rho_tu_q05=wq(rt, .05), rho_tu_q95=wq(rt, .95), sigma_tu_median=wq(B.get1(D, "sigma_tu"), .5),
                        delta_median=wq(B.get1(D, "delta_bar"), .5),
                        **{f"Delta_T_{c}_median": wq(dT[c], .5) for c in L.THRESH}, **{f"Delta_T_{c}_lo95": wq(dT[c], .025) for c in L.THRESH},
                        **{f"Delta_T_{c}_hi95": wq(dT[c], .975) for c in L.THRESH})
    res["rho_tu_prior_sensitivity"] = sens
    res["corr_log_rho_tu_with"] = {k: float(np.corrcoef(np.log(rt), v)[0, 1]) for k, v in
                                   [("sigma_tu", B.get1(D, "sigma_tu")), ("delta_bar", B.get1(D, "delta_bar")), ("Delta_T_0.1", dT[0.1]),
                                    ("sigma_u", B.get1(D, "sigma_u"))]}
    np.savez(L.scratch_dir() / f"{tag}_delta_T_draws.npz", **{str(c): dT[c] for c in L.THRESH})
    L.save_json(res, L.ROOT / "diagnostics" / f"{tag}_diagnostics.json")
    print(json.dumps({k: res[k] for k in ["gates", "delta_T_chain_spread", "delta", "hsgp", "rho_tu_prior_sensitivity"]}, indent=1, default=str))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", required=True); ap.add_argument("--cand", required=True)
    a = ap.parse_args(); main(a.tag, a.cand)
