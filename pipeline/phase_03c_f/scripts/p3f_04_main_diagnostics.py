"""Phase 3C-F: full diagnostics of the corrected full-data R3 fit (main_R3).

Gates (config main_gates): R-hat < 1.01 and bulk / tail ESS >= 400 for every hyperparameter (incl. shares, sigma_u, sigma_B,
sigma_tu, log rho_tu, log sigma_site) and Delta_T at 0.05 / 0.10 / 0.20; divergences <= 0.1%; E-BFMI >= 0.3 per chain; chain-median
spread of Delta_T <= 0.5 posterior SD; posterior predictive calibration (Phase 3B p3b_ppc, read-only).
Also: Delta_T vs Phase 3C pilot, Phase 3C-R R1 screen and M1-S2; delta; variance shares; HSGP range rule; catchment-stratified flow
statistic; prior sensitivity by PSIS for rho_tu (2 alternatives) and the variance priors (3 alternatives), with Delta_T reweighted.
Helper functions are imported read-only from phase_03c_r/scripts/p3r_04_full_diagnostics.py. Outputs only in phase_03c_f.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as st

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import p3f_lib as L  # noqa: E402
import p3r_04_full_diagnostics as R4  # noqa: E402  (phase_03c_r, read-only helpers)
import p3b_ppc as P  # noqa: E402
import p3c_06_hsgp_accuracy as H  # noqa: E402

B = L.B; C = L.C
TAG = "main_R3"
HYP = ["alpha", "gamma", "delta_bar", "sigma_site", "share", "sigma_u", "sigma_B", "sigma_A", "sigma_C", "sigma_w", "rho", "sigma_tu", "sigma_om", "sigma",
       "lambda", "tau_sig", "sigma_e", "tau_s", "beta", "harm"]


def wquant(v, w, p):
    o = np.argsort(v); cw = np.cumsum(w[o]); return float(v[o][min(len(v) - 1, np.searchsorted(cw, p))])


def psis(lw):
    import arviz as az
    lw_s, k = az.psislw(lw)
    w = np.exp(lw_s - lw_s.max()); return w / w.sum(), float(k)


def main():
    meas, sites, sy = L.load_tables()
    mc = L.R.model_config("R3")
    D = L.load_draws(TAG)
    sd = json.load(open(L.ROOT / "diagnostics" / f"{TAG}_sampling_diagnostics.json"))
    nch = sd["chains"]; S = B.ndraws(D); ch = np.repeat(np.arange(nch), S // nch)
    g = L.CFG["main_gates"]
    summ = pd.read_csv(L.ROOT / "tables" / f"{TAG}_posterior_summary.csv", index_col=0)
    rows = summ.loc[[i for i in summ.index if i.split("[")[0] in HYP]]
    diag = {i: dict(r_hat=float(r.r_hat), ess_bulk=float(r.ess_bulk), ess_tail=float(r.ess_tail)) for i, r in rows.iterrows()}
    for v in ["rho_tu", "sigma_u", "sigma_B", "sigma_tu", "sigma_site"]:
        diag[f"log_{v}"] = R4.rhat_ess(np.log(B.get1(D, v)), nch)
    dT = R4.delta_draws(D, sites, sy, mc)
    for c in L.THRESH:
        diag[f"Delta_T_{c}"] = R4.rhat_ess(dT[c], nch)
    spread = {}
    for c in L.THRESH:
        med = [float(np.median(dT[c][ch == k])) for k in range(nch)]
        spread[str(c)] = dict(chain_medians=med, range=float(max(med) - min(med)), posterior_sd=float(dT[c].std()),
                              ok=bool(max(med) - min(med) <= g["delta_T_chain_median_range_max_sd"] * dT[c].std()))
    worst_r = max(diag, key=lambda k: diag[k]["r_hat"]); worst_b = min(diag, key=lambda k: diag[k]["ess_bulk"]); worst_t = min(diag, key=lambda k: diag[k]["ess_tail"])
    gates = dict(rhat=dict(value=diag[worst_r]["r_hat"], at=worst_r, ok=diag[worst_r]["r_hat"] < g["max_rhat"]),
                 ess_bulk=dict(value=diag[worst_b]["ess_bulk"], at=worst_b, ok=diag[worst_b]["ess_bulk"] >= g["min_ess_bulk"]),
                 ess_tail=dict(value=diag[worst_t]["ess_tail"], at=worst_t, ok=diag[worst_t]["ess_tail"] >= g["min_ess_tail"]),
                 divergences=dict(value=sd["divergences"], share=sd["divergence_share"], ok=sd["divergence_share"] <= g["max_divergence_share"]),
                 ebfmi=dict(value=min(sd["ebfmi"]), ok=min(sd["ebfmi"]) >= g["min_ebfmi_each_chain"]),
                 delta_T_chain_spread=dict(ok=all(v["ok"] for v in spread.values())),
                 latent=dict(value=sd["latent_max_rhat"], at=sd["latent_argmax_rhat"], ok=sd["latent_max_rhat"] <= g["latent_max_rhat_report"]))
    # PPC
    out_ppc, mt, et = P.ppc(D, meas, sites, sy, mc, n_draws=200, seed=L.CFG["model"]["seeds"]["ppc"])
    for k, t in et.items():
        t.to_csv(L.ROOT / "tables" / f"{TAG}_ppc_samplemean_exceed_{k}.csv", index=False)
    for k, t in mt.items():
        t.to_csv(L.ROOT / "tables" / f"{TAG}_ppc_monthly_{k}.csv")
    exn = et["network"]
    gates["ppc_samplemean_exceed_network"] = dict(inside=f"{int(exn.inside.sum())}/{len(exn)}", ok=bool(exn.inside.all()))
    gates["all_pass"] = bool(all(v["ok"] for v in gates.values() if isinstance(v, dict)))
    res = dict(tag=TAG, draws=S, chains=nch, sampling={k: sd[k] for k in ["cpu_h", "wall_s", "divergences", "divergences_by_chain", "ebfmi", "stepsize_by_chain",
                                                                          "mean_treedepth", "treedepth_hits", "mean_leapfrog", "latent_max_rhat", "latent_argmax_rhat",
                                                                          "latent_min_ess_bulk", "latent_min_ess_tail"]},
               key_diagnostics=diag, delta_T_chain_spread=spread, gates=gates, ppc=out_ppc)
    # Delta_T vs earlier posteriors
    r1 = json.load(open(L.TA / "phase_03c_r" / "diagnostics" / "screen_R1_diagnostics.json"))["delta_T"]
    dtab = []
    for c in L.THRESH:
        x = dT[c]; o = r1[str(c)]
        dtab.append(dict(threshold=c, median=float(np.median(x)), lo95=float(np.quantile(x, .025)), hi95=float(np.quantile(x, .975)), p_gt0=float((x > 0).mean()),
                         sd=float(x.std()), r1_screen_median=o["median"], r1_lo95=o["lo95"], r1_hi95=o["hi95"], p3c_median=o["p3c_median"], p3c_lo95=o["p3c_lo95"],
                         p3c_hi95=o["p3c_hi95"], m1s2_median=o["m1s2_median"], m1s2_lo95=o["m1s2_lo95"], m1s2_hi95=o["m1s2_hi95"],
                         diff_vs_p3c=float(np.median(x) - o["p3c_median"]), diff_vs_r1=float(np.median(x) - o["median"]), diff_vs_m1s2=float(np.median(x) - o["m1s2_median"])))
    res["delta_T"] = dtab
    pd.DataFrame(dtab).to_csv(L.ROOT / "tables" / f"{TAG}_delta_T.csv", index=False)
    dsens = {}
    for lab, kw in [("without_location_terms", dict(include_location=False)), ("river_term_as_new_site_effect", dict(include_rn=False))]:
        Dt = L.R.thin(D, 2000); r = C.delta_T(Dt, sites, sy, mc, **kw)
        dsens[lab] = {str(c): float(np.median(r[c]["D"])) for c in L.THRESH}
    res["delta_T_sensitivity_definition"] = dsens
    res["delta"] = dict(this=B.band(B.get1(D, "delta_bar")), p_gt0=float((B.get1(D, "delta_bar") > 0).mean()))
    # posterior summaries, variance shares
    vc = {}
    for v in ["sigma_site", "sigma_u", "sigma_B", "sigma_tu", "rho_tu", "sigma_A", "sigma_C", "sigma_w", "rho", "sigma_om", "sigma", "tau_sig", "lambda",
              "delta_bar", "sigma_e", "tau_s", "alpha", "gamma"]:
        x = B.get1(D, v)
        vc[v] = dict(median=float(np.median(x)), lo90=float(np.quantile(x, .05)), hi90=float(np.quantile(x, .95)),
                     chain_medians=[float(np.median(x[ch == k])) for k in range(nch)])
    sh = np.asarray(D["share"])
    for i, n in enumerate(["share_u", "share_B", "share_tu"]):
        vc[n] = dict(median=float(np.median(sh[:, i])), lo90=float(np.quantile(sh[:, i], .05)), hi90=float(np.quantile(sh[:, i], .95)),
                     chain_medians=[float(np.median(sh[ch == k, i])) for k in range(nch)])
    res["posterior"] = vc
    names = ["sigma_site", "sigma_u", "sigma_B", "sigma_tu", "rho_tu", "sigma_C", "sigma_A", "sigma_w", "rho", "sigma_om", "delta_bar"]
    M = np.column_stack([np.log(B.get1(D, v)) if v != "delta_bar" else B.get1(D, v) for v in names] + [dT[0.1]])
    res["posterior_correlation"] = pd.DataFrame(np.corrcoef(M.T), index=names + ["Delta_T_0.1"], columns=names + ["Delta_T_0.1"]).round(3).to_dict()
    # Delta_T dependence on weakly identified components (posterior bins)
    dep = {}
    for lab, x in [("share_u", sh[:, 0]), ("share_B", sh[:, 1]), ("log_rho_tu", np.log(B.get1(D, "rho_tu")))]:
        q = np.quantile(x, [0, .2, .4, .6, .8, 1])
        b = np.clip(np.searchsorted(q, x, side="right") - 1, 0, 4)
        dep[lab] = {f"q{k}": {str(c): float(np.median(dT[c][b == k])) for c in L.THRESH} for k in range(5)}
        dep[lab]["corr_with_Delta_T_0.1"] = float(np.corrcoef(x, dT[0.1])[0, 1])
    res["delta_T_by_weak_component_quintile"] = dep
    # flow-connectivity (stratified), HSGP rule
    PS = B.PairSets(sites); mancat = sites.mancat.astype(object).to_numpy()
    same_c = np.array([isinstance(mancat[i], str) and mancat[i] == mancat[j] for i, j in zip(PS.a, PS.b)])
    cc = C.site_components(L.R.thin(D, 200), sites, mc)
    res["flow_decomposition"] = {"vB+u": R4.flow_decomp(PS, cc["vB"] + cc["u"], same_c), "r = vC+vB+u": R4.flow_decomp(PS, cc["vC"] + cc["vB"] + cc["u"], same_c),
                                 "raw q = vC+vB+u+v_tu": R4.flow_decomp(PS, cc["vC"] + cc["vB"] + cc["u"] + cc["rn"], same_c)}
    rho = B.get1(D, "rho"); q95 = float(np.quantile(rho, .95)); x = sites.x_km.to_numpy(); y = sites.y_km.to_numpy()
    dd = np.sqrt((x[:, None] - x[None]) ** 2 + (y[:, None] - y[None]) ** 2)
    res["hsgp"] = dict(rho_q50=float(np.median(rho)), rho_q95=q95, rule_le_55=q95 <= 55,
                       max_abs_err_at_q95=float(np.abs(H.approx_corr(x, y, q95) - np.exp(-0.5 * dd ** 2 / q95 ** 2)).max()))
    # prior sensitivity (PSIS)
    dj = json.load(open(L.scratch_dir() / "inputs" / "data.json"))
    sdu, sdb, sdt, ra, rb = dj["sigma_u_sd"], dj["sigma_B_sd"], dj["sigma_tu_sd"], dj["rho_tu_a"], dj["rho_tu_b"]
    su, sb, stu, rt, ss = B.get1(D, "sigma_u"), B.get1(D, "sigma_B"), B.get1(D, "sigma_tu"), B.get1(D, "rho_tu"), B.get1(D, "sigma_site")
    lhn = lambda s, sc: st.halfnorm.logpdf(s, scale=sc)  # noqa: E731
    orig_var = lhn(su, sdu) + lhn(sb, sdb) + lhn(stu, sdt) + 5 * np.log(ss) - np.log(su) - np.log(sb) - np.log(stu)   # density in (sigma_site, pi1, pi2)
    a1, b1 = R4.invgamma_from_quantiles(5.0, 0.02, 300.0, 0.10)
    alts = {"rho_tu: InvGamma P(rho>300 km)=0.10": st.invgamma.logpdf(rt, a1, scale=b1) - st.invgamma.logpdf(rt, ra, scale=rb),
            "rho_tu: log-uniform [2, 5000] km": np.where((rt >= 2) & (rt <= 5000), -np.log(rt), -np.inf) - st.invgamma.logpdf(rt, ra, scale=rb),
            "variance: half-normal scales x 0.5": lhn(su, .5 * sdu) + lhn(sb, .5 * sdb) + lhn(stu, .5 * sdt) - (lhn(su, sdu) + lhn(sb, sdb) + lhn(stu, sdt)),
            "variance: half-normal scales x 2": lhn(su, 2 * sdu) + lhn(sb, 2 * sdb) + lhn(stu, 2 * sdt) - (lhn(su, sdu) + lhn(sb, sdb) + lhn(stu, sdt)),
            "variance: Dirichlet(2,2,2) shares x half-normal total SD": lhn(ss, np.sqrt(sdu ** 2 + sdb ** 2 + sdt ** 2)) + (np.log(120.0) + np.log(sh).sum(axis=1)) - orig_var}
    sens = dict(prior_sds=dict(sigma_u=sdu, sigma_B=sdb, sigma_tu=sdt, rho_tu_invgamma=[ra, rb]), alt_invgamma=[a1, b1])
    for nm, lw in alts.items():
        w, k = psis(np.asarray(lw, float))
        sens[nm] = dict(pareto_k=k, reliable=k <= L.CFG["identifiability"]["pareto_k_max"], ess_is=float(1 / (w ** 2).sum()),
                        sigma_u=wquant(su, w, .5), sigma_B=wquant(sb, w, .5), sigma_tu=wquant(stu, w, .5), share_u=wquant(sh[:, 0], w, .5),
                        share_B=wquant(sh[:, 1], w, .5), rho_tu_median=wquant(rt, w, .5), rho_tu_q95=wquant(rt, w, .95), delta=wquant(B.get1(D, "delta_bar"), w, .5),
                        **{f"Delta_T_{c}": [wquant(dT[c], w, .025), wquant(dT[c], w, .5), wquant(dT[c], w, .975)] for c in L.THRESH})
    res["prior_sensitivity"] = sens
    np.savez(L.scratch_dir() / f"{TAG}_delta_T_draws.npz", **{str(c): dT[c] for c in L.THRESH})
    L.save_json(res, L.ROOT / "diagnostics" / f"{TAG}_diagnostics.json")
    print(json.dumps({k: res[k] for k in ["gates", "delta_T_chain_spread", "delta", "hsgp"]}, indent=1, default=str))
    print(pd.DataFrame(dtab).round(4).to_string())
    print(json.dumps(sens, indent=1, default=str))


if __name__ == "__main__":
    main()
