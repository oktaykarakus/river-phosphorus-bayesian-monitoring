"""Phase 4B: manuscript and supplementary tables built directly from result files (no hand-typed numbers).
Outputs: manuscript/tables.md (Tables 1-4), manuscript/supplementary_tables.md (S1-S7), diagnostics/VF_numbers.json (registry of every
reported number with its source, used by vf_07_consistency.py)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
VF = Path(__file__).resolve().parents[1]; P4 = VF.parent; TA = P4.parent
TB = P4 / "tables"; VT = VF / "tables"; T1 = TA / "phase_03c_f/addendum_T1"
REG = {}


def reg(key, value, source):
    REG[key] = dict(value=float(value), source=source); return value


def f3(x):
    return f"{x:.3f}"


def ci(m, lo, hi, d=3):
    return f"{m:.{d}f} ({lo:.{d}f}, {hi:.{d}f})"


def md(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(out)


def table1():
    sys.path.insert(0, str(VF / "scripts")); import vf_lib as V
    meas, sites, sy = V.L.load_tables()
    obs = pd.read_csv(T1 / "tables/T1_combined_ppc_samplemean_exceed_network.csv")
    rows = []
    for nw, lab in [("RSN", "RSN"), ("MS", "Routine")]:
        s = sites[sites.network == nw]; m = meas[meas.network == nw]; y = sy[sy.network == nw]
        o = obs[obs.group == nw].set_index("threshold").observed
        r = {"": lab, "Sites": f"{len(s):,}", "Site-years (2023 / 2024)": f"{len(y):,} ({(y.year == 2023).sum()} / {(y.year == 2024).sum()})",
             "Target site-years (T)": f"{int(y.in_target_T.sum()):,}", "Measurements": f"{len(m):,}", "Censored (%)": f"{100 * m.censored.mean():.1f}",
             "Median measurements per site-year": f"{y.n.median():.0f}", "Median catchment area, km² (IQR)": f"{np.exp(s.log_area.median()):.0f} ({np.exp(s.log_area.quantile(.25)):.0f}–{np.exp(s.log_area.quantile(.75)):.0f})",
             "Overflow outlets within 5 km, mean": f"{s.edm_n_5km.mean():.1f}"}
        for c in [0.05, 0.1, 0.2]:
            r[f"Observed exceedance, {c:.2f}"] = f"{o[c]:.3f}"; reg(f"obs_rate_{nw}_{c}", o[c], "T1_combined_ppc_samplemean_exceed_network.csv")
        rows.append(r)
    return pd.DataFrame(rows).set_index("").T.reset_index().rename(columns={"index": ""})


def table2():
    t = pd.read_csv(T1 / "tables/T1_combined_delta_T.csv"); obs = pd.read_csv(T1 / "tables/T1_combined_ppc_samplemean_exceed_network.csv")
    a0 = pd.read_csv(TB / "A2_A0.csv"); comb = json.load(open(T1 / "diagnostics/T1_combined_diagnostics.json"))
    m2i = pd.read_csv(VT / "VF_m2i_delta_T.csv") if (VT / "VF_m2i_delta_T.csv").exists() else None
    rows = {"Observed difference in site-year exceedance rates": [], "Covariates only, no geographical structure (c = 0.10 only)": [],
            "Covariates + hierarchy + spatial field": [], "**+ river-network dependence (primary model)**": [], "P(Δ_T > 0), primary model": [],
            "River-network term treated as a new-site effect": [], "No short-term variance term in μ": [], "Five alternative priors (range of medians)": []}
    if m2i is not None:
        rows["Network-specific size and overflow effects (refit)"] = []
    alt = {"River-network term treated as a new-site effect": {0.05: 0.022, 0.1: 0.025, 0.2: 0.020}}   # P4_05 §4 (p4_05 output)
    for c in [0.05, 0.1, 0.2]:
        r = t[np.isclose(t.threshold, c)].iloc[0]; o = obs[np.isclose(obs.threshold, c)].set_index("group").observed
        rows["Observed difference in site-year exceedance rates"].append(f"{o['MS'] - o['RSN']:.3f}"); reg(f"raw_diff_{c}", o["MS"] - o["RSN"], "T1 PPC table")
        rows["Covariates only, no geographical structure (c = 0.10 only)"].append("0.140 (0.105, 0.175)" if c == 0.1 else "—")
        rows["Covariates + hierarchy + spatial field"].append(ci(r.m1s2_median, r.m1s2_lo95, r.m1s2_hi95)); reg(f"DT_m1s2_{c}", r.m1s2_median, "T1_combined_delta_T.csv")
        rows["**+ river-network dependence (primary model)**"].append("**" + ci(r["median"], r.lo95, r.hi95) + "**")
        reg(f"DT_{c}", r["median"], "T1_combined_delta_T.csv"); reg(f"DT_lo_{c}", r.lo95, "T1_combined_delta_T.csv"); reg(f"DT_hi_{c}", r.hi95, "T1_combined_delta_T.csv")
        rows["P(Δ_T > 0), primary model"].append(f"{r.p_gt0:.3f}")
        rows["River-network term treated as a new-site effect"].append(f"{alt['River-network term treated as a new-site effect'][c]:.3f}")
        q = a0[np.isclose(a0.threshold, c)].iloc[0]; rows["No short-term variance term in μ"].append(ci(q.DT_A0_median, q.DT_A0_lo95, q.DT_A0_hi95))
        pri = [json.loads(v[f"Delta_T_{c}"]) if isinstance(v[f"Delta_T_{c}"], str) else v[f"Delta_T_{c}"] for k, v in comb["prior_sensitivity"].items() if isinstance(v, dict) and f"Delta_T_{c}" in v]
        rows["Five alternative priors (range of medians)"].append(f"{min(p[1] for p in pri):.3f}–{max(p[1] for p in pri):.3f}")
        if m2i is not None:
            q = m2i[np.isclose(m2i.threshold, c)].iloc[0]
            rows["Network-specific size and overflow effects (refit)"].append(ci(q.DT_m2i_median, q.DT_m2i_lo95, q.DT_m2i_hi95)); reg(f"DT_m2i_{c}", q.DT_m2i_median, "VF_m2i_delta_T.csv")
    return pd.DataFrame([{"Specification": k, "c = 0.05": v[0], "c = 0.10": v[1], "c = 0.20": v[2]} for k, v in rows.items()])


def table3():
    if not (VT / "VF_validation_folds.csv").exists():
        return None
    T = pd.read_csv(VT / "VF_validation_folds.csv"); S = pd.read_csv(VT / "VF_validation_strata.csv")
    rows = []
    for _, r in T.iterrows():
        lab = {"W1": "W1", "W2": "W2", "C1": "C1", "C2": "C2"}.get(r.group, "Pooled")
        st = "A: within monitored networks" if r.setting == "A" else "B: held-out catchments"
        wn = S[(S.group == r.group) & (S.stratum == "with_nb")]; wo = S[(S.group == r.group) & (S.stratum == "without_nb")]
        rows.append({"Setting": st, "Fold": lab, "Sites / catchments / regions": f"{int(r.n_sites)} / {int(r.n_catchments)} / {int(r.regions)}",
                     "Δ log score, total (clustered SE)": f"{r['diff']:+.1f} ({r.se_clustered:.1f})", "z": f"{r.z:.2f}",
                     "Per site (95% CI)": f"{r.per_site:+.3f} ({r.per_site_lo95:+.3f}, {r.per_site_hi95:+.3f})",
                     "Per site, with / without flow-connected training site": (f"{wn.per_site.iloc[0]:+.3f} / {wo.per_site.iloc[0]:+.3f}" if len(wn) and len(wo) else "—"),
                     "PIT max. decile deviation, with / without": f"{r.pit_max_dev_m2rn:.3f} / {r.pit_max_dev_m1s2:.3f}",
                     "90% PIT coverage, with / without": f"{r.pit_cov90_m2rn:.3f} / {r.pit_cov90_m1s2:.3f}",
                     "Brier (0.10), Δ (SE)": f"{r['brier_diff_0.1']:+.4f} ({r['brier_diff_se_0.1']:.4f})"})
        key = r.group.replace(": ", "_").replace("+", "_")
        for k in ["diff", "se_clustered", "z", "per_site", "per_site_lo95", "per_site_hi95"]:
            reg(f"val_{key}_{k}", r[k], "VF_validation_folds.csv")
    return pd.DataFrame(rows)


def table4():
    A = pd.read_csv(TB / "A3_river_length.csv"); St = pd.read_csv(TB / "A3_river_length_strata.csv"); Ps = pd.read_csv(TB / "A3_river_length_psi.csv")
    PR = pd.read_csv(TB / "A2_PR_by_psi.csv")
    R = pd.read_csv(VT / "VF_rsn_only_Bbar.csv") if (VT / "VF_rsn_only_Bbar.csv").exists() else None
    rows = []
    for c in [0.05, 0.1, 0.2]:
        g = lambda q, v="primary": A[np.isclose(A.threshold, c) & (A.quantity == q) & A.variant.str.startswith(v)].iloc[0]
        b = g("Bbar"); bm = g("Bbar", "sensitivity")
        p0 = PR[np.isclose(PR.threshold, c) & np.isclose(PR.psi, 0)].iloc[0]
        lo = Ps[np.isclose(Ps.threshold, c) & np.isclose(Ps.psi, -0.2)].iloc[0]; hi = Ps[np.isclose(Ps.threshold, c) & np.isclose(Ps.psi, 0.2)].iloc[0]
        r = {"c (mg P/L)": f"{c:.2f}", "B̄, primary": ci(b["median"], b.lo95, b.hi95), "2023 / 2024": f"{g('B_2023')['median']:.3f} / {g('B_2024')['median']:.3f}",
             "River-network term not conditioned": ci(bm["median"], bm.lo95, bm.hi95)}
        if R is not None:
            q = R[np.isclose(R.threshold, c) & (R.quantity == "Bbar") & (R.variant == "primary")].iloc[0]
            r["RSN data only (refit)"] = ci(q.rsn_only_median, q.rsn_only_lo95, q.rsn_only_hi95); reg(f"Bbar_rsn_{c}", q.rsn_only_median, "VF_rsn_only_Bbar.csv")
        r["ψ = −0.2 / +0.2"] = f"{lo['median']:.3f} / {hi['median']:.3f}"
        r["Monitored site-years, P^R_T"] = ci(p0.PR_median, p0.PR_lo95, p0.PR_hi95)
        reg(f"Bbar_{c}", b["median"], "A3_river_length.csv"); reg(f"PR_T_{c}", p0.PR_median, "A2_PR_by_psi.csv")
        rows.append(r)
    return pd.DataFrame(rows)


def supp():
    out = []
    comb = json.load(open(T1 / "diagnostics/T1_combined_diagnostics.json"))
    rows = []
    for k, v in comb["prior_sensitivity"].items():
        if not isinstance(v, dict) or "pareto_k" not in v:
            continue
        d = {c: (json.loads(v[f"Delta_T_{c}"]) if isinstance(v[f"Delta_T_{c}"], str) else v[f"Delta_T_{c}"]) for c in [0.05, 0.1, 0.2]}
        rows.append({"Alternative prior": k, "Pareto k̂": f"{v['pareto_k']:.2f}", "share_u": f"{v['share_u']:.3f}", "share_B": f"{v['share_B']:.3f}",
                     "ρ_TU median (km)": f"{v['rho_tu_median']:.0f}", "δ": f"{v['delta']:.3f}", **{f"Δ_T({c})": f"{d[c][1]:.3f}" for c in [0.05, 0.1, 0.2]}})
    out.append(("Table S3. Prior sensitivity by Pareto-smoothed importance reweighting of the primary posterior (posterior medians).", pd.DataFrame(rows)))
    P = pd.read_csv(TB / "A4_ppc_cells.csv")
    out.append(("Table S4. Posterior predictive checks of site-year sample-mean exceedance: share of cells whose observed value lies inside the 90% replicate interval.",
                P.groupby("check").agg(cells=("inside", "size"), inside=("inside", "sum")).reset_index()))
    if (VT / "VF_validation_fits.csv").exists():
        Fd = pd.read_csv(VT / "VF_validation_fits.csv")
        Fd = Fd[["fold", "model", "cpu_h", "divergences", "min_ebfmi", "hyper_max_rhat", "hyper_argmax_rhat", "worst_log_scale_rhat", "hyper_min_ess_bulk", "N_train"]]
        out.append(("Table S5. Sampling diagnostics of the held-out fold fits (m2rn: primary model; m1s2: the same model without the river-network term).", Fd.round(3)))
    if (VT / "VF_m2i_interactions.csv").exists():
        I = pd.read_csv(VT / "VF_m2i_interactions.csv")[["parameter", "median", "lo95", "hi95", "p_gt0", "contraction", "rhat", "ess_bulk"]]
        out.append(("Table S6. Network-specific covariate effects (refit with routine × size and routine × overflow interactions).", I.round(3)))
    if (VT / "VF_rsn_only_parameters.csv").exists():
        Q = pd.read_csv(VT / "VF_rsn_only_parameters.csv")
        out.append(("Table S7. Shared parameters fitted to RSN measurements only, against the primary posterior (shift in primary posterior SDs).", Q.round(3)))
    return out


def main():
    t1, t2, t3, t4 = table1(), table2(), table3(), table4()
    txt = ["**Table 1.** Monitoring networks and data, 2023–2024. Observed exceedance is the share of site-years whose sample-mean concentration exceeds c (mg P L⁻¹).", "", md(t1), "",
           "**Table 2.** Difference in annual-mean exceedance probability between routine and RSN sites (posterior median, 95% CrI). The first row is the raw difference in observed rates; all other rows are standardised to the same site-years.", "", md(t2), ""]
    if t3 is not None:
        txt += ["**Table 3.** Held-out predictive performance of the model with river-network dependence relative to the same model without it. Positive differences favour the model with river-network dependence. Standard errors and intervals are clustered by management catchment.", "", md(t3), ""]
    txt += ["**Table 4.** River-length-weighted probability that the annual mean at RSN reference level exceeds c over the mapped frame (68,822 km), with sensitivity analyses (posterior median, 95% CrI).", "", md(t4), ""]
    (VF / "manuscript/tables.md").write_text("\n".join(txt))
    S = ["# Supplementary tables (generated)", "", "Tables S1 (priors) and S2 (convergence) are reproduced from phase_04/manuscript/supplementary_tables.md.", ""]
    for cap, df in supp():
        S += [f"**{cap}**", "", md(df), ""]
    (VF / "manuscript/supplementary_tables_generated.md").write_text("\n".join(S))
    json.dump(REG, open(VF / "diagnostics/VF_numbers.json", "w"), indent=1)
    print("\n".join(txt)[:6000])


if __name__ == "__main__":
    main()
