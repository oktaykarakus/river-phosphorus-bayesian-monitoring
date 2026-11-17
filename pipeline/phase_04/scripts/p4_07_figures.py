"""Phase 4 B3: publication figures (draft) from Phase 4 tables and the authoritative posterior. Read-only on earlier phases.
Palette: reference categorical slots in fixed order (RSN = blue #2a78d6, MS = orange #eb6834, aqua #1baf7a, violet #6250d6);
sequential blue ramp for magnitude. Outputs: phase_04/figures/fig{1..7}_*.png (300 dpi) + .pdf
"""
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import p4_lib as P  # noqa: E402

TA = P.TA; OUT = P.P4 / "figures"; TB = P.P4 / "tables"
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#9a9893", "#e4e3df"
NET = {"RSN": BLUE, "MS": ORANGE}
SEQ = LinearSegmentedColormap.from_list("seqblue", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])
plt.rcParams.update({"font.family": "sans-serif", "font.size": 8.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.titlesize": 9.5, "axes.titleweight": "bold", "axes.titlecolor": INK, "legend.frameon": False,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300, "figure.dpi": 150})
REG = {1: "Anglian", 2: "Midlands", 3: "NorthEast", 4: "NorthWest", 5: "SouthWest", 6: "Southern", 7: "Thames"}


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight"); fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight"); plt.close(fig)


def grid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, lw=0.6); ax.set_axisbelow(True)


def load_geo():
    import pyogrio
    G = pd.read_parquet(P.scratch() / "gs_links.parquet")
    geo = pyogrio.read_dataframe(TA / "phase_02/data_processed/frame_links_england.gpkg", columns=["id"]).set_index("id").loc[G.id]
    segs = [np.asarray(g.coords)[:, :2] / 1000 for g in geo.geometry.values]
    return G, segs


def fig1_map(G, segs, sites):
    fig, ax = plt.subplots(figsize=(6.2, 7.2))
    lw = 0.15 + 0.5 * (G.la_hat.to_numpy() - G.la_hat.min()) / (G.la_hat.max() - G.la_hat.min())
    ax.add_collection(LineCollection(segs, colors="#b9c3cf", linewidths=lw))
    for nw, mk in [("MS", "o"), ("RSN", "^")]:
        s = sites[sites.network == nw]
        ax.scatter(s.x_km, s.y_km, s=7 if nw == "MS" else 9, marker=mk, c=NET[nw], edgecolors="white", linewidths=0.3, label=f"{nw} ({len(s):,} sites)", zorder=3)
    ax.set_aspect("equal"); ax.autoscale(); ax.axis("off")
    ax.legend(loc="upper left", fontsize=8, markerscale=1.8)
    ax.set_title("Monitoring sites and the restricted river frame (G_S, 68,822 km)", loc="left")
    ax.plot([140, 240], [20, 20], color=INK2, lw=1.2); ax.text(190, 26, "100 km", ha="center", color=INK2, fontsize=7)
    save(fig, "fig1_network_map")


def fig2_composition(G, sites):
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.0))
    bins = np.linspace(0, 9, 37)
    w = G.len_england_m.to_numpy(); ax[0].hist(G.la_hat, bins=bins, weights=w / w.sum(), histtype="stepfilled", color="#e4e3df", label="G_S river length")
    for nw in ["RSN", "MS"]:
        s = sites[sites.network == nw]; ax[0].hist(s.log_area, bins=bins, weights=np.ones(len(s)) / len(s), histtype="step", lw=1.6, color=NET[nw], label=f"{nw} sites")
    ax[0].set_xlabel("log modelled upstream area (km²)"); ax[0].set_ylabel("share"); ax[0].legend(fontsize=7); ax[0].set_title("a  River size", loc="left")
    b2 = np.arange(-0.5, 30.5, 2)
    ax[1].hist(np.minimum(G.edm_n_5km, 29), bins=b2, weights=w / w.sum(), histtype="stepfilled", color="#e4e3df")
    for nw in ["RSN", "MS"]:
        s = sites[sites.network == nw]; ax[1].hist(np.minimum(s.edm_n_5km, 29), bins=b2, weights=np.ones(len(s)) / len(s), histtype="step", lw=1.6, color=NET[nw])
    ax[1].set_xlabel("storm-overflow outlets within 5 km (capped at 29)"); ax[1].set_title("b  Pressure", loc="left")
    reg = pd.DataFrame({"G_S length": G.groupby("region_idx").len_england_m.sum() / G.len_england_m.sum()})
    for nw in ["RSN", "MS"]:
        s = sites[sites.network == nw]; reg[nw] = s.groupby("region_idx").size() / len(s)
    reg.index = [REG[i] for i in reg.index]; y = np.arange(len(reg))
    ax[2].barh(y - 0.27, reg["G_S length"], height=0.25, color="#c9c8c3", label="G_S length")
    ax[2].barh(y, reg["RSN"], height=0.25, color=BLUE, label="RSN"); ax[2].barh(y + 0.27, reg["MS"], height=0.25, color=ORANGE, label="MS")
    ax[2].set_yticks(y, reg.index); ax[2].invert_yaxis(); ax[2].set_xlabel("share"); ax[2].set_title("c  Region", loc="left"); ax[2].legend(fontsize=7)
    for a in ax[:2]:
        grid(a)
    grid(ax[2], "x")
    fig.tight_layout(); save(fig, "fig2_network_composition")


def fig3_variance():
    D = P.load_post(2000)
    sh = np.asarray(D["share"]); rho_tu = np.asarray(D["rho_tu"]).ravel(); rho = np.asarray(D["rho"]).ravel()
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.0))
    lab = ["site (independent)", "water body", "river network (tail-up)"]
    for k, (l, c) in enumerate(zip(lab, [VIOLET, AQUA, BLUE])):
        ax[0].hist(sh[:, k], bins=np.linspace(0, 1, 51), histtype="stepfilled", alpha=0.85, color=c, label=l)
    ax[0].set_xlabel("share of site-level variance"); ax[0].set_ylabel("posterior draws"); ax[0].legend(fontsize=7); ax[0].set_title("a  Variance partition", loc="left")
    h = np.linspace(0, 400, 200); q = np.quantile(np.exp(-h[None, :] / rho_tu[:, None]), [0.05, 0.5, 0.95], axis=0)
    ax[1].fill_between(h, q[0], q[2], color=BLUE, alpha=0.2, lw=0); ax[1].plot(h, q[1], color=BLUE, lw=2)
    ax[1].set_xlabel("along-river distance (km)"); ax[1].set_ylabel("correlation (equal flow volume)"); ax[1].set_ylim(0, 1)
    ax[1].set_title("b  River-network (tail-up) dependence", loc="left")
    d = np.linspace(0, 120, 200); qf = np.quantile(np.exp(-0.5 * (d[None, :] / rho[:, None]) ** 2), [0.05, 0.5, 0.95], axis=0)
    ax[2].fill_between(d, qf[0], qf[2], color=AQUA, alpha=0.25, lw=0); ax[2].plot(d, qf[1], color=AQUA, lw=2)
    ax[2].set_xlabel("Euclidean distance (km)"); ax[2].set_ylabel("correlation"); ax[2].set_ylim(0, 1); ax[2].set_title("c  Smooth spatial field", loc="left")
    for a in ax:
        grid(a)
    fig.tight_layout(); save(fig, "fig3_dependence_structure")


def fig4_contrasts():
    t = pd.read_csv(TA / "phase_03c_f/addendum_T1/tables/T1_combined_delta_T.csv")
    obs = pd.read_csv(TA / "phase_03c_f/addendum_T1/tables/T1_combined_ppc_samplemean_exceed_network.csv")
    a0 = pd.read_csv(TB / "A2_A0.csv"); ut = pd.read_csv(TB / "upper_tail_T1_combined.csv")
    comb = json.load(open(TA / "phase_03c_f/addendum_T1/diagnostics/T1_combined_diagnostics.json"))
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.6), sharey=True)
    for j, c in enumerate([0.05, 0.1, 0.2]):
        r = t[t.threshold == c].iloc[0]
        o = obs[obs.threshold == c].set_index("group")
        pm, pr, nm, nr = o.loc["MS", "observed"], o.loc["RSN", "observed"], o.loc["MS", "n"], o.loc["RSN", "n"]
        se = np.sqrt(pm * (1 - pm) / nm + pr * (1 - pr) / nr)
        ps = comb["prior_sensitivity"]; pri = [v[f"Delta_T_{c}"] for k, v in ps.items() if isinstance(v, dict) and f"Delta_T_{c}" in v]
        rows = [("Raw sample-mean rates (site-years)", pm - pr, pm - pr - 1.96 * se, pm - pr + 1.96 * se, MUTED),
                ("Hierarchy + spatial field", r.m1s2_median, r.m1s2_lo95, r.m1s2_hi95, AQUA),
                ("+ river-network dependence (final)", r["median"], r.lo95, r.hi95, BLUE),
                ("final, no short-term variance in μ", *a0[a0.threshold == c][["DT_A0_median", "DT_A0_lo95", "DT_A0_hi95"]].iloc[0].tolist(), VIOLET),
                ("final, prior alternatives (range of medians)", np.median([p[1] for p in pri]), min(p[1] for p in pri), max(p[1] for p in pri), VIOLET)]
        for k, (lab, m, lo, hi, col) in enumerate(rows):
            ax[j].plot([lo, hi], [k, k], color=col, lw=2.2, solid_capstyle="round"); ax[j].plot(m, k, "o", ms=6, color=col, mec="white", mew=1)
        ax[j].axvline(0, color=INK2, lw=0.8)
        ax[j].set_title(f"c = {c:.2f} mg P/L", loc="left"); ax[j].set_xlabel("MS − RSN exceedance probability"); grid(ax[j], "x")
        ax[j].set_yticks(range(len(rows)), [x[0] for x in rows]); ax[j].invert_yaxis(); ax[j].set_xlim(-0.01, 0.27)
    fig.tight_layout(); save(fig, "fig4_network_contrasts")


def fig5_validation():
    fr = pd.read_csv(TB / "A1_fold_review.csv").sort_values("fold"); pr = pd.read_csv(TB / "A1_precision_by_K.csv")
    fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.0), gridspec_kw=dict(width_ratios=[1.2, 1]))
    for k, r in enumerate(fr.itertuples()):
        col = BLUE if r.setting == "A" else ORANGE
        ax[0].plot([r.per_site - 1.96 * r.per_site_se, r.per_site + 1.96 * r.per_site_se], [k, k], color=col, lw=2.2, solid_capstyle="round")
        ax[0].plot(r.per_site, k, "o", color=col, mec="white", ms=6)
        ax[0].text(r.per_site + 1.96 * r.per_site_se + 0.004, k, f"{r.sites} sites / {r.catchments} catchments", va="center", fontsize=7, color=INK2)
    ax[0].axvline(0, color=INK2, lw=0.8)
    ax[0].set_yticks(range(len(fr)), [f"{f} ({'A: within monitored networks' if s == 'A' else 'B: unmonitored catchments'})" for f, s in zip(fr.fold, fr.setting)])
    ax[0].invert_yaxis(); ax[0].set_xlabel("held-out log score per site, with − without river network"); ax[0].set_title("a  Predictive gain by setting", loc="left")
    ax[0].set_xlim(-0.06, 0.2); grid(ax[0], "x")
    ax[1].fill_between(pr.K_catchments, pr.lower95_if_true_equals_C1, pr.upper95_if_true_equals_C1, color=ORANGE, alpha=0.2, lw=0)
    ax[1].axhline(pr.margin_half_setting_A_gain.iloc[0], color=BLUE, ls="--", lw=1); ax[1].text(18, pr.margin_half_setting_A_gain.iloc[0] + 0.002, "½ setting-A gain", color=INK2, fontsize=7)
    ax[1].axhline(0, color=INK2, lw=0.8); ax[1].set_xlabel("held-out catchments (setting B)"); ax[1].set_ylabel("95% interval, per-site difference")
    ax[1].set_title("b  Precision for unmonitored catchments", loc="left"); grid(ax[1])
    fig.tight_layout(); save(fig, "fig5_predictive_validation")


def fig6_selection():
    g = pd.read_csv(TB / "A2_selection_grid.csv"); pr = pd.read_csv(TB / "A2_PR_by_psi.csv"); bl = pd.read_csv(TB / "A3_river_length_psi.csv")
    fig, ax = plt.subplots(1, 2, figsize=(9.0, 3.2))
    c = 0.1
    for k, col in [(1.0, BLUE), (0.5, AQUA), (0.25, VIOLET)]:
        d = g[(g.threshold == c) & (g.kappa == k)].sort_values("psi")
        ax[0].fill_between(d.psi, d.DS_lo95, d.DS_hi95, color=col, alpha=0.15, lw=0); ax[0].plot(d.psi, d.DS_median, color=col, lw=2, marker="o", ms=4, label=f"κ = {k}")
    ax[0].axhline(0, color=INK2, lw=0.8); ax[0].axvspan(-0.2, 0.2, color="#f4f3f0", zorder=0)
    ax[0].set_xlabel("ψ: RSN level minus comparable-reach level (log scale)"); ax[0].set_ylabel("siting contrast Δˢ_T (c = 0.10)")
    ax[0].set_title("a  Network contrast under unobserved selection", loc="left"); ax[0].legend(fontsize=7, loc="upper left"); grid(ax[0])
    for cc, col in [(0.05, AQUA), (0.1, BLUE), (0.2, VIOLET)]:
        d = pr[pr.threshold == cc].sort_values("psi"); b = bl[bl.threshold == cc].sort_values("psi")
        ax[1].plot(d.psi, d.PR_median, color=col, lw=2, label=f"{cc:.2f} mg/L: target site-years")
        ax[1].fill_between(b.psi, b.lo95, b.hi95, color=col, alpha=0.15, lw=0); ax[1].plot(b.psi, b["median"], color=col, lw=1.4, ls="--")
    ax[1].axvspan(-0.2, 0.2, color="#f4f3f0", zorder=0)
    ax[1].set_xlabel("ψ (log scale)"); ax[1].set_ylabel("RSN-level exceedance"); ax[1].set_title("b  Reference level: site-years (solid), river length (dashed)", loc="left")
    ax[1].legend(fontsize=7); grid(ax[1])
    fig.tight_layout(); save(fig, "fig6_selection_sensitivity")


def fig7_river_length(G, segs):
    LM = pd.read_parquet(P.scratch() / "gs_link_mean_pi.parquet").set_index("id").loc[G.id]
    st = pd.read_csv(TB / "A3_river_length_strata.csv")
    fig = plt.figure(figsize=(10.5, 6.6)); a0 = fig.add_axes([0.0, 0.0, 0.55, 1.0]); a1 = fig.add_axes([0.66, 0.12, 0.32, 0.8])
    v = LM["pi_0.1"].to_numpy(); o = np.argsort(v)
    lc = LineCollection([segs[i] for i in o], array=v[o], cmap=SEQ, linewidths=0.35, norm=plt.Normalize(0, 1)); a0.add_collection(lc)
    a0.set_aspect("equal"); a0.autoscale(); a0.axis("off")
    cax = fig.add_axes([0.08, 0.06, 0.32, 0.018])
    cb = fig.colorbar(lc, cax=cax, orientation="horizontal"); cb.set_label("P(annual mean > 0.10 mg P/L), RSN level", color=INK2); cb.outline.set_visible(False)
    a0.set_title("a  Reference-level exceedance probability by link", loc="left")
    keep = ["all", "flow_connected_site", "no_flow_connected_site", "observed_water_body", "new_water_body", "no_rsn_support"] + [f"region_{k}" for k in range(1, 8)]
    d = st[(st.threshold == 0.1) & st.stratum.isin(keep)].set_index("stratum").loc[keep]
    names = {"all": "All G_S", "flow_connected_site": "flow-connected to a site", "no_flow_connected_site": "not flow-connected", "observed_water_body": "monitored water body",
             "new_water_body": "unmonitored water body", "no_rsn_support": "cells without RSN sites", **{f"region_{k}": REG[k] for k in range(1, 8)}}
    y = np.arange(len(d))
    for k, (s, r) in enumerate(d.iterrows()):
        col = BLUE if s == "all" else INK2
        a1.plot([r.lo95, r.hi95], [k, k], color=col, lw=2.2, solid_capstyle="round"); a1.plot(r["median"], k, "o", color=col, mec="white", ms=6)
        a1.text(1.0, k, f"{r.km:,.0f} km", va="center", ha="right", fontsize=7, color=MUTED)
    a1.set_yticks(y, [names[s] for s in d.index]); a1.invert_yaxis(); a1.set_xlim(0, 1.0); a1.set_xlabel("river-length-weighted exceedance, c = 0.10")
    a1.set_title("b  B̄(0.10) by stratum (95% CrI)", loc="left"); grid(a1, "x")
    save(fig, "fig7_river_length")


def main():
    sites = pd.read_parquet(TA / "phase_03a/data_processed/sites.parquet")
    G, segs = load_geo()
    fig1_map(G, segs, sites); fig2_composition(G, sites); fig3_variance(); fig4_contrasts(); fig5_validation()
    if (TB / "A3_river_length_psi.csv").exists():
        fig6_selection(); fig7_river_length(G, segs)
    print(sorted(p.name for p in OUT.glob("*.png")))


if __name__ == "__main__":
    main()
