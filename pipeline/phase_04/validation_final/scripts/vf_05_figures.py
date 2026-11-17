"""Phase 4B B9: publication figures (final). Read-only on every earlier output; writes validation_final/figures/.

Figures (double-column width 174 mm unless stated; 7-8 pt type; no in-figure titles, the captions carry the message):
  Fig1 monitoring sites on the river frame (map)           Fig5 dependence structure (variance shares, correlation functions)
  Fig2 model schematic                                     Fig6 held-out predictive validation (final converged fold fits)
  Fig3 network composition                                 Fig7 sensitivity to unobserved selection
  Fig4 network contrast: raw -> standardised + robustness  Fig8 river-length-weighted reference-level exceedance
Palette: fixed categorical order (blue, orange, aqua, violet) checked by scripts/vf_palette_check.py; network identity is also carried
by marker shape (RSN triangle, routine circle) and direct labels; magnitude uses one blue ramp.
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
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

sys.dont_write_bytecode = True
VF = Path(__file__).resolve().parents[1]; P4 = VF.parent; TA = P4.parent
OUT = VF / "figures"; TB = P4 / "tables"; VT = VF / "tables"
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
CAT = [BLUE, ORANGE, AQUA]                     # violet dropped: blue-violet fails the deutan all-pairs check (VF_palette_check)
INK, INK2, MUTED, GRID, PALE = "#0b0b0b", "#52514e", "#8a8883", "#e4e3df", "#f3f2ef"
NET = {"RSN": BLUE, "MS": ORANGE}
NETLAB = {"RSN": "RSN", "MS": "Routine"}
SEQ_STEPS = ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ = LinearSegmentedColormap.from_list("seqblue", SEQ_STEPS)
MM = 1 / 25.4; W2 = 174 * MM; W1 = 84 * MM; W15 = 129 * MM
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"], "font.size": 7.5, "axes.labelsize": 7.5,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 600, "figure.dpi": 150,
                     "pdf.fonttype": 42, "ps.fonttype": 42})
REG = {1: "Anglian", 2: "Midlands", 3: "North East", 4: "North West", 5: "South West", 6: "Southern", 7: "Thames"}


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", dpi=400); fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight"); plt.close(fig)


def grid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, lw=0.5); ax.set_axisbelow(True)


def letter(ax, s, x=-0.02, y=1.02):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=9, fontweight="bold", color=INK, ha="right", va="bottom")


def sites_table():
    return pd.read_parquet(TA / "phase_03a/data_processed/sites.parquet")


def load_geo():
    import pyogrio
    G = pd.read_parquet(Path("~/track_a_scratch/phase_04/gs_links.parquet").expanduser())
    geo = pyogrio.read_dataframe(TA / "phase_02/data_processed/frame_links_england.gpkg", columns=["id"]).set_index("id").loc[G.id]
    segs = [np.asarray(g.coords)[:, :2] / 1000 for g in geo.geometry.values]
    return G, segs


def scalebar(ax, x0, y0, km=100):
    ax.plot([x0, x0 + km], [y0, y0], color=INK2, lw=1.0, solid_capstyle="butt")
    ax.text(x0 + km / 2, y0 - 6, f"{km} km", ha="center", va="top", color=INK2, fontsize=6.5)


def north(ax, x, y):
    ax.annotate("N", xy=(x, y + 28), xytext=(x, y), ha="center", va="center", fontsize=7, color=INK2,
                arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.8))


# ------------------------------------------------------------------ Fig 1
def fig1(G, segs, sites):
    fig, ax = plt.subplots(figsize=(W15, W15 * 1.18))
    lw = 0.12 + 0.55 * (G.la_hat.to_numpy() - G.la_hat.min()) / (G.la_hat.max() - G.la_hat.min())
    ax.add_collection(LineCollection(segs, colors="#bfc8d3", linewidths=lw, zorder=1))
    for nw, mk, s_ in [("MS", "o", 5.5), ("RSN", "^", 8)]:
        s = sites[sites.network == nw]
        ax.scatter(s.x_km, s.y_km, s=s_, marker=mk, c=NET[nw], edgecolors="white", linewidths=0.25, zorder=3,
                   label=f"{'Routine statutory sites' if nw == 'MS' else 'River Surveillance Network (RSN)'}  (n = {len(s):,})")
    ax.set_aspect("equal"); ax.autoscale(); ax.axis("off")
    h, l = ax.get_legend_handles_labels()
    ax.legend(h[::-1], l[::-1], loc="upper left", markerscale=1.6, handletextpad=0.3, borderaxespad=0)
    ax.text(0.0, 0.905, "Lines: mapped inland rivers in the restricted frame\n(68,822 km; width increases with catchment area)", transform=ax.transAxes,
            fontsize=6.5, color=MUTED, va="top")
    xmin = min(s[:, 0].min() for s in segs); ymin = min(s[:, 1].min() for s in segs)
    scalebar(ax, xmin + 10, ymin + 5); north(ax, xmin + 30, ymin + 45)
    save(fig, "Fig1_monitoring_networks")


# ------------------------------------------------------------------ Fig 2 (schematic)
def box(ax, x, y, w, h, text, fc="white", ec=MUTED, fs=6.6, bold_first=True, color=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.012", fc=fc, ec=ec, lw=0.7))
    lines = text.split("\n")
    n = len(lines); sp = 0.036; yy = y + h / 2 + (n - 1) * sp / 2
    for k, ln in enumerate(lines):
        ax.text(x + w / 2, yy - k * sp, ln, ha="center", va="center", fontsize=fs, color=color if k else INK,
                fontweight="bold" if (k == 0 and bold_first) else "normal")


def arrow(ax, p, q, color=INK2, ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=7, color=color, lw=0.8, ls=ls, connectionstyle=f"arc3,rad={rad}", shrinkA=1, shrinkB=1))


def fig2():
    fig = plt.figure(figsize=(W2, W2 * 0.52))
    ax = fig.add_axes([0, 0, 0.66, 1]); ax.set_xlim(-0.012, 1); ax.set_ylim(0, 1); ax.axis("off")
    fs = 6.3
    box(ax, 0.00, 0.83, 0.33, 0.15, "Measurements $y_{ij}$\nlognormal; censored at the\nobservation's own reporting limit", fc=PALE, fs=fs)
    box(ax, 0.38, 0.83, 0.30, 0.15, "Within-year terms\nseasonal harmonics; regional\nmonth-by-year shocks; $\\sigma_i$", fs=fs)
    box(ax, 0.00, 0.58, 0.33, 0.15, "Site-year level\n$\\theta_i + \\gamma_t + \\omega_{it}$ (log scale)", fs=fs)
    box(ax, 0.38, 0.58, 0.30, 0.15, "Latent annual mean\n$\\mu_{it} = \\int_0^1 \\exp\\{\\eta_{it}(\\tau) + \\sigma_i^2/2\\}\\,d\\tau$", fs=fs)
    box(ax, 0.73, 0.58, 0.25, 0.15, "Exceedance\n$\\Pr(\\mu_{it} > c)$, $c$ = 0.05,\n0.10, 0.20 mg P/L", fc=PALE, fs=fs)
    box(ax, 0.00, 0.30, 0.33, 0.18, "Site level $\\theta_i$\n$m_i + u_i + b_{w(i)} + v_{TU}(s_i)$\n($u$, $b$, $v_{TU}$ integrated out jointly)", fs=fs)
    box(ax, 0.38, 0.32, 0.30, 0.14, "Network offset $\\delta$\nat routine sites\n(sensitivity: $\\psi$, $\\kappa$)", ec=ORANGE, fs=fs)
    box(ax, 0.00, 0.02, 0.33, 0.19, "Fixed and location terms $m_i$\nsize, size$^2$, storm overflows,\ngeology; WMA and catchment\neffects; smooth field $w(s)$", fs=fs)
    box(ax, 0.38, 0.04, 0.30, 0.15, "River-network term $v_{TU}$\ntail-up exponential covariance\nalong flow paths (panel b)", ec=BLUE, fs=fs)
    box(ax, 0.73, 0.30, 0.22, 0.18, "Standardised contrast\n$\\Delta_T$: mean over $T$ of\n$\\pi_{routine} - \\pi_{RSN}$", fc=PALE, fs=fs)
    box(ax, 0.73, 0.02, 0.22, 0.19, "River-length estimate\n$\\bar B$: length-weighted\n$\\pi_{RSN}$ over\nmapped rivers", fc=PALE, fs=fs)
    arrow(ax, (0.165, 0.73), (0.165, 0.83)); arrow(ax, (0.38, 0.905), (0.33, 0.905)); arrow(ax, (0.53, 0.83), (0.53, 0.73))
    arrow(ax, (0.33, 0.655), (0.38, 0.655)); arrow(ax, (0.68, 0.655), (0.73, 0.655)); arrow(ax, (0.165, 0.48), (0.165, 0.58))
    arrow(ax, (0.165, 0.21), (0.165, 0.30)); arrow(ax, (0.45, 0.19), (0.30, 0.30)); arrow(ax, (0.38, 0.39), (0.33, 0.39), color=ORANGE)
    arrow(ax, (0.84, 0.58), (0.84, 0.48)); ax.plot([0.975, 0.975], [0.58, 0.115], color=INK2, lw=0.8); arrow(ax, (0.976, 0.115), (0.95, 0.115))
    ax.text(0.0, 1.0, "a", fontsize=9, fontweight="bold", va="top", ha="left", transform=ax.transAxes)
    bx = fig.add_axes([0.69, 0.02, 0.30, 0.96]); bx.set_xlim(0, 10); bx.set_ylim(-5.2, 10); bx.axis("off")
    segs = {"A": ((1.0, 9.0), (3.5, 6.0)), "B": ((6.5, 9.4), (3.5, 6.0)), "C": ((3.5, 6.0), (4.6, 3.2)), "D": ((8.8, 6.4), (4.6, 3.2)), "E": ((4.6, 3.2), (5.4, 0.2))}
    wid = {"A": 1.0, "B": 1.6, "C": 2.6, "D": 1.3, "E": 3.4}
    for k, ((x0, y0), (x1, y1)) in segs.items():
        bx.plot([x0, x1], [y0, y1], color="#9fb6d6", lw=wid[k] * 1.2, solid_capstyle="round", zorder=1)
        bx.annotate("", xy=(x0 + 0.62 * (x1 - x0), y0 + 0.62 * (y1 - y0)), xytext=(x0 + 0.48 * (x1 - x0), y0 + 0.48 * (y1 - y0)),
                    arrowprops=dict(arrowstyle="-|>", color="#6f8fbf", lw=0.6, mutation_scale=6))
    pts = {"$s_1$": ((1.9, 7.9), "RSN"), "$s_2$": ((5.6, 8.4), "MS"), "$s_3$": ((4.1, 4.5), "MS"), "$s_4$": ((7.3, 5.4), "RSN"), "$s_5$": ((5.15, 1.2), "RSN")}
    for k, ((x, y), nw) in pts.items():
        bx.scatter(x, y, s=30, marker="^" if nw == "RSN" else "o", c=NET[nw], edgecolors="white", linewidths=0.6, zorder=3)
        bx.text(x + 0.4, y + 0.1, k, fontsize=6.5, color=INK2, va="center")
    bx.scatter([], [], marker="^", c=BLUE, s=22, label="RSN"); bx.scatter([], [], marker="o", c=ORANGE, s=22, label="Routine")
    bx.legend(loc="center left", fontsize=6.2, handletextpad=0.2, borderaxespad=0.1, bbox_to_anchor=(0.66, 0.40))
    bx.text(7.2, 9.6, "flow", fontsize=6.2, color="#6f8fbf")
    bx.text(0.0, -0.9, "Flow-connected pairs ($s_1$–$s_3$, $s_3$–$s_5$):\ncorr $= e^{-h/\\rho_{TU}}\\sqrt{\\mathrm{flow\\ volume\\ ratio}}$", fontsize=6.2, color=INK2, va="top")
    bx.text(0.0, -3.4, "Flow-unconnected pairs ($s_1$–$s_2$, $s_3$–$s_4$):\nindependent in $v_{TU}$", fontsize=6.2, color=INK2, va="top")
    bx.text(0.0, 10.0, "b", fontsize=9, fontweight="bold", va="top")
    save(fig, "Fig2_model_schematic")


# ------------------------------------------------------------------ Fig 3
def fig3(G, sites):
    fig, ax = plt.subplots(1, 3, figsize=(W2, W2 * 0.30), gridspec_kw=dict(width_ratios=[1, 1, 1.05]))
    bins = np.linspace(0, 9, 31)
    w = G.len_england_m.to_numpy()
    ax[0].hist(G.la_hat, bins=bins, weights=w / w.sum(), histtype="stepfilled", color=GRID, label="River length")
    for nw in ["RSN", "MS"]:
        s = sites[sites.network == nw]
        ax[0].hist(s.log_area, bins=bins, weights=np.ones(len(s)) / len(s), histtype="step", lw=1.3, color=NET[nw], label=f"{NETLAB[nw]} sites")
    ax[0].set_xlabel("Upstream catchment area (km²)"); ax[0].set_ylabel("Share")
    ax[0].set_xticks(np.log([1, 10, 100, 1000]), ["1", "10", "100", "1,000"]); ax[0].legend(loc="upper right", handlelength=1.2)
    b2 = np.arange(-0.5, 30.5, 2)
    ax[1].hist(np.minimum(G.edm_n_5km, 29), bins=b2, weights=w / w.sum(), histtype="stepfilled", color=GRID)
    for nw in ["RSN", "MS"]:
        s = sites[sites.network == nw]; ax[1].hist(np.minimum(s.edm_n_5km, 29), bins=b2, weights=np.ones(len(s)) / len(s), histtype="step", lw=1.3, color=NET[nw])
    ax[1].set_xlabel("Storm-overflow outlets within 5 km"); ax[1].set_xticks([0, 10, 20, 29], ["0", "10", "20", "≥29"])
    reg = pd.DataFrame({"River length": G.groupby("region_idx").len_england_m.sum() / G.len_england_m.sum()})
    for nw in ["RSN", "MS"]:
        s = sites[sites.network == nw]; reg[nw] = s.groupby("region_idx").size() / len(s)
    reg.index = [REG[i] for i in reg.index]; y = np.arange(len(reg))
    ax[2].barh(y - 0.27, reg["River length"], height=0.25, color="#c9c8c3", label="River length")
    ax[2].barh(y, reg["RSN"], height=0.25, color=BLUE, label="RSN sites"); ax[2].barh(y + 0.27, reg["MS"], height=0.25, color=ORANGE, label="Routine sites")
    ax[2].set_yticks(y, reg.index); ax[2].invert_yaxis(); ax[2].set_xlabel("Share"); ax[2].legend(loc="lower right", handlelength=1.0)
    for a, l_ in zip(ax, "abc"):
        letter(a, l_, x=-0.08 if l_ != "c" else -0.30)
    grid(ax[0]); grid(ax[1]); grid(ax[2], "x")
    fig.tight_layout(w_pad=1.2); save(fig, "Fig3_network_composition")


# ------------------------------------------------------------------ Fig 4
def fig4():
    t = pd.read_csv(TA / "phase_03c_f/addendum_T1/tables/T1_combined_delta_T.csv")
    obs = pd.read_csv(TA / "phase_03c_f/addendum_T1/tables/T1_combined_ppc_samplemean_exceed_network.csv")
    a0 = pd.read_csv(TB / "A2_A0.csv")
    comb = json.load(open(TA / "phase_03c_f/addendum_T1/diagnostics/T1_combined_diagnostics.json"))
    m2i = pd.read_csv(VT / "VF_m2i_delta_T.csv") if (VT / "VF_m2i_delta_T.csv").exists() else None
    fig, ax = plt.subplots(1, 3, figsize=(W2, W2 * 0.36), sharey=True)
    for j, c in enumerate([0.05, 0.1, 0.2]):
        r = t[np.isclose(t.threshold, c)].iloc[0]
        o = obs[np.isclose(obs.threshold, c)].set_index("group")
        pm, pr, nm, nr = o.loc["MS", "observed"], o.loc["RSN", "observed"], o.loc["MS", "n"], o.loc["RSN", "n"]
        se = np.sqrt(pm * (1 - pm) / nm + pr * (1 - pr) / nr)
        ps = comb["prior_sensitivity"]; pri = [v[f"Delta_T_{c}"] for k, v in ps.items() if isinstance(v, dict) and f"Delta_T_{c}" in v]
        rows = [("Observed difference (site-year means)", pm - pr, pm - pr - 1.96 * se, pm - pr + 1.96 * se, MUTED, "o"),
                ("Covariates, hierarchy, spatial field", r.m1s2_median, r.m1s2_lo95, r.m1s2_hi95, MUTED, "o"),
                ("+ river-network dependence (primary)", r["median"], r.lo95, r.hi95, BLUE, "o"),
                ("Annual mean without short-term variance", *a0[np.isclose(a0.threshold, c)][["DT_A0_median", "DT_A0_lo95", "DT_A0_hi95"]].iloc[0].tolist(), INK2, "D"),
                ("Five alternative priors (range of medians)", np.median([p[1] for p in pri]), min(p[1] for p in pri), max(p[1] for p in pri), INK2, "D")]
        if m2i is not None:
            q = m2i[np.isclose(m2i.threshold, c)].iloc[0]
            rows.append(("Network-specific size and overflow effects", q.DT_m2i_median, q.DT_m2i_lo95, q.DT_m2i_hi95, INK2, "D"))
        for k, (lab, m, lo, hi, col, mk) in enumerate(rows):
            ax[j].plot([lo, hi], [k, k], color=col, lw=1.8, solid_capstyle="round"); ax[j].plot(m, k, mk, ms=4.5 if mk == "o" else 3.8, color=col, mec="white", mew=0.7)
        ax[j].axvline(0, color=INK2, lw=0.6); ax[j].axhline(2.5, color=GRID, lw=0.6, ls="-")
        ax[j].set_title(f"c = {c:.2f} mg P/L", loc="left", fontsize=7.5, color=INK, fontweight="bold")
        grid(ax[j], "x")
        ax[j].set_yticks(range(len(rows)), [x[0] for x in rows]); ax[j].set_ylim(len(rows) - 0.5, -0.6); ax[j].set_xlim(-0.01, 0.27)
        ax[j].set_xticks([0, 0.1, 0.2])
    fig.supxlabel("Difference in exceedance probability, routine − RSN", fontsize=7.5, color=INK2, x=0.62, y=0.02)
    fig.tight_layout(w_pad=0.8); save(fig, "Fig4_network_contrast")


# ------------------------------------------------------------------ Fig 5
def fig5():
    sys.path.insert(0, str(P4 / "scripts"))
    import p4_lib as P
    D = P.load_post(4000)
    sh = np.asarray(D["share"]); rho_tu = np.asarray(D["rho_tu"]).ravel(); rho = np.asarray(D["rho"]).ravel()
    fig, ax = plt.subplots(1, 3, figsize=(W2, W2 * 0.30), gridspec_kw=dict(width_ratios=[1.15, 1, 1]))
    lab = ["Independent site", "Water body", "River network (tail-up)"]
    for k, (l_, c) in enumerate(zip(lab, [INK2, INK2, BLUE])):
        q = np.quantile(sh[:, k], [0.05, 0.25, 0.5, 0.75, 0.95])
        ax[0].plot([q[0], q[4]], [k, k], color=c, lw=1.2, solid_capstyle="round"); ax[0].plot([q[1], q[3]], [k, k], color=c, lw=4, solid_capstyle="butt")
        ax[0].plot(q[2], k, "o", color="white", mec=c, ms=4.5, mew=1.2)
    ax[0].set_yticks(range(3), lab); ax[0].invert_yaxis(); ax[0].set_xlim(0, 1); ax[0].set_ylim(2.6, -0.6)
    ax[0].set_xlabel("Share of site-level variance"); grid(ax[0], "x")
    h = np.linspace(0, 400, 200); q = np.quantile(np.exp(-h[None, :] / rho_tu[:, None]), [0.05, 0.5, 0.95], axis=0)
    ax[1].fill_between(h, q[0], q[2], color=BLUE, alpha=0.18, lw=0); ax[1].plot(h, q[1], color=BLUE, lw=1.5)
    ax[1].set_xlabel("Along-river distance (km)"); ax[1].set_ylabel("Correlation (equal flow volume)"); ax[1].set_ylim(0, 1.02)
    d = np.linspace(0, 120, 200); qf = np.quantile(np.exp(-0.5 * (d[None, :] / rho[:, None]) ** 2), [0.05, 0.5, 0.95], axis=0)
    ax[2].fill_between(d, qf[0], qf[2], color=AQUA, alpha=0.22, lw=0); ax[2].plot(d, qf[1], color=AQUA, lw=1.5)
    ax[2].set_xlabel("Euclidean distance (km)"); ax[2].set_ylabel("Correlation"); ax[2].set_ylim(0, 1.02)
    for a, l_ in zip(ax, "abc"):
        letter(a, l_, x=-0.08 if l_ != "a" else -0.45);
    grid(ax[1]); grid(ax[2])
    fig.tight_layout(w_pad=1.0); save(fig, "Fig5_dependence_structure")


# ------------------------------------------------------------------ Fig 6
def fig6():
    if not (VT / "VF_validation_folds.csv").exists():
        print("Fig6: validation results not yet available"); return
    T = pd.read_csv(VT / "VF_validation_folds.csv"); St = pd.read_csv(VT / "VF_validation_strata.csv")
    js = json.load(open(VF / "diagnostics/VF_validation.json"))
    order = [g for g in ["W1", "W2", "A: W1+W2", "C1", "C2"] + [g for g in T.group if g.startswith("B:")] if g in set(T.group)]
    T = T.set_index("group").loc[order]
    fig, ax = plt.subplots(1, 2, figsize=(W2, W2 * 0.40), gridspec_kw=dict(width_ratios=[1.25, 1]))
    margin = js.get("C5", {}).get("margin")
    names = {"W1": "W1", "W2": "W2", "A: W1+W2": "Pooled", "C1": "C1", "C2": "C2"}
    # rows: a header line above each setting, so that group labels never sit on an interval
    ys, ylab, y, prev = [], [], 0.0, None
    heads = {"A": ("New sites within monitored river networks", BLUE), "B": ("Sites in held-out catchments", ORANGE)}
    for g, r in T.iterrows():
        if r.setting != prev:
            y += 0.3 if prev is None else 0.8
            ax[0].text(-0.058, y - 0.75, heads[r.setting][0], fontsize=6.4, color=heads[r.setting][1], va="center", ha="left", zorder=5,
                       bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none"))
            prev = r.setting
        ys.append(y); ylab.append(names.get(g, "Pooled")); y += 1
    for (g, r), k in zip(T.iterrows(), ys):
        col = BLUE if r.setting == "A" else ORANGE; pooled = g.startswith(("A:", "B:"))
        ax[0].plot([r.per_site_lo95, r.per_site_hi95], [k, k], color=col, lw=2.2 if pooled else 1.5, solid_capstyle="round")
        ax[0].plot(r.per_site, k, "D" if pooled else "o", color=col if pooled else "white", mec=col, mew=1.1, ms=4.5)
        ax[0].text(0.268, k, f"{int(r.n_sites)} sites, {int(r.n_catchments)} catchments", va="center", ha="right", fontsize=6.2, color=MUTED)
    if margin:
        ax[0].axvspan(-margin, margin, ymin=0, ymax=1, color=PALE, zorder=0)
    ax[0].axvline(0, color=INK2, lw=0.6)
    ax[0].set_yticks(ys, ylab); ax[0].set_ylim(max(ys) + 0.6, -0.75); ax[0].set_xlim(-0.06, 0.27)
    ax[0].set_xticks([-0.05, 0, 0.05, 0.10, 0.15])
    ax[0].set_xlabel("Held-out log score per site:\nwith − without river-network dependence"); grid(ax[0], "x")
    # panel b: by flow-connected training neighbour
    groups = [g for g in ["A: W1+W2"] + [g for g in St.group.unique() if g.startswith("B:")] if g in set(St.group)]
    k = 0; yt, yl = [], []
    for g in groups:
        col = BLUE if g.startswith("A") else ORANGE
        for s_, lab in [("with_nb", "with"), ("without_nb", "without")]:
            r = St[(St.group == g) & (St.stratum == s_)]
            if r.empty:
                continue
            r = r.iloc[0]
            ax[1].plot([r.per_site_lo95, r.per_site_hi95], [k, k], color=col, lw=1.6, solid_capstyle="round")
            ax[1].plot(r.per_site, k, "o" if s_ == "with_nb" else "s", color=col, mec="white", ms=4.5)
            ax[1].text(0.30, k, f"n = {int(r.n_sites)}", va="center", ha="right", fontsize=6.2, color=MUTED)
            yt.append(k); yl.append(f"{'Within networks' if g.startswith('A') else 'Held-out catchments'}, {lab}"); k += 1
        k += 0.5
    ax[1].axvline(0, color=INK2, lw=0.6); ax[1].set_yticks(yt, yl); ax[1].set_ylim(max(yt) + 0.5, -0.5); ax[1].set_xlim(-0.08, 0.31)
    ax[1].set_xlabel("Held-out log score per site\n(flow-connected training site within 25 km)"); grid(ax[1], "x")
    letter(ax[0], "a", x=-0.12); letter(ax[1], "b", x=-0.62)
    fig.tight_layout(w_pad=1.0); save(fig, "Fig6_predictive_validation")


# ------------------------------------------------------------------ Fig 7
def fig7():
    g = pd.read_csv(TB / "A2_selection_grid.csv"); pr = pd.read_csv(TB / "A2_PR_by_psi.csv"); bl = pd.read_csv(TB / "A3_river_length_psi.csv")
    fig, ax = plt.subplots(1, 2, figsize=(W2, W2 * 0.33))
    c = 0.1
    for k, col, mk in [(1.0, SEQ_STEPS[5], "o"), (0.5, SEQ_STEPS[3], "s"), (0.25, SEQ_STEPS[1], "^")]:
        d = g[np.isclose(g.threshold, c) & np.isclose(g.kappa, k) & (g.psi.abs() <= 0.2 + 1e-9)].sort_values("psi")
        ax[0].fill_between(d.psi, d.DS_lo95, d.DS_hi95, color=col, alpha=0.13, lw=0)
        ax[0].plot(d.psi, d.DS_median, color=col, lw=1.4, marker=mk, ms=3.5, label=f"$\\kappa$ = {k:g}")
    ax[0].axhline(0, color=INK2, lw=0.6); ax[0].axvline(0, color=GRID, lw=0.6)
    ax[0].set_xlabel("$\\psi$: residual selection of RSN sites (log scale)"); ax[0].set_ylabel("Siting contrast $\\Delta^{S}_{T}$ (0.10 mg P/L)")
    ax[0].legend(loc="upper left", handlelength=1.4); grid(ax[0]); ax[0].set_xticks([-0.2, -0.1, 0, 0.1, 0.2])
    for cc, col, mk in [(0.05, SEQ_STEPS[1], "s"), (0.1, SEQ_STEPS[3], "o"), (0.2, SEQ_STEPS[5], "^")]:
        d = pr[np.isclose(pr.threshold, cc) & (pr.psi.abs() <= 0.2 + 1e-9)].sort_values("psi")
        b = bl[np.isclose(bl.threshold, cc) & (bl.psi.abs() <= 0.2 + 1e-9)].sort_values("psi")
        ax[1].plot(d.psi, d.PR_median, color=col, lw=1.4, marker=mk, ms=3.2)
        ax[1].fill_between(b.psi, b.lo95, b.hi95, color=col, alpha=0.13, lw=0); ax[1].plot(b.psi, b["median"], color=col, lw=1.2, ls="--")
        ax[1].text(0.215, b["median"].iloc[-1], f"{cc:.2f} mg P/L", fontsize=6.3, color=INK2, va="center")
    ax[1].plot([], [], color=INK2, lw=1.4, label="Target site-years"); ax[1].plot([], [], color=INK2, lw=1.2, ls="--", label="River length")
    ax[1].legend(loc="upper right", handlelength=2.0, bbox_to_anchor=(0.86, 1.0))
    ax[1].set_xlabel("$\\psi$ (log scale)"); ax[1].set_ylabel("Reference-level exceedance probability"); grid(ax[1]); ax[1].set_xlim(-0.22, 0.3)
    ax[1].set_xticks([-0.2, -0.1, 0, 0.1, 0.2]); ax[1].set_ylim(0, 0.85)
    letter(ax[0], "a", x=-0.16); letter(ax[1], "b", x=-0.16)
    fig.tight_layout(w_pad=2.0); save(fig, "Fig7_selection_sensitivity")


# ------------------------------------------------------------------ Fig 8
def fig8(G, segs):
    LM = pd.read_parquet(Path("~/track_a_scratch/phase_04/gs_link_mean_pi.parquet").expanduser()).set_index("id").loc[G.id]
    st = pd.read_csv(TB / "A3_river_length_strata.csv")
    rsn = pd.read_csv(VT / "VF_rsn_only_Bbar.csv") if (VT / "VF_rsn_only_Bbar.csv").exists() else None
    fig = plt.figure(figsize=(W2, W2 * 0.62))
    a0 = fig.add_axes([0.0, 0.06, 0.50, 0.94]); a1 = fig.add_axes([0.70, 0.12, 0.29, 0.84])
    v = LM["pi_0.1"].to_numpy(); o = np.argsort(v)
    lc = LineCollection([segs[i] for i in o], array=v[o], cmap=SEQ, linewidths=0.28, norm=plt.Normalize(0, 1)); a0.add_collection(lc)
    a0.set_aspect("equal"); a0.autoscale(); a0.axis("off")
    xmin = min(s[:, 0].min() for s in segs); ymin = min(s[:, 1].min() for s in segs); scalebar(a0, xmin + 10, ymin + 5)
    cax = fig.add_axes([0.04, 0.05, 0.30, 0.016])
    cb = fig.colorbar(lc, cax=cax, orientation="horizontal"); cb.outline.set_visible(False)
    cb.set_label("Pr(annual mean > 0.10 mg P/L), RSN reference level", color=INK2, fontsize=6.8); cb.ax.tick_params(labelsize=6.5)
    keep = ["all", "flow_connected_site", "no_flow_connected_site", "observed_water_body", "new_water_body", "no_rsn_support"] + [f"region_{k}" for k in [7, 2, 1, 6, 5, 4, 3]]
    d = st[np.isclose(st.threshold, 0.1) & st.stratum.isin(keep)].set_index("stratum").loc[keep]
    names = {"all": "All mapped rivers in frame", "flow_connected_site": "Flow-connected to a site", "no_flow_connected_site": "Not flow-connected",
             "observed_water_body": "Water body with a site", "new_water_body": "Water body without a site", "no_rsn_support": "Cells without RSN support",
             **{f"region_{k}": REG[k] for k in range(1, 8)}}
    yy = []; y = 0
    for k, (s, r) in enumerate(d.iterrows()):
        if s in ("flow_connected_site", "region_7"):
            y += 0.6
        col = BLUE if s == "all" else INK2
        a1.plot([r.lo95, r.hi95], [y, y], color=col, lw=1.8, solid_capstyle="round"); a1.plot(r["median"], y, "o", color=col, mec="white", ms=4.5)
        a1.text(0.99, y, f"{r.km:,.0f} km", va="center", ha="right", fontsize=6.2, color=MUTED)
        yy.append(y); y += 1
    if rsn is not None:
        q = rsn[np.isclose(rsn.threshold, 0.1) & (rsn.quantity == "Bbar") & (rsn.variant == "primary")].iloc[0]
        a1.plot([q.rsn_only_lo95, q.rsn_only_hi95], [yy[0] + 0.38] * 2, color=AQUA, lw=1.4); a1.plot(q.rsn_only_median, yy[0] + 0.38, "s", color=AQUA, mec="white", ms=3.8)
        a1.text(q.rsn_only_hi95 + 0.02, yy[0] + 0.38, "RSN data only", fontsize=6.2, color=INK2, va="center")
    a1.set_yticks(yy, [names[s] for s in d.index]); a1.set_ylim(max(yy) + 0.6, -0.6); a1.set_xlim(0, 1.0)
    a1.set_xlabel("River-length-weighted exceedance,\nc = 0.10 mg P/L (95% CrI)"); grid(a1, "x")
    a0.text(0.0, 1.0, "a", transform=a0.transAxes, fontsize=9, fontweight="bold", va="top"); letter(a1, "b", x=-0.62)
    save(fig, "Fig8_river_length")


def main(which=None):
    OUT.mkdir(exist_ok=True)
    sites = sites_table()
    need_geo = which is None or any(w in which for w in ("1", "3", "8"))
    G, segs = load_geo() if need_geo else (None, None)
    todo = {"1": lambda: fig1(G, segs, sites), "2": fig2, "3": lambda: fig3(G, sites), "4": fig4, "5": fig5, "6": fig6, "7": fig7, "8": lambda: fig8(G, segs)}
    for k, f in todo.items():
        if which is None or k in which:
            f()
    print(sorted(p.name for p in OUT.glob("*.png")))


if __name__ == "__main__":
    main(sys.argv[1:] or None)
