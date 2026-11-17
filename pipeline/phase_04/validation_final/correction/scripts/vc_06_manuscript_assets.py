"""Correction stage: manuscript assets from verified outputs only (no fitting, no scoring).
  - Table S6 (scoring precision and consistency checks) -> appended to manuscript/supplementary_tables_generated.md (idempotent)
  - Fig. 6 re-rendered with the Phase 4B scores (unchanged) plus thin intervals including correlated Monte Carlo error, and W1 marked †
Sources: Phase 4B diagnostics/VF_validation.json, tables/VF_validation_*.csv; correction diagnostics VC_score_checks.json, VC_rb_*.json."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
VC = Path(__file__).resolve().parents[1]; VFD = VC.parent
sys.path.insert(0, str(VFD / "scripts"))
import vf_05_figures as FG  # noqa: E402

plt = FG.plt
V = json.load(open(VFD / "diagnostics/VF_validation.json")); G = {g["group"]: g for g in V["groups"]}
CK = json.load(open(VC / "diagnostics/VC_score_checks.json")); EX = CK["exploratory_correlated_mcse"]
RB = {f: json.load(open(VC / f"diagnostics/VC_rb_{f}_alldraws_AMENDED.json")) for f in ["W1", "W2", "C1", "C2"]}
SD = {f: json.load(open(VFD / f"diagnostics/vf_{f}_R3_sampling_diagnostics.json")) for f in ["W1", "W2", "C1", "C2"]}


def inflated(g):
    if g in EX:
        e = EX[g]; n = G[g]["n_sites"]; se = float(np.hypot(e["se_clustered"], e["mcse_correlated"]))
        return (e["diff"] - 1.96 * se) / n, (e["diff"] + 1.96 * se) / n
    return None


def table_s6():
    rows = ["**Table S6. Scoring precision and consistency checks of the held-out comparisons.** Monte Carlo (MC) error of the fold "
            "difference as a share of its clustered standard error: as specified (sum of site-level errors) and allowing for correlation "
            "across sites through shared posterior draws (the specified tolerance is 10%). Pareto k̂: sites whose predictive density over "
            "posterior draws has a heavy upper tail (k̂ ≥ 0.7), at which MC error is estimated unreliably. Consistency check: maximum "
            "discrepancy between the conditional predictive mean and the direct posterior draws of held-out site levels (tolerance 0.05), "
            "on the 400 scoring draws as specified, and on all 4,000 draws. Pooled rows: per-site 95% interval with the correlated MC "
            "error added to the clustered standard error.", "",
            "| Fold | Sampling criteria met | MC error / SE, specified | MC error / SE, correlated | Sites with k̂ ≥ 0.7 | Consistency, 400 draws | Consistency, all draws |",
            "|---|---|---|---|---|---|---|"]
    for f in ["W1", "W2", "C1", "C2"]:
        d = SD[f]; ok = d["divergence_share"] <= 0.001
        rows.append(f"| {f} | {'yes' if ok else 'no (1.75% divergent)'} | {EX[f]['ratio_frozen']:.3f} | {EX[f]['ratio_correlated']:.3f} | "
                    f"{CK[f]['S2']['n_sites_k_ge_07']} | {V['rao_blackwell_check'][f]['max_abs_mean_diff']:.3f} | {RB[f]['max_abs_mean_diff']:.3f} |")
    rows += ["", "| Setting | Δ log score (clustered SE) | Correlated MC error | Per site, clustered 95% CI | Per site, 95% CI with MC error | z with MC error |",
             "|---|---|---|---|---|---|"]
    for g, lab in [("A: W1+W2", "A: within monitored networks†"), ("B: C1+C2", "B: held-out catchments")]:
        e = EX[g]; gg = G[g]
        rows.append(f"| {lab} | {gg['diff']:+.1f} ({gg['se_clustered']:.1f}) | {e['mcse_correlated']:.1f} | {gg['per_site_lo95']:+.3f}, {gg['per_site_hi95']:+.3f} | "
                    f"{e['lo95']:+.3f}, {e['hi95']:+.3f} | {e['z_mc_inflated']:.2f} |")
    rows += ["", "† Includes fold W1, whose fit did not meet the sampling criteria."]
    p = VFD / "manuscript/supplementary_tables_generated.md"; t = p.read_text()
    if "**Table S6." in t:
        t = t[:t.index("**Table S6.")].rstrip() + "\n"
    p.write_text(t.rstrip() + "\n\n" + "\n".join(rows) + "\n")
    print("\n".join(rows))


def fig6():
    T = pd.read_csv(VFD / "tables/VF_validation_folds.csv"); St = pd.read_csv(VFD / "tables/VF_validation_strata.csv")
    order = ["W1", "W2", "A: W1+W2", "C1", "C2", "B: C1+C2"]
    T = T.set_index("group").loc[order]
    fig, ax = plt.subplots(1, 2, figsize=(FG.W2, FG.W2 * 0.40), gridspec_kw=dict(width_ratios=[1.25, 1]))
    margin = V["C5"]["margin"]
    names = {"W1": "W1†", "W2": "W2", "A: W1+W2": "Pooled†", "C1": "C1", "C2": "C2", "B: C1+C2": "Pooled"}
    ys, ylab, y, prev = [], [], 0.0, None
    heads = {"A": ("New sites within monitored river networks", FG.BLUE), "B": ("Sites in held-out catchments", FG.ORANGE)}
    for g, r in T.iterrows():
        if r.setting != prev:
            y += 0.3 if prev is None else 0.8
            ax[0].text(-0.058, y - 0.75, heads[r.setting][0], fontsize=6.4, color=heads[r.setting][1], va="center", ha="left", zorder=5,
                       bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none"))
            prev = r.setting
        ys.append(y); ylab.append(names[g]); y += 1
    for (g, r), k in zip(T.iterrows(), ys):
        col = FG.BLUE if r.setting == "A" else FG.ORANGE; pooled = g.startswith(("A:", "B:"))
        inf = inflated(g)
        if inf:
            ax[0].plot(list(inf), [k, k], color=col, lw=0.8, solid_capstyle="butt", zorder=2)
        ax[0].plot([r.per_site_lo95, r.per_site_hi95], [k, k], color=col, lw=2.2 if pooled else 1.5, solid_capstyle="round", zorder=3)
        ax[0].plot(r.per_site, k, "D" if pooled else "o", color=col if pooled else "white", mec=col, mew=1.1, ms=4.5, zorder=4)
        ax[0].text(0.268, k, f"{int(r.n_sites)} sites, {int(r.n_catchments)} catchments", va="center", ha="right", fontsize=6.2, color=FG.MUTED)
    ax[0].axvspan(-margin, margin, ymin=0, ymax=1, color=FG.PALE, zorder=0)
    ax[0].axvline(0, color=FG.INK2, lw=0.6)
    ax[0].set_yticks(ys, ylab); ax[0].set_ylim(max(ys) + 0.6, -0.75); ax[0].set_xlim(-0.07, 0.27)
    ax[0].set_xticks([-0.05, 0, 0.05, 0.10, 0.15])
    ax[0].set_xlabel("Held-out log score per site:\nwith − without river-network dependence"); FG.grid(ax[0], "x")
    k = 0; yt, yl = [], []
    for g in ["A: W1+W2", "B: C1+C2"]:
        col = FG.BLUE if g.startswith("A") else FG.ORANGE
        for s_, lab in [("with_nb", "with"), ("without_nb", "without")]:
            r = St[(St.group == g) & (St.stratum == s_)]
            if r.empty:
                continue
            r = r.iloc[0]
            ax[1].plot([r.per_site_lo95, r.per_site_hi95], [k, k], color=col, lw=1.6, solid_capstyle="round")
            ax[1].plot(r.per_site, k, "o" if s_ == "with_nb" else "s", color=col, mec="white", ms=4.5)
            ax[1].text(0.30, k, f"n = {int(r.n_sites)}", va="center", ha="right", fontsize=6.2, color=FG.MUTED)
            yt.append(k); yl.append(f"{'Within networks†' if g.startswith('A') else 'Held-out catchments'}, {lab}"); k += 1
        k += 0.5
    ax[1].axvline(0, color=FG.INK2, lw=0.6); ax[1].set_yticks(yt, yl); ax[1].set_ylim(max(yt) + 0.5, -0.5); ax[1].set_xlim(-0.08, 0.31)
    ax[1].set_xlabel("Held-out log score per site\n(flow-connected training site within 25 km)"); FG.grid(ax[1], "x")
    FG.letter(ax[0], "a", x=-0.12); FG.letter(ax[1], "b", x=-0.62)
    fig.tight_layout(w_pad=1.0); FG.save(fig, "Fig6_predictive_validation")


if __name__ == "__main__":
    table_s6(); fig6()
