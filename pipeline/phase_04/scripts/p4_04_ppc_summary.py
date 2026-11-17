"""Phase 4 A4: posterior predictive adequacy of the authoritative posterior.

(1) Tabulates the replicate checks already computed on the combined 4 x 5,000 posterior by the unchanged Phase 3B/3C-F PPC code
    (phase_03c_f/addendum_T1/tables/T1_combined_ppc_*): site-year sample-mean exceedance by network, region and size class;
    monthly log-mean profiles; and applies the pre-declared material-miss rule (P4-V0 §3 A4).
(2) Adds a LIMIT-FREE low-tail check (the censored-share check needs the unobserved limits of uncensored values): share of
    measurements below 0.005 and 0.01 mg/L by network, observed (censored values counted when their limit <= the cut) vs
    posterior predictive replicates at the observed dates (200 draws, seed 20261122).
Outputs: tables/A4_ppc_cells.csv, A4_lowtail.csv; diagnostics/A4_ppc.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import p4_lib as P  # noqa: E402

L = P.redirect()
B = L.B
T1 = P.P3F / "addendum_T1"


def main():
    out = {}
    cells = []
    for nm in ["samplemean_exceed_network", "samplemean_exceed_region_network", "samplemean_exceed_size_network"]:
        t = pd.read_csv(T1 / "tables" / f"T1_combined_ppc_{nm}.csv"); t["check"] = nm
        t["direction"] = np.where(t.observed > t.rep95, "obs above", np.where(t.observed < t.rep05, "obs below", "inside"))
        cells.append(t)
        out[nm] = dict(inside=int(t.inside.sum()), cells=len(t))
    Cc = pd.concat(cells, ignore_index=True); Cc.to_csv(P.P4 / "tables/A4_ppc_cells.csv", index=False)
    reg = Cc[Cc.check == "samplemean_exceed_region_network"].copy(); reg["network"] = reg.group.str.split("|").str[1]
    rule = {}
    for nw, g in reg.groupby("network"):
        above = int((g.direction == "obs above").sum()); below = int((g.direction == "obs below").sum())
        rule[nw] = dict(cells=len(g), outside=above + below, above=above, below=below, material_by_rule=bool(max(above, below) >= 3))
    out["material_miss_rule_region_x_threshold"] = rule
    for nm in ["monthly_network", "monthly_region_network"]:
        t = pd.read_csv(T1 / "tables" / f"T1_combined_ppc_{nm}.csv")
        out[nm] = {g: dict(inside=int(d.inside.sum()), months=len(d), mean_abs_z=float(d.z.abs().mean())) for g, d in t.groupby("g")} if nm == "monthly_network" else \
            dict(inside=int(t.inside.sum()), cells=len(t), share_inside=float(t.inside.mean()), mean_abs_z=float(t.z.abs().mean()))
    comb = json.load(open(T1 / "diagnostics/T1_combined_diagnostics.json"))
    out["quantiles"] = {k: comb["ppc"][k] for k in ["quantiles_RSN", "quantiles_MS"]}
    out["censored_share_one_limit_rule"] = {k: comb["ppc"][k] for k in ["censored_RSN", "censored_MS"]}
    out["flow_dependence"] = comb["flow_decomposition"]
    # (2) limit-free low tail
    meas, sites, sy = L.load_tables(); mc = L.R.model_config("R3")
    D = P.load_post(200)
    eta = B.eta_obs(D, meas, sites, sy, mc)                                  # (200, N)
    sdo = B.sig_site(D, sites, mc)[:, meas.site_idx.to_numpy() - 1]
    rng = np.random.default_rng(P.CFG["seeds"]["ppc"])
    yrep = eta + sdo * rng.standard_normal(eta.shape)
    y = meas.value.to_numpy() if "value" in meas else meas.y.to_numpy()
    cen = (meas.censored.to_numpy() if "censored" in meas else meas.cens.to_numpy()).astype(bool)
    lim = meas.limit.to_numpy() if "limit" in meas else meas.L.to_numpy()
    net = sites.network.to_numpy()[meas.site_idx.to_numpy() - 1]
    rows = []
    for cut in [0.005, 0.01, 0.02]:
        obs_below = np.where(cen, lim <= cut + 1e-12, y < cut)
        amb = cen & (lim > cut + 1e-12)
        for nw in ["RSN", "MS"]:
            m = net == nw
            rep = (yrep[:, m] < np.log(cut)).mean(1)
            rows.append(dict(cut=cut, network=nw, n=int(m.sum()), observed=float(obs_below[m].mean()), observed_upper_if_ambiguous_below=float((obs_below | amb)[m].mean()),
                             rep05=float(np.quantile(rep, .05)), rep_median=float(np.median(rep)), rep95=float(np.quantile(rep, .95)),
                             inside=bool(np.quantile(rep, .05) <= obs_below[m].mean() <= np.quantile(rep, .95))))
    LT = pd.DataFrame(rows); LT.to_csv(P.P4 / "tables/A4_lowtail.csv", index=False)
    out["lowtail_limit_free"] = LT.round(4).to_dict("records")
    P.save_json(out, P.P4 / "diagnostics/A4_ppc.json")
    print(json.dumps({k: out[k] for k in out if k not in ("quantiles", "flow_dependence")}, indent=1)[:4000])


if __name__ == "__main__":
    main()
