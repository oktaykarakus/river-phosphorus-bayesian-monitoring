"""Phase 3C-F: prediction-setting comparison from EXISTING held-out scores (no new fit).

Setting A — new sites within monitored river networks: Phase 3C folds W1, W2 (water-body-blocked; phase_03c/tables).
Setting B — entirely unmonitored catchments: Phase 3C-R fold C1 (catchment-blocked; phase_03c_r/tables).
Never pooled across settings. For each fold: site-level SE (as reported) and management-catchment-clustered SE (sites in a
catchment are not independent: they share catchment / water-body / river effects), per-site mean difference, catchment-level sign
counts, leave-one-catchment-out range, and the flow-neighbour stratification (training site within 25 km river distance).
Outputs: tables/prediction_settings_by_fold.csv, tables/prediction_settings_by_catchment.csv, diagnostics/prediction_settings.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import p3f_lib as L  # noqa: E402

TA = L.TA


def flow_nb_flags(sites, held):
    Cp = pd.read_csv(L.C.DP / "rn_pairs_connected.csv"); Cp = Cp[Cp.h_km <= 25]
    nb = {}
    for a, b in zip(Cp.up_site, Cp.down_site):
        nb.setdefault(a, set()).add(b); nb.setdefault(b, set()).add(a)
    hs = set(held)
    return {s: bool(nb.get(s, set()) - hs) for s in held}


def fold_stats(d, label):
    """d: one row per held-out site with columns diff, mancat, has_nb."""
    n = len(d); tot = d["diff"].sum()
    se_site = np.sqrt(n) * d["diff"].std(ddof=1)
    g = d.groupby("mancat")["diff"].sum(); G = len(g)
    se_clu = np.sqrt(G / (G - 1) * ((g - g.mean()) ** 2).sum())        # SE of the total, catchment-clustered (CR1-type)
    loo = [tot - g[c] for c in g.index]
    rng = np.random.default_rng(20261115)
    boot = np.array([g.sample(G, replace=True, random_state=rng.integers(1 << 31)).sum() for _ in range(4000)])
    out = dict(setting=label["setting"], fold=label["fold"], n_sites=n, n_catchments=G, diff=tot, se_site=se_site, z_site=tot / se_site,
               se_catchment_clustered=se_clu, z_clustered=tot / se_clu, boot_catchment_95=np.quantile(boot, [.025, .975]).round(2).tolist(),
               per_site_mean=tot / n, catchments_m2rn_better=int((g > 0).sum()), loo_catchment_range=[float(min(loo)), float(max(loo))],
               largest_catchment_contribution=float(g.abs().max()), share_sites_with_flow_nb=float(d.has_nb.mean()))
    for flag, name in [(True, "with_nb"), (False, "without_nb")]:
        s = d[d.has_nb == flag]
        if len(s) > 1:
            gs = s.groupby("mancat")["diff"].sum(); Gs = len(gs)
            out[f"{name}_n"] = len(s); out[f"{name}_diff"] = float(s["diff"].sum())
            out[f"{name}_se_site"] = float(np.sqrt(len(s)) * s["diff"].std(ddof=1))
            out[f"{name}_se_clustered"] = float(np.sqrt(Gs / (Gs - 1) * ((gs - gs.mean()) ** 2).sum())) if Gs > 1 else None
            out[f"{name}_per_site"] = float(s["diff"].mean())
    return out, g


def main():
    meas, sites, sy = L.load_tables()
    cat = sites.set_index("site").mancat
    rows, cats = [], []
    W = pd.read_csv(TA / "phase_03c" / "tables" / "validation_site_scores.csv")
    for fold in ["W1", "W2"]:
        w = W[W.fold == fold].pivot_table(index="site", columns="model", values="elpd_site")
        held = list(w.index); nb = flow_nb_flags(sites, held)
        d = pd.DataFrame(dict(site=held, diff=(w["m2rn"] - w["m1s2"]).values, mancat=cat.reindex(held).values, has_nb=[nb[s] for s in held]))
        o, g = fold_stats(d, dict(setting="A: new sites within monitored networks", fold=fold)); rows.append(o)
        cats.append(pd.DataFrame(dict(fold=fold, mancat=g.index, diff=g.values)))
    Cs = pd.read_csv(TA / "phase_03c_r" / "tables" / "validation_C1_site_scores.csv")
    c = Cs.pivot_table(index="site", columns="model", values="elpd_site")
    nbc = Cs[Cs.model == "m1s2"].set_index("site").has_flow_training_nb_25km
    d = pd.DataFrame(dict(site=c.index, diff=(c["m2rn"] - c["m1s2"]).values, mancat=cat.reindex(c.index).values, has_nb=nbc.reindex(c.index).values))
    o, g = fold_stats(d, dict(setting="B: unmonitored catchments", fold="C1")); rows.append(o)
    cats.append(pd.DataFrame(dict(fold="C1", mancat=g.index, diff=g.values)))
    # setting-A combined (two folds of the SAME setting; folds are disjoint site sets)
    a = [r for r in rows if r["fold"] in ("W1", "W2")]
    A_tot = sum(r["diff"] for r in a); A_se = np.sqrt(sum(r["se_catchment_clustered"] ** 2 for r in a))
    A_se_site = np.sqrt(sum(r["se_site"] ** 2 for r in a))
    T = pd.DataFrame(rows); T.to_csv(L.ROOT / "tables" / "prediction_settings_by_fold.csv", index=False)
    pd.concat(cats).to_csv(L.ROOT / "tables" / "prediction_settings_by_catchment.csv", index=False)
    S = pd.read_csv(TA / "phase_03c" / "tables" / "validation_summary.csv").set_index("fold")
    C1 = json.load(open(TA / "phase_03c_r" / "diagnostics" / "validation_C1_summary.json"))
    calib = {"A (W1+W2)": {m: dict(pit_max_dev=S.loc["ALL", f"pit_max_dev_decile_{m}"], pit_cov90=S.loc["ALL", f"pit_cov90_{m}"],
                                   siteyear_cov90=S.loc["ALL", f"siteyear_cov90_{m}"],
                                   pred_minus_obs_rate={c: float(S.loc["ALL", f"pred_rate_{c}_{m}"] - S.loc["ALL", f"obs_rate_{c}"]) for c in ["0.05", "0.1", "0.2"]})
                           for m in ["m1s2", "m2rn"]},
             "B (C1)": {m: dict(pit_max_dev=C1[f"pit_max_dev_decile_{m}"], pit_cov90=C1[f"pit_cov90_{m}"], siteyear_cov90=C1[f"siteyear_cov90_{m}"],
                                pred_minus_obs_rate={c: C1[f"pred_rate_{c}_{m}"] - C1[f"obs_rate_{c}"] for c in ["0.05", "0.1", "0.2"]})
                        for m in ["m1s2", "m2rn"]}}
    out = dict(by_fold=T.round(4).to_dict("records"),
               setting_A_combined=dict(diff=A_tot, se_site=A_se_site, se_clustered=A_se, z_clustered=A_tot / A_se, n_sites=int(sum(r["n_sites"] for r in a))),
               setting_B=dict(diff=rows[-1]["diff"], se_site=rows[-1]["se_site"], se_clustered=rows[-1]["se_catchment_clustered"],
                              z_clustered=rows[-1]["z_clustered"], n_catchments=rows[-1]["n_catchments"]),
               calibration=calib,
               note="SEs are for the total ELPD difference. Clustered SE treats management catchments as the independent units.")
    L.save_json(out, L.ROOT / "diagnostics" / "prediction_settings.json")
    pd.set_option("display.width", 250)
    print(T.round(3).T.to_string()); print(json.dumps({k: out[k] for k in ["setting_A_combined", "setting_B"]}, indent=1)); print(json.dumps(calib, indent=1))


if __name__ == "__main__":
    main()
