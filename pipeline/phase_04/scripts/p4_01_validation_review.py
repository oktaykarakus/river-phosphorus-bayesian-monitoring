"""Phase 4 A1: review of the completed held-out experiments (no refit), precision analysis for setting B, and the frozen
assignment of the proposed additional catchment folds C2, C3 (P4-V0 §3 A1; drawn now, before any fit, seed 20261113).
Inputs (read-only): phase_03c/tables/validation_site_scores.csv (W1, W2), phase_03c_r/tables/validation_C1_site_scores.csv (C1),
phase_03c_f/tables/prediction_settings_by_fold.csv / _by_catchment.csv.
Outputs: tables/A1_fold_review.csv, A1_region_coverage.csv, A1_precision_by_K.csv; validation/fold_assignments_C2_C3.csv;
diagnostics/A1_validation_review.json
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import p4_lib as P  # noqa: E402

TA = P.TA


def main():
    sites = pd.read_parquet(TA / "phase_03a/data_processed/sites.parquet")
    W = pd.read_csv(TA / "phase_03c/tables/validation_site_scores.csv")
    C1 = pd.read_csv(TA / "phase_03c_r/tables/validation_C1_site_scores.csv"); C1["fold"] = "C1"
    S = pd.concat([W, C1], ignore_index=True)
    S["model"] = S.model.str.lower().replace({"m2rn": "m2rn", "r0": "m2rn", "m2-rn": "m2rn", "m1-s2": "m1s2"})
    piv = S.pivot_table(index=["fold", "site"], columns="model", values="elpd_site").reset_index()
    mods = [c for c in piv.columns if c not in ("fold", "site")]
    m2 = [c for c in mods if c != "m1s2"][0]
    piv["diff"] = piv[m2] - piv["m1s2"]
    piv = piv.merge(sites[["site", "mancat", "region", "network"]], on="site", how="left")
    piv["mancat"] = piv.mancat.fillna("__none_" + piv.site)
    rows = []
    for f, d in piv.groupby("fold"):
        cat = d.groupby("mancat").agg(diff=("diff", "sum"), n=("diff", "size"))
        K = len(cat); tot = d["diff"].sum(); n = len(d)
        se_cl = np.sqrt(K / (K - 1) * ((cat["diff"] - cat.n * tot / n) ** 2).sum())
        rows.append(dict(fold=f, setting="A" if f.startswith("W") else "B", sites=n, catchments=K, regions=d.region.nunique(), diff_total=tot,
                         se_clustered=se_cl, z_clustered=tot / se_cl, per_site=tot / n, per_site_se=se_cl / n))
    FR = pd.DataFrame(rows); FR.to_csv(P.P4 / "tables/A1_fold_review.csv", index=False)
    RC = piv.groupby(["fold", "region"]).agg(sites=("site", "size"), catchments=("mancat", "nunique"), diff=("diff", "sum")).reset_index()
    RC.to_csv(P.P4 / "tables/A1_region_coverage.csv", index=False)
    # precision for setting B as a function of the number of held-out catchments K (ratio estimator, C1 between-catchment spread)
    c1 = piv[piv.fold == "C1"].groupby("mancat").agg(d=("diff", "sum"), n=("diff", "size"))
    r = c1.d.sum() / c1.n.sum(); resid = c1.d - r * c1.n; nbar = c1.n.mean()
    sd_res = resid.std(ddof=1)
    gainA = FR[FR.setting == "A"].diff_total.sum() / FR[FR.setting == "A"].sites.sum()
    prec = []
    for K in [16, 32, 48, 64, 86]:
        se = sd_res / (np.sqrt(K) * nbar)
        prec.append(dict(K_catchments=K, per_site_se=se, upper95_if_true_equals_C1=r + 1.96 * se, lower95_if_true_equals_C1=r - 1.96 * se,
                         margin_half_setting_A_gain=gainA / 2, half_width_95=1.96 * se))
    PR = pd.DataFrame(prec); PR.to_csv(P.P4 / "tables/A1_precision_by_K.csv", index=False)
    # frozen C2 / C3: region-stratified draw of 32 catchments from those not held out in C1 (seed 20261113)
    held_c1 = set(piv[piv.fold == "C1"].mancat)
    cats = sites.dropna(subset=["mancat"]).groupby("mancat").agg(region=("region", lambda x: x.value_counts().index[0]), n_sites=("site", "size")).reset_index()
    pool = cats[~cats.mancat.isin(held_c1)].copy()
    rng = np.random.default_rng(P.CFG["seeds"]["folds"])
    pool["u"] = rng.random(len(pool))
    target = 32; share = pool.region.value_counts() / len(pool)
    k_reg = np.maximum(2, np.floor(share * target).astype(int))
    while k_reg.sum() < target:
        k_reg[(share * target - k_reg).idxmax()] += 1
    while k_reg.sum() > target:
        k_reg[k_reg.idxmax()] -= 1
    pick = []
    for reg, k in k_reg.items():
        g = pool[pool.region == reg].sort_values("u").head(int(k)).reset_index(drop=True)
        g["fold"] = np.where(np.arange(len(g)) % 2 == 0, "C2", "C3")
        pick.append(g)
    FA = pd.concat(pick, ignore_index=True)[["fold", "mancat", "region", "n_sites"]].sort_values(["fold", "region", "mancat"])
    FA.to_csv(P.P4 / "validation/fold_assignments_C2_C3.csv", index=False)
    cov_all = pd.concat([piv[piv.fold == "C1"].groupby("region").mancat.nunique().rename("C1"),
                         FA[FA.fold == "C2"].groupby("region").size().rename("C2"), FA[FA.fold == "C3"].groupby("region").size().rename("C3")], axis=1).fillna(0).astype(int)
    out = dict(fold_review=FR.round(4).to_dict("records"), setting_A_per_site_gain=gainA, C1_per_site=r, C1_between_catchment_resid_sd=sd_res,
               C1_mean_sites_per_catchment=nbar, precision_by_K=PR.round(4).to_dict("records"),
               C1_regions=sorted(piv[piv.fold == "C1"].region.unique()), C1_catchments_by_region=piv[piv.fold == "C1"].groupby("region").mancat.nunique().to_dict(),
               proposed_C2_C3=dict(n_catchments=int(len(FA)), sites=int(FA.n_sites.sum()), coverage_by_region=cov_all.to_dict("index")))
    P.save_json(out, P.P4 / "diagnostics/A1_validation_review.json")
    pd.set_option("display.width", 220)
    print(FR.round(3).to_string()); print(PR.round(4).to_string()); print(cov_all); print(RC[RC.fold == "C1"])


if __name__ == "__main__":
    main()
