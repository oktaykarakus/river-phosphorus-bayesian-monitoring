"""Consolidated table of all validation fits: predictive differences, every pre-declared diagnostic, verdicts.

Reads only existing outputs (no fitting, no scoring):
  diagnostics/vf_{fold}_R3_sampling_diagnostics.json, diagnostics/VF_validation.json,
  diagnostics/VF_rb_check_{fold}[_4000draws_EXPLORATORY].json
Writes tables/VF_consolidated_validation.csv. Exploratory columns are prefixed "expl_".
Pre-declared rules (P4-V0 / VF_00 §5, unchanged): divergences <= 0.1%, E-BFMI >= 0.3, hyperparameter R-hat <= 1.05,
MCSE of the fold ELPD difference < 0.10 x clustered SE, Rao-Blackwell max site disagreement <= 0.05 (400 draws).
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.dont_write_bytecode = True
VF = Path(__file__).resolve().parents[1]


def main():
    V = json.load(open(VF / "diagnostics/VF_validation.json"))
    G = {g["group"]: g for g in V["groups"]}
    rows = []
    for f in ["W1", "W2", "C1", "C2"]:
        d = json.load(open(VF / f"diagnostics/vf_{f}_R3_sampling_diagnostics.json"))
        ls = d.get("log_scale_diagnostics", {})
        rhat = max([d["hyper_max_rhat"]] + [v["r_hat"] for v in ls.values()])
        g = G[f]; rb = V["rao_blackwell_check"][f]
        rb4 = json.load(open(VF / f"diagnostics/VF_rb_check_{f}_4000draws_EXPLORATORY.json"))
        div_ok = d["divergence_share"] <= 0.001; eb_ok = min(d["ebfmi"]) >= 0.3; rh_ok = rhat <= 1.05
        mc_ok = g["mcse_over_se"] < 0.10; rb_ok = rb["max_abs_mean_diff"] <= 0.05
        cmp_ok = V["adequacy"][f"{f}_m1s2"]["pass_all"]
        fails = [n for n, ok in [("divergences", div_ok), ("E-BFMI", eb_ok), ("R-hat", rh_ok), ("MCSE", mc_ok),
                                 ("Rao-Blackwell", rb_ok), ("comparator fit", cmp_ok)] if not ok]
        rows.append(dict(
            fold=f, setting=g["setting"], n_sites=g["n_sites"], n_catchments=g["n_catchments"],
            elpd_diff=g["diff"], se_clustered=g["se_clustered"], z=g["z"], per_site=g["per_site"],
            per_site_lo95=g["per_site_lo95"], per_site_hi95=g["per_site_hi95"],
            cpu_h=d["cpu_h"], divergences=d["divergences"], divergence_pct=100 * d["divergence_share"],
            min_ebfmi=min(d["ebfmi"]), max_rhat_hyper=rhat, min_ess_bulk_hyper=d["hyper_min_ess_bulk"],
            argmin_ess_bulk=d["hyper_argmin_ess_bulk"], min_ess_tail_hyper=d["hyper_min_ess_tail"],
            argmin_ess_tail=d["hyper_argmin_ess_tail"], min_ess_bulk_latent=d["latent_min_ess_bulk"],
            min_ess_tail_latent=d["latent_min_ess_tail"], treedepth_hits=d["treedepth_hits"],
            mcse_diff=g["mcse_diff"], mcse_over_se=g["mcse_over_se"], rb_max_400=rb["max_abs_mean_diff"],
            comparator_pass=cmp_ok, sampler_pass=div_ok and eb_ok and rh_ok, scoring_precision_pass=mc_ok and rb_ok,
            failed_criteria="; ".join(fails) or "none",
            status="VALIDATED" if not fails else "NOT VALIDATED",
            expl_rb_max_4000=rb4["max_abs_mean_diff"], expl_rb_sd_z_4000=rb4["sd_z"]))
    for grp in ["A: W1+W2", "B: C1+C2"]:
        g = G[grp]
        rows.append(dict(fold=grp, setting=g["setting"], n_sites=g["n_sites"], n_catchments=g["n_catchments"],
                         elpd_diff=g["diff"], se_clustered=g["se_clustered"], z=g["z"], per_site=g["per_site"],
                         per_site_lo95=g["per_site_lo95"], per_site_hi95=g["per_site_hi95"], mcse_diff=g["mcse_diff"],
                         mcse_over_se=g["mcse_over_se"], status="PROVISIONAL (component folds not validated)"))
    T = pd.DataFrame(rows)
    T.to_csv(VF / "tables/VF_consolidated_validation.csv", index=False)
    pd.set_option("display.width", 250)
    print(T.round(3).to_string())


if __name__ == "__main__":
    main()
