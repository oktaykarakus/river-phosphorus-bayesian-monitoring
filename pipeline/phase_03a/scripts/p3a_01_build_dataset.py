"""Phase 3A — build the frozen Stan-ready modelling dataset from retained Pilot 01 / Phase 2 / Phase 2B inputs.

Training population = all eligible (primary-rule) site-years at network-matched sites with defined size and
storm-overflow covariates (Phase 2 analysis sample, 2,344 site-years). The frozen primary target T (2,256
site-years, phase_02b/specifications/target_T_siteyears.csv) is carried as a FLAG, not used to filter training data.

Outputs (phase_03a/data_processed/):
  measurements.parquet, sites.parquet, siteyears.parquet, data_dictionary.csv, dataset_validation.json
Run from phase_03a/:  environment/venv/bin/python -I scripts/p3a_01_build_dataset.py
"""
import glob, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TA = ROOT.parent
PILOT, P2, P2B = TA / "pilot_01", TA / "phase_02", TA / "phase_02b"
OUT = ROOT / "data_processed"; OUT.mkdir(exist_ok=True)

REGIONS = ["Anglian", "Midlands", "NorthEast", "NorthWest", "SouthWest", "Southern", "Thames"]
GEOLOGY = ["Calcareous", "Siliceous", "Organic", "Unknown"]   # Calcareous = reference


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main():
    checks = {}
    const = json.load(open(P2B / "specifications/standardisation_constants.json"))
    sy = pd.read_csv(P2 / "data_processed/site_years_p2.csv")
    S = pd.read_csv(P2 / "data_processed/sites_matched.csv")
    obs = pd.read_csv(PILOT / "data/observations_clean.csv")
    T = pd.read_csv(P2B / "specifications/target_T_siteyears.csv")
    inputs = {str(p.relative_to(TA)): sha(p) for p in [P2 / "data_processed/site_years_p2.csv", P2 / "data_processed/sites_matched.csv",
                                                       PILOT / "data/observations_clean.csv", P2B / "specifications/target_T_siteyears.csv",
                                                       P2B / "specifications/standardisation_constants.json"]}

    # ---------------- site-years (training population)
    sy["geology"] = sy.geology.fillna("Unknown").replace({"Salt": "Unknown"})
    D = sy[(sy.cls != "U") & sy.log_area.notna() & sy.log_edm5.notna()].copy()
    checks["siteyears_by_network"] = D.network.value_counts().to_dict()
    assert len(D) == 2344 and (D.network == "RSN").sum() == 839 and (D.network == "MS").sum() == 1505
    assert not D.duplicated(["site", "year", "network"]).any()
    # a site belongs to exactly one network (disjoint sets)
    nets = D.groupby("site").network.nunique()
    assert (nets == 1).all(); checks["sites_in_both_networks"] = int((nets > 1).sum())

    # ---------------- sites
    site_cols = ["site", "network", "region", "wma", "mancat", "easting", "northing", "log_area", "log_edm5", "edm_n_5km", "geology", "cls", "dist",
                 "link_water_body_id", "pt_water_body_id", "size_cls", "size_typology", "link_strahler"]
    site = D.sort_values(["site", "year"]).groupby("site", as_index=False).first()[site_cols]
    # site attributes must be constant across a site's years
    for c in ["network", "region", "wma", "log_area", "log_edm5", "geology", "easting", "northing"]:
        assert (D.groupby("site")[c].nunique(dropna=False) == 1).all(), c
    site = site.sort_values(["network", "site"], ascending=[False, True]).reset_index(drop=True)   # RSN first
    site["site_idx"] = np.arange(1, len(site) + 1)
    site["ms"] = (site.network == "MS").astype(int)
    site["region_idx"] = site.region.map({r: i + 1 for i, r in enumerate(REGIONS)})
    assert site.region_idx.notna().all()
    site["geology_idx"] = site.geology.map({g: i + 1 for i, g in enumerate(GEOLOGY)})
    assert site.geology_idx.notna().all()
    site["wb"] = site.link_water_body_id.fillna(site.pt_water_body_id)
    for col, name in [("wma", "wma_idx"), ("mancat", "mancat_idx"), ("wb", "wb_idx")]:
        lev = sorted(site[col].dropna().unique()); site[name] = site[col].map({v: i + 1 for i, v in enumerate(lev)})
        checks[f"levels_{col}"] = len(lev); checks[f"missing_{col}"] = int(site[col].isna().sum())
    site["x_km"] = site.easting / 1000; site["y_km"] = site.northing / 1000
    site["z_logarea"] = (site.log_area - const["log_area_mean"]) / const["log_area_sd"]
    site["z_edm"] = (site.log_edm5 - const["log1p_edm5_mean"]) / const["log1p_edm5_sd"]
    checks["z_logarea_mean_sd"] = [float(site.z_logarea.mean()), float(site.z_logarea.std())]
    checks["z_edm_mean_sd"] = [float(site.z_edm.mean()), float(site.z_edm.std())]
    assert abs(site.z_logarea.mean()) < 1e-6 and abs(site.z_logarea.std() - 1) < 1e-6
    assert abs(site.z_edm.mean()) < 1e-6 and abs(site.z_edm.std() - 1) < 1e-6
    checks["sites_by_network"] = site.network.value_counts().to_dict()
    assert len(site) == 1832

    # ---------------- site-years
    D = D.merge(site[["site", "site_idx"]], on="site", how="left", validate="many_to_one")
    D = D.sort_values(["site_idx", "year"]).reset_index(drop=True)
    D["siteyear_idx"] = np.arange(1, len(D) + 1)
    D["year_idx"] = (D.year - 2022).astype(int)                 # 2023 -> 1, 2024 -> 2
    Tk = set(zip(T.site, T.year, T.network))
    D["in_target_T"] = [int((s, y, n) in Tk) for s, y, n in zip(D.site, D.year, D.network)]
    D["target_weight"] = D.in_target_T.astype(float)            # one weight per target site-year
    assert D.in_target_T.sum() == 2256 and D[D.in_target_T == 1].network.value_counts().to_dict() == {"MS": 1461, "RSN": 795}
    assert len(T) == 2256 and len(Tk) == 2256, "T must be fully contained in training site-years"
    checks["target_T"] = D[D.in_target_T == 1].network.value_counts().to_dict()
    siteyears = D[["siteyear_idx", "site_idx", "site", "network", "year", "year_idx", "n", "n_quarters", "n_months", "n_censored",
                   "mean_lo", "mean_hi", "in_target_T", "target_weight"]]

    # ---------------- measurements
    o = obs.merge(D[["site", "year", "network", "site_idx", "siteyear_idx"]], on=["site", "year", "network"], how="inner", validate="many_to_one")
    assert o.id.is_unique, "duplicated measurement ids"
    assert len(o) == 21798 and (o.network == "RSN").sum() == 9140 and (o.network == "MS").sum() == 12658
    # each site-year's measurement count equals the Phase 2 n
    cnt = o.groupby("siteyear_idx").size().reindex(D.siteyear_idx)
    assert (cnt.values == D.n.values).all(); checks["siteyear_counts_match_n"] = True
    # units and determinand from raw pages
    raw = pd.concat([pd.read_csv(f, dtype=str, usecols=["id", "result", "unit", "determinand.notation"])
                     for f in sorted(glob.glob(str(PILOT / "raw/wqa_0180_F6_*_page*.csv")))])
    raw = raw[raw.id.isin(set(o.id))].drop_duplicates("id")
    assert len(raw) == len(o)
    assert (raw.unit == "MILLIGRAM PER LITRE").all() and (raw["determinand.notation"].astype(str).str.lstrip("0") == "180").all()
    raw["decimals"] = raw.result.str.lstrip("<>").str.split(".").str[1].fillna("").str.len()
    raw["raw_censored"] = raw.result.str.startswith("<")
    o = o.merge(raw[["id", "result", "decimals", "raw_censored"]], on="id", how="left", validate="one_to_one")
    assert (o.raw_censored == o.censored).all(), "censoring flag disagrees with raw result string"
    assert (pd.to_numeric(o.result.str.lstrip("<>")) == o.value).all(), "value disagrees with raw result string"
    assert (o.value > 0).all() and not o.greater_than.any()
    t = pd.to_datetime(o.time)
    days = np.where(t.dt.is_leap_year, 366, 365)
    o["tau"] = (t.dt.dayofyear - 1 + (t.dt.hour * 60 + t.dt.minute) / 1440.0) / days
    assert ((o.tau >= 0) & (o.tau < 1)).all()
    o["log_y"] = np.where(o.censored, np.nan, np.log(o.value))
    o["limit"] = np.where(o.censored, o.value, np.nan)
    o["log_limit"] = np.log(o["limit"])
    o["half_width"] = 0.5 * 10.0 ** (-o.decimals.astype(float))      # reporting resolution (R1 sensitivity)
    assert (o.censored == o.log_limit.notna()).all()
    checks["censored"] = int(o.censored.sum()); assert checks["censored"] == 946
    checks["censoring_limits"] = o[o.censored].limit.round(4).value_counts().to_dict()
    o = o.sort_values(["siteyear_idx", "tau", "id"]).reset_index(drop=True)
    o["obs_idx"] = np.arange(1, len(o) + 1)
    meas = o[["obs_idx", "id", "site_idx", "siteyear_idx", "site", "network", "year", "time", "month", "tau", "value", "censored",
              "log_y", "limit", "log_limit", "decimals", "half_width"]]

    # ---------------- write
    site_out = site[["site_idx", "site", "network", "ms", "region", "region_idx", "wma", "wma_idx", "mancat", "mancat_idx", "wb", "wb_idx",
                     "x_km", "y_km", "log_area", "z_logarea", "edm_n_5km", "log_edm5", "z_edm", "geology", "geology_idx", "cls", "dist",
                     "size_cls", "size_typology", "link_strahler"]]
    site_out.to_parquet(OUT / "sites.parquet", index=False)
    siteyears.to_parquet(OUT / "siteyears.parquet", index=False)
    meas.to_parquet(OUT / "measurements.parquet", index=False)
    dd = []
    desc = {
        "obs_idx": "1-based measurement index (sorted by site-year, time)", "id": "WQ Archive observation URI", "site_idx": "1-based site index (RSN first)",
        "siteyear_idx": "1-based site-year index", "tau": "time of year in [0,1): (day-of-year - 1 + fraction of day)/days in year",
        "value": "reported value, mg P/L (limit if censored)", "censored": "True if reported as '<L'", "log_y": "log value (NaN if censored)",
        "limit": "observation-specific reporting limit mg P/L (censored only)", "log_limit": "log limit", "decimals": "decimal places in reported string",
        "half_width": "half reporting resolution 0.5*10^-decimals (R1)", "ms": "1 = routine statutory (MS), 0 = RSN", "z_logarea": "standardised log modelled upstream area",
        "z_edm": "standardised log(1+storm overflows within 5 km)", "geology_idx": "1 Calcareous (ref), 2 Siliceous, 3 Organic, 4 Unknown",
        "region_idx": "1..7 in order " + ",".join(REGIONS), "year_idx": "1 = 2023, 2 = 2024", "in_target_T": "member of frozen Phase 2B target T",
        "target_weight": "standardisation weight (1 per target site-year)", "cls": "Phase 2 match class (A/B/C)", "mean_hi": "Phase 2 observed site-year sample mean (censored at limit)"}
    for tname, df in [("measurements", meas), ("sites", site_out), ("siteyears", siteyears)]:
        for c in df.columns:
            dd.append(dict(table=tname, column=c, dtype=str(df[c].dtype), n_missing=int(df[c].isna().sum()), description=desc.get(c, "")))
    pd.DataFrame(dd).to_csv(OUT / "data_dictionary.csv", index=False)
    checks["n_measurements"] = len(meas); checks["n_sites"] = len(site_out); checks["n_siteyears"] = len(siteyears)
    checks["decimals_counts"] = meas.decimals.value_counts().sort_index().to_dict()
    checks["inputs_sha256"] = inputs
    checks["outputs_sha256"] = {f: sha(OUT / f) for f in ["measurements.parquet", "sites.parquet", "siteyears.parquet", "data_dictionary.csv"]}
    json.dump(checks, open(OUT / "dataset_validation.json", "w"), indent=2, default=str)
    print(json.dumps({k: v for k, v in checks.items() if k not in ("inputs_sha256", "outputs_sha256")}, indent=1, default=str))


if __name__ == "__main__":
    main()
