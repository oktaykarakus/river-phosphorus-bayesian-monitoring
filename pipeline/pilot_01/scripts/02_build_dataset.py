"""
P2: Build the observation-level and site-year analytical datasets.

Inputs
  raw/wqa_0180_F6_<year>_page*.csv        national pull (01_download_wqa.py)
  data/rsn_sites_official.csv             official RSN site list (02a_extract_rsn_sites.py)

Outputs
  data/observations_clean.csv             one row per observation, with network label and censoring parsed
  data/site_years.csv                     one row per (network, site, year)
  metadata/p2_qc_counts.json              exclusion / QC counts

Network definitions (per observation; see P02_dataset_construction.md)
  RSN       : sampling point in the official RSN site list AND purpose MN
  MS        : purpose MS AND sampling point NOT in the RSN site list         (primary "routine statutory")
  MN_OTHER  : purpose MN AND sampling point NOT in the RSN site list         (non-RSN national-policy monitoring)
  ROUTINE_BROAD = MS or MN_OTHER (pooled at the site-year level)            (sensitivity definition)

Censoring: '<x' results are kept as left-censored with limit x. Annual statistics are reported as
bounds: lower bound sets censored values to 0, upper bound sets them to their limit. No substitution.

Run from pilot_01/:  python3 -I scripts/02_build_dataset.py
"""
import glob, json
import numpy as np
import pandas as pd

RIVER = "RIVER / RUNNING SURFACE WATER"
PURPOSE_MN = "MONITORING  (NATIONAL AGENCY POLICY)"
PURPOSE_MS = "ENVIRONMENTAL MONITORING STATUTORY (EU DIRECTIVES)"


def load_raw():
    files = sorted(glob.glob("raw/wqa_0180_F6_*_page*.csv"))
    df = pd.concat([pd.read_csv(f, dtype=str) for f in files], ignore_index=True)
    return df


def parse(df):
    qc = {"raw_rows": int(len(df))}
    qc["dup_ids"] = int(df["id"].duplicated().sum())
    df = df.drop_duplicates("id")
    qc["non_river_material_excluded"] = int((df.sampleMaterialType != RIVER).sum())
    df = df[df.sampleMaterialType == RIVER].copy()
    assert (df["unit"] == "MILLIGRAM PER LITRE").all()
    assert (df["determinand.notation"].astype(str).str.zfill(4) == "0180").all()
    r = df["result"].str.strip()
    df["censored"] = r.str.startswith("<")
    df["greater_than"] = r.str.startswith(">")
    num = pd.to_numeric(r.str.lstrip("<>"), errors="coerce")
    qc["unparseable_results_excluded"] = int(num.isna().sum())
    df["value"] = num
    df = df[df["value"].notna()].copy()
    qc["negative_values_excluded"] = int((df["value"] < 0).sum())
    df = df[df["value"] >= 0]
    df["time"] = pd.to_datetime(df["phenomenonTime"])
    df["year"] = df["time"].dt.year
    df["month"] = df["time"].dt.month
    df["quarter"] = df["time"].dt.quarter
    df = df.rename(columns={"samplingPoint.notation": "site", "samplingPoint.prefLabel": "label",
                            "samplingPoint.easting": "easting", "samplingPoint.northing": "northing",
                            "samplingPoint.region": "region", "samplingPoint.area": "area",
                            "samplingPoint.subArea": "subarea",
                            "samplingPoint.samplingPointStatus": "point_status"})
    df["easting"] = df["easting"].astype(float)
    df["northing"] = df["northing"].astype(float)
    qc["river_obs"] = int(len(df))
    qc["dup_site_time"] = int(df.duplicated(["site", "time"]).sum())
    return df, qc


def label_networks(df, rsn_ids):
    in_rsn = df["site"].isin(rsn_ids)
    pur = df["samplingPurpose"]
    net = np.where(in_rsn & (pur == PURPOSE_MN), "RSN",
          np.where(~in_rsn & (pur == PURPOSE_MS), "MS",
          np.where(~in_rsn & (pur == PURPOSE_MN), "MN_OTHER", "OTHER")))
    df["network"] = net
    df["site_in_rsn_list"] = in_rsn
    df["rsn_name_pattern"] = df["site"].str.contains(r"-RSN\d{4}$", regex=True)
    return df


def site_year(df, network_name, mask):
    d = df[mask].copy()
    d["v_lo"] = np.where(d["censored"], 0.0, d["value"])
    d["v_hi"] = d["value"]
    g = d.groupby(["site", "year"])
    out = g.agg(n=("value", "size"), n_censored=("censored", "sum"),
                max_censor_limit=("value", lambda s: np.nan),
                mean_lo=("v_lo", "mean"), mean_hi=("v_hi", "mean"),
                median_lo=("v_lo", "median"), median_hi=("v_hi", "median"),
                n_months=("month", "nunique"), n_quarters=("quarter", "nunique"),
                first_date=("time", "min"), last_date=("time", "max"),
                easting=("easting", "first"), northing=("northing", "first"),
                region=("region", "first"), area=("area", "first"), subarea=("subarea", "first"),
                label=("label", "first"), point_status=("point_status", "first"),
                n_purposes=("samplingPurpose", "nunique")).reset_index()
    cl = d[d["censored"]].groupby(["site", "year"])["value"].max()
    out = out.drop(columns="max_censor_limit").merge(cl.rename("max_censor_limit"),
                                                     left_on=["site", "year"], right_index=True, how="left")
    q = d.groupby(["site", "year"])["quarter"].apply(lambda s: "".join(str(x) for x in sorted(set(s))))
    out = out.merge(q.rename("quarters"), left_on=["site", "year"], right_index=True)
    m = d.groupby(["site", "year"])["month"].apply(lambda s: ",".join(str(x) for x in sorted(set(s))))
    out = out.merge(m.rename("months"), left_on=["site", "year"], right_index=True)
    out["network"] = network_name
    lab = out["label"].str.upper()
    out["label_mentions_stw"] = lab.str.contains(r"\bSTW\b|S\.T\.W|SEWAGE|\bWWTW\b|\bSTF\b|\bWRC\b|\bWWTP\b", regex=True)
    out["label_ds"] = lab.str.contains(r"\bD/?S\b|DOWNSTREAM", regex=True)
    out["label_us"] = lab.str.contains(r"\bU/?S\b|UPSTREAM", regex=True)
    return out


def main():
    rsn = pd.read_csv("data/rsn_sites_official.csv", dtype=str)
    rsn_ids = set(rsn["wims_id"].dropna().str.strip())
    df, qc = parse(load_raw())
    df = label_networks(df, rsn_ids)
    qc["rsn_official_wims_ids"] = len(rsn_ids)
    qc["obs_by_network"] = df["network"].value_counts().to_dict()
    qc["rsn_name_pattern_not_in_official_list_obs"] = int((df.rsn_name_pattern & ~df.site_in_rsn_list).sum())
    qc["official_rsn_site_non_MN_obs"] = df[df.site_in_rsn_list & (df.network == "OTHER")]["samplingPurpose"].value_counts().to_dict()
    qc["official_rsn_site_MS_obs"] = int((df.site_in_rsn_list & (df.samplingPurpose == PURPOSE_MS)).sum())
    qc["censored_limits"] = df[df.censored]["value"].value_counts().sort_index().to_dict()
    cols = ["id", "site", "label", "easting", "northing", "region", "area", "subarea", "point_status",
            "time", "year", "month", "quarter", "samplingPurpose", "network", "site_in_rsn_list",
            "value", "censored", "greater_than"]
    df[cols].to_csv("data/observations_clean.csv", index=False)

    sy = pd.concat([
        site_year(df, "RSN", df.network == "RSN"),
        site_year(df, "MS", df.network == "MS"),
        site_year(df, "MN_OTHER", df.network == "MN_OTHER"),
        site_year(df, "ROUTINE_BROAD", df.network.isin(["MS", "MN_OTHER"])),
    ], ignore_index=True)
    attrs = rsn.rename(columns={"wims_id": "site"})[["site", "network_id", "current_panel", "site_status",
                                                      "waterbody_cat", "replacement_site_flag"]]
    attrs = attrs.add_prefix("rsn_").rename(columns={"rsn_site": "site"})
    sy = sy.merge(attrs, on="site", how="left")
    sy.to_csv("data/site_years.csv", index=False)
    qc["site_years_by_network"] = sy["network"].value_counts().to_dict()
    with open("metadata/p2_qc_counts.json", "w") as fh:
        json.dump(qc, fh, indent=1, default=str)
    print(json.dumps(qc, indent=1, default=str))


if __name__ == "__main__":
    main()
