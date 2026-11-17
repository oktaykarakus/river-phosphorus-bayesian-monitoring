"""
P1/P2 integrity: cross-check RSN orthophosphate (det 0180) 2023-2024 in the official ARD
'RSN Water Quality Samples.csv' against the API pull (data/observations_clean.csv, network RSN).
Output: metadata/ard_api_crosscheck.json
Run from pilot_01/: python3 -I scripts/02b_crosscheck_ard.py
"""
import io, json, zipfile
import pandas as pd
z = zipfile.ZipFile("raw/rsn_ard/NCEA_River_Surveillance_Network_ARD_Open.zip")
f = z.open("NCEA_River_Surveillance_Network_ARD_Open/RSN Water Quality Samples.csv")
parts = []
for ch in pd.read_csv(io.TextIOWrapper(f, encoding="utf-8-sig"), dtype=str, chunksize=500_000):
    ch = ch[ch.det_code.str.lstrip("0") == "180"]
    parts.append(ch)
a = pd.concat(parts)
a["year"] = a.sample_datetime.str[:4]
a = a[a.year.isin(["2023", "2024"])]
ref = pd.read_csv("data/rsn_sites_official.csv", dtype=str)[["network_id", "wims_id", "wims_region_code"]]
a["api_site"] = a.merge(pd.read_csv("data/rsn_sites_official.csv", dtype=str)[["network_id", "wims_region_code"]]
                        .drop_duplicates("network_id"), on="network_id", how="left")["wims_region_code"].values + "-" + a.wims_id
a["time"] = pd.to_datetime(a.sample_datetime)
o = pd.read_csv("data/observations_clean.csv", dtype={"site": str})
o = o[o.site.isin(set(pd.read_csv("data/rsn_sites_official.csv", dtype=str).api_notation)) & (o.year.isin([2023, 2024]))]
o["time"] = pd.to_datetime(o.time)
m = a.merge(o, left_on=["api_site", "time"], right_on=["site", "time"], how="outer", indicator=True)
both = m[m._merge == "both"]
val_a = pd.to_numeric(both.meas_result, errors="coerce")
out = {
    "ard_rows_0180_2023_24": int(len(a)),
    "ard_purpose_codes": a.purpose_code.value_counts().to_dict(),
    "ard_units": a.unit_short_desc.value_counts().to_dict(),
    "ard_material": a.sample_material_desc.value_counts().to_dict(),
    "api_rows_at_rsn_sites_2023_24_any_purpose": int(len(o)),
    "api_rows_at_rsn_sites_by_network": o.network.value_counts().to_dict(),
    "matched_on_site_and_time": int(len(both)),
    "ard_only": int((m._merge == "left_only").sum()),
    "api_only": int((m._merge == "right_only").sum()),
    "api_only_by_purpose": m[m._merge == "right_only"].samplingPurpose.value_counts().to_dict(),
    "ard_only_by_purpose": m[m._merge == "left_only"].purpose_code.value_counts().to_dict(),
    "matched_value_abs_diff_max": float((val_a - both.value).abs().max()),
    "matched_value_exact_equal_frac": float(((val_a - both.value).abs() < 1e-9).mean()),
    "matched_censor_flag_agree_frac": float(((both.meas_sign == "<") == both.censored.astype(bool)).mean()),
}
json.dump(out, open("metadata/ard_api_crosscheck.json", "w"), indent=1, default=str)
print(json.dumps(out, indent=1, default=str))
