"""
P2a: Official RSN site list.

Primary source: EA "River Surveillance Network analysis ready data (NCEA)" open ARD zip
(raw/rsn_ard/NCEA_River_Surveillance_Network_ARD_Open.zip; OGL v3). The RSN Sites table
(and any site-version history table) provides wims_id = Water Quality Archive sampling-point id.

Fallback (ONLY if the ARD zip is unavailable; flagged in output): sampling points whose notation
matches '{region}-RSNnnnn' in the WQ archive pull.

Output: data/rsn_sites_official.csv (wims_id, network_id, current_panel, site_status,
        replacement_site_flag, waterbody_id, source_table, source) ; metadata/rsn_site_list_log.json
Run from pilot_01/: python3 -I scripts/02a_extract_rsn_sites.py
"""
import io, json, os, re, zipfile, glob
import pandas as pd

ZIP = "raw/rsn_ard/NCEA_River_Surveillance_Network_ARD_Open.zip"


def from_ard():
    """Official RSN site list = union of 'RSN Sites.csv' (current versions) and 'RSN Sites Archive.csv'
    (previous versions). API sampling-point notation = wims_region_code + '-' + wims_id."""
    z = zipfile.ZipFile(ZIP)
    pre = "NCEA_River_Surveillance_Network_ARD_Open/"
    log = {"source": "EA RSN ARD (official)", "members": z.namelist()}
    frames = []
    for name in ["RSN Sites.csv", "RSN Sites Archive.csv"]:
        df = pd.read_csv(io.BytesIO(z.read(pre + name)), dtype=str, encoding="utf-8-sig")
        df["source_table"] = name
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["api_notation"] = df["wims_region_code"].str.strip() + "-" + df["wims_id"].str.strip()
    cat = pd.read_csv(io.BytesIO(z.read(pre + "RSN Site Catchments.csv")), dtype=str, encoding="utf-8-sig")
    log["rows_sites_plus_archive"] = int(len(df))
    log["unique_network_id"] = int(df.network_id.nunique())
    log["unique_api_notation"] = int(df.api_notation.nunique())
    log["site_status_current_versions"] = df[df.source_table == "RSN Sites.csv"].site_status.value_counts().to_dict()
    log["waterbody_cat_current_versions"] = df[df.source_table == "RSN Sites.csv"].waterbody_cat.value_counts(dropna=False).to_dict()
    log["replacement_flag_current_versions"] = df[df.source_table == "RSN Sites.csv"].replacement_site_flag.value_counts().to_dict()
    # one row per api_notation: prefer the current-version record
    df["_ord"] = (df.source_table != "RSN Sites.csv").astype(int)
    one = df.sort_values(["api_notation", "_ord"]).drop_duplicates("api_notation").drop(columns="_ord")
    one = one.merge(cat[["network_id", "rbd_name", "mancat_name", "opcat_name"]].drop_duplicates("network_id"),
                    on="network_id", how="left")
    one["wims_id"] = one["api_notation"]          # key used downstream = API sampling-point notation
    one["source"] = "EA RSN ARD (official)"
    return one, log


def from_pattern():
    files = sorted(glob.glob("raw/wqa_0180_F6_*_page*.csv"))
    pts = pd.concat([pd.read_csv(f, usecols=["samplingPoint.notation"], dtype=str) for f in files])
    ids = sorted(set(pts["samplingPoint.notation"][pts["samplingPoint.notation"].str.contains(r"-RSN\d{4}$")]))
    df = pd.DataFrame({"wims_id": ids, "source": "FALLBACK notation pattern -RSNnnnn"})
    return df, {"source": "FALLBACK pattern", "unique_wims_id": len(ids)}


def main():
    if os.path.exists(ZIP) and zipfile.is_zipfile(ZIP):
        df, log = from_ard()
    else:
        df, log = from_pattern()
    df.to_csv("data/rsn_sites_official.csv", index=False)
    with open("metadata/rsn_site_list_log.json", "w") as fh:
        json.dump(log, fh, indent=1)
    print(json.dumps(log, indent=1))


if __name__ == "__main__":
    main()
