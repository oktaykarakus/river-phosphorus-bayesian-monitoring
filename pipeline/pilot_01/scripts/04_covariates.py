"""
P4 covariates (light-weight, official sources only).

1. Storm-overflow proximity from the EA EDM 2024 Storm Overflow Annual Return (OGL; data.gov.uk,
   'Event Duration Monitoring - Storm Overflows - Annual Returns'). For each monitoring site:
     edm_n_2km, edm_n_5km      number of storm-overflow outlets within 2 / 5 km (EUCLIDEAN, not network distance)
     edm_spillhrs_5km          total 2024 spill duration (h) of outlets within 5 km
     edm_nearest_km            distance to nearest outlet
   Euclidean proximity ignores flow direction (an upstream/downstream-agnostic pressure proxy).
   Outlets in Wales are excluded by the England-only site domain implicitly (distances only).

Input : raw/edm/EDM_2024_Storm_Overflow_Annual_Return.zip, data/site_years.csv
Output: data/edm_outlets_2024.csv, data/site_covariates.csv, metadata/edm_parse_log.json

Run from pilot_01/:  python3 -I scripts/04_covariates.py
"""
import json, re, zipfile, io
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

# OS National Grid 100 km square letters -> (e, n) offsets in 100 km units
_L = "ABCDEFGHJKLMNOPQRSTUVWXYZ"  # no I


def ngr_to_en(ngr):
    if not isinstance(ngr, str):
        return (np.nan, np.nan)
    s = re.sub(r"\s+", "", ngr.upper())
    m = re.fullmatch(r"([A-HJ-Z]{2})(\d+)", s)
    if not m or len(m.group(2)) % 2:
        return (np.nan, np.nan)
    l1, l2 = _L.index(m.group(1)[0]), _L.index(m.group(1)[1])
    e100 = ((l1 - 2) % 5) * 5 + (l2 % 5)
    n100 = (19 - (l1 // 5) * 5) - (l2 // 5)
    digits = m.group(2); k = len(digits) // 2
    scale = 10 ** (5 - k)
    e = e100 * 100000 + int(digits[:k]) * scale
    n = n100 * 100000 + int(digits[k:]) * scale
    return (float(e), float(n))


def hms_to_hours(x):
    """'hh:mm:ss' (hours may exceed 24) or datetime.time / timedelta -> hours; else NaN."""
    import datetime as _dt
    if isinstance(x, _dt.timedelta):
        return x.total_seconds() / 3600
    if isinstance(x, _dt.time):
        return x.hour + x.minute / 60 + x.second / 3600
    if isinstance(x, (int, float)) and not pd.isna(x):
        return float(x) * 24 if x < 50 else np.nan   # Excel day fraction
    m = re.fullmatch(r"\s*(\d+):(\d{1,2}):(\d{1,2})\s*", str(x))
    return int(m.group(1)) + int(m.group(2)) / 60 + int(m.group(3)) / 3600 if m else np.nan


def load_edm(path):
    log = {}
    z = zipfile.ZipFile(path)
    log["zip_members"] = z.namelist()
    frames = []
    for name in z.namelist():
        low = name.lower()
        if low.endswith((".xlsx", ".xls")):
            xl = pd.ExcelFile(io.BytesIO(z.read(name)))
            for sh in xl.sheet_names:
                df = xl.parse(sh, header=1)   # row 0 is a title/banner row in the EA template
                # header may not be on first row: find a row containing 'NGR'
                if not any("ngr" in str(c).lower() for c in df.columns):
                    raw = xl.parse(sh, header=None)
                    hdr = raw.index[raw.apply(lambda r: r.astype(str).str.contains("NGR", case=False).any(), axis=1)]
                    if len(hdr) == 0:
                        continue
                    df = xl.parse(sh, header=int(hdr[0]))
                df["__sheet"] = f"{name}:{sh}"
                frames.append(df)
        elif low.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(z.read(name)), encoding_errors="replace")
            df["__sheet"] = name
            frames.append(df)
    log["sheets_with_ngr"] = [f["__sheet"].iloc[0] for f in frames if len(f)]
    return frames, log


def main():
    frames, log = load_edm("raw/edm/EDM_2024_Storm_Overflow_Annual_Return.zip")
    rows = []
    for df in frames:
        cols = {c: str(c).lower() for c in df.columns}
        ngr_cols = [c for c, l in cols.items() if "ngr" in l]
        # prefer an outlet/discharge NGR
        pref = [c for c in ngr_cols if "outlet" in cols[c] or "discharge" in cols[c]] or ngr_cols
        dur = [c for c, l in cols.items() if l.startswith("total duration")]
        cnt = [c for c, l in cols.items() if l.startswith("counted spills")]
        if not pref:
            continue
        log.setdefault("columns_used", []).append({"sheet": df["__sheet"].iloc[0], "ngr": str(pref[0]),
                                                   "duration": str(dur[0]) if dur else None,
                                                   "count": str(cnt[0]) if cnt else None, "rows": int(len(df))})
        en = df[pref[0]].map(ngr_to_en)
        out = pd.DataFrame({"easting": [x[0] for x in en], "northing": [x[1] for x in en],
                            "spill_hrs": df[dur[0]].map(hms_to_hours) if dur else np.nan,
                            "spill_count": pd.to_numeric(df[cnt[0]], errors="coerce") if cnt else np.nan,
                            "sheet": df["__sheet"]})
        rows.append(out)
    edm = pd.concat(rows, ignore_index=True)
    log["outlet_rows"] = int(len(edm))
    log["outlet_rows_unparsed_ngr"] = int(edm.easting.isna().sum())
    edm = edm.dropna(subset=["easting", "northing"])
    log["outlets_dup_location"] = int(edm.duplicated(["easting", "northing"]).sum())
    edm.to_csv("data/edm_outlets_2024.csv", index=False)

    sy = pd.read_csv("data/site_years.csv")
    sites = sy.drop_duplicates("site")[["site", "easting", "northing"]].reset_index(drop=True)
    tree = cKDTree(edm[["easting", "northing"]].values)
    X = sites[["easting", "northing"]].values
    d, _ = tree.query(X, k=1)
    sites["edm_nearest_km"] = d / 1e3
    for r in (2000, 5000):
        nb = tree.query_ball_point(X, r=r)
        sites[f"edm_n_{r // 1000}km"] = [len(x) for x in nb]
        if r == 5000:
            hrs = edm["spill_hrs"].fillna(0).values
            sites["edm_spillhrs_5km"] = [float(hrs[x].sum()) for x in nb]
    sites.drop(columns=["easting", "northing"]).to_csv("data/site_covariates.csv", index=False)
    with open("metadata/edm_parse_log.json", "w") as fh:
        json.dump(log, fh, indent=1, default=str)
    print(json.dumps(log, indent=1, default=str))
    print(sites.describe())


if __name__ == "__main__":
    # sanity check of the NGR parser on known references
    assert ngr_to_en("TQ3000080000") == (530000.0, 180000.0)
    assert ngr_to_en("SU 41 15") == (441000.0, 115000.0)
    main()
