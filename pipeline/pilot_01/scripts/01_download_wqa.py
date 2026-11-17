"""
P1: National pull of orthophosphate (0180) observations at river sampling points (type F6),
all sampling purposes, 2023 and 2024, from the EA Water Quality Archive API (v1.3.1).

Raw CSV pages are retained verbatim in raw/wqa_0180_F6_<year>_page<k>.csv and a provenance
log (URLs, timestamps, response headers, row counts, sha256) is written to
metadata/wqa_download_log.json.

Run from pilot_01/:  python3 -I scripts/01_download_wqa.py
"""
import hashlib, json, os, time, urllib.request, datetime

BASE = "https://environment.data.gov.uk/water-quality/data/observation"
LIMIT = 2500
OUT_RAW = "raw"
LOG = "metadata/wqa_download_log.json"
QUERIES = {
    y: {"determinand": "0180", "samplingPointType": "F6",
        "dateFrom": f"{y}-01-01", "dateTo": f"{y}-12-31"} for y in (2023, 2024)
}


def fetch(url):
    req = urllib.request.Request(url, method="POST", headers={
        "Accept": "text/csv", "User-Agent": "track-a-pilot/0.1 (research)"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read(), dict(r.headers)
        except Exception as e:  # noqa
            err = e
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"failed {url}: {err}")


def main():
    os.makedirs(OUT_RAW, exist_ok=True)
    log = {"api": BASE, "api_version_header": "1", "licence": "Open Government Licence v3.0",
           "retrieved_utc": datetime.datetime.utcnow().isoformat() + "Z", "pages": []}
    for year, params in QUERIES.items():
        skip, total = 0, None
        k = 0
        while total is None or skip < total:
            q = "&".join(f"{a}={b}" for a, b in params.items())
            url = f"{BASE}?{q}&limit={LIMIT}&skip={skip}"
            body, hdr = fetch(url)
            total = int(hdr.get("x-total-items") or hdr.get("X-Total-Items"))
            fn = f"{OUT_RAW}/wqa_0180_F6_{year}_page{k:02d}.csv"
            with open(fn, "wb") as fh:
                fh.write(body)
            nrows = body.count(b"\n") - 1
            log["pages"].append({"year": year, "url": url, "file": fn, "rows_approx": nrows,
                                 "x_total_items": total, "bytes": len(body),
                                 "sha256": hashlib.sha256(body).hexdigest(),
                                 "time_utc": datetime.datetime.utcnow().isoformat() + "Z"})
            print(year, k, skip, total, nrows, len(body))
            skip += LIMIT
            k += 1
            time.sleep(1.0)
    with open(LOG, "w") as fh:
        json.dump(log, fh, indent=1)


if __name__ == "__main__":
    main()
