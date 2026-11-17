"""Record source provenance, SHA-256 checksums of raw + processed files, schema inventory and software versions.
Run from phase_02/ (first and again at the end of the pipeline):  python -I scripts/01_provenance.py
"""
import sys, hashlib, platform, subprocess
sys.path.insert(0, __import__("os").path.dirname(__file__))
from common import *  # noqa


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ensure_dirs()
    rows = []
    for base in ["data_raw", "data_processed", "tables"]:
        for p in sorted((ROOT / base).rglob("*")):
            if p.is_file() and p.name != ".DS_Store":
                rows.append(dict(path=str(p.relative_to(ROOT)), bytes=p.stat().st_size, sha256=sha(p)))
    for p in [PILOT / "data/site_years.csv", PILOT / "data/rsn_sites_official.csv", PILOT / "data/site_covariates.csv",
              PILOT / "data/edm_outlets_2024.csv", RSN_ARD_ZIP, PILOT / "results/p5_adjusted_differences.csv"]:
        rows.append(dict(path="../pilot_01/" + str(p.relative_to(PILOT)), bytes=p.stat().st_size, sha256=sha(p)))
    pd.DataFrame(rows).to_csv(PROV / "checksums.tsv", sep="\t", index=False)
    # cached GeoPackages must be byte-identical to the retained extracted files
    chk = {}
    for src, dst in [(OS_GPKG_SRC, OS_GPKG), (WFD_GPKG_SRC, WFD_GPKG), (WMA_GPKG_SRC, WMA_GPKG)]:
        chk[src.name] = dict(source_sha256=sha(src), cache_sha256=sha(dst), identical=sha(src) == sha(dst))
    dump_json(chk, PROV / "gpkg_cache_check.json")
    # software
    import pandas, numpy, scipy, sklearn, matplotlib, geopandas, pyogrio, pyproj, shapely, networkx, statsmodels, pyarrow
    v = {m.__name__: m.__version__ for m in [pandas, numpy, scipy, sklearn, matplotlib, geopandas, pyogrio, pyproj, shapely, networkx, statsmodels, pyarrow]}
    v["python"] = sys.version; v["platform"] = platform.platform(); v["gdal"] = pyogrio.__gdal_version_string__
    v["geos"] = shapely.geos_version_string; v["interpreter"] = sys.executable
    dump_json(v, PROV / "software_versions.json")
    try:
        frz = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
        (PROV / "pip_freeze.txt").write_text(frz)
    except Exception as e:  # noqa
        log(f"pip freeze failed: {e}")
    log(f"{len(rows)} files checksummed; cache identical: {all(x['identical'] for x in chk.values())}")


if __name__ == "__main__":
    main()
