"""Shared paths, constants and small helpers for Track A Phase 2.

All scripts are run from phase_02/ with the `data-science` conda interpreter:
    /opt/anaconda3/envs/data-science/bin/python -I scripts/<script>.py
(-I isolates the interpreter; the scripts directory is added to sys.path explicitly below.)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]          # phase_02/
PILOT = ROOT.parent / "pilot_01"                     # frozen Pilot 01 (read-only)
RAW = ROOT / "data_raw"
PROC = ROOT / "data_processed"
TAB = ROOT / "tables"
FIG = ROOT / "figures"
PROV = ROOT / "provenance"

OS_GPKG_SRC = RAW / "os_open_rivers/extracted/Data/oprvrs_gb.gpkg"
WFD_GPKG_SRC = RAW / "wfd/extracted/Water_Framework_Directive_WFD_River_Water_Body_Catchments_Cycle_3_Classification_2022_Full_Resolution.gpkg"
WMA_GPKG_SRC = RAW / "ea_areas/extracted/Administrative_Boundaries_Water_Management_Areas.gpkg"


def _local_copy(src: Path) -> Path:
    """GeoPackages are SQLite files; random page reads from the project's USB volume are extremely slow.
    Read them from a byte-identical copy on local disk (size + mtime checked; SHA-256 recorded in provenance)."""
    import shutil, tempfile
    cache = Path(os.environ.get("TRACKA_CACHE", Path(tempfile.gettempdir()) / "trackA_phase2_cache"))
    cache.mkdir(parents=True, exist_ok=True)
    dst = cache / src.name
    if not src.exists():
        return src
    if not dst.exists() or dst.stat().st_size != src.stat().st_size or dst.stat().st_mtime < src.stat().st_mtime:
        shutil.copyfile(src, dst)
    return dst


OS_GPKG = _local_copy(OS_GPKG_SRC)
WFD_GPKG = _local_copy(WFD_GPKG_SRC)
WMA_GPKG = _local_copy(WMA_GPKG_SRC)
NRFA_JSON = RAW / "nrfa/nrfa_station_info.json"
RSN_ARD_ZIP = PILOT / "raw/rsn_ard/NCEA_River_Surveillance_Network_ARD_Open.zip"

CRS = "EPSG:27700"
THRESH = [0.05, 0.10, 0.20]
RULES = {
    "any": lambda d: d["n"] >= 1,
    "primary": lambda d: (d["n"] >= 6) & (d["n_quarters"] >= 3),
    "strict": lambda d: (d["n"] >= 10) & (d["n_quarters"] == 4),
    "monthly": lambda d: (d["n"] >= 11) & (d["n_months"] >= 10) & (d["n_quarters"] == 4),
}
# Pilot 01 pooled primary-rule headline (README / P03) used for the reproduction test
PILOT_BASELINE = {0.05: (0.554, 0.778, 0.225), 0.10: (0.354, 0.559, 0.205), 0.20: (0.145, 0.309, 0.164)}

# UKTAG / WFD river typology catchment-size bands (km^2): Extra Small <10, Small 10-100,
# Medium 100-1000, Large 1000-10000 (Very Large >10000 does not occur in England's typology field)
SIZE_BREAKS = [0, 10, 100, 1000, np.inf]
SIZE_LABELS = ["XS (<10 km²)", "S (10–100)", "M (100–1,000)", "L (≥1,000)"]
TYPO_SIZE_MAP = {"Extra Small": SIZE_LABELS[0], "Small": SIZE_LABELS[1], "Medium": SIZE_LABELS[2], "Large": SIZE_LABELS[3]}

# Water-management-area name crosswalk: pilot WQ-API 'area' string -> WMA 'long_name'
AREA_TO_WMA = {
    "ANGLIAN - CAMBS AND BEDFORDSHIRE": "Cambridgeshire and Bedfordshire",
    "ANGLIAN - ESSEX NORFOLK AND SUFFOLK": "Essex Norfolk and Suffolk",
    "ANGLIAN - LINCS AND NORTHANTS": "Lincolnshire and Northamptonshire",
    "MIDLANDS - DERBYS NOTTS AND LEICS": "Derbyshire Nottinghamshire and Leicestershire",
    "MIDLANDS - SHROPS HEREF WORCS AND GLOS": "Shropshire Herefordshire Worcestershire and Gloucestershire",
    "MIDLANDS - STAFFS WARKS AND WEST MIDS": "Staffordshire Warwickshire and West Midlands",
    "NORTH EAST - NORTHUMBERLAND DURHAM AND TEES": "Northumberland Durham and Tees",
    "NORTH EAST - YORKSHIRE": "Yorkshire",
    "NORTH WEST - CUMBRIA AND LANCASHIRE": "Cumbria and Lancashire",
    "NORTH WEST - GTR MANCS MERSEY AND CHES": "Greater Manchester Merseyside and Cheshire",
    "SOUTH WEST - DEVON AND CORNWALL": "Devon and Cornwall",
    "SOUTH WEST - WESSEX": "Wessex",
    "SOUTHERN - KENT AND SOUTH LONDON": "Kent and South London",
    "SOUTHERN - SOLENT AND SOUTH DOWNS": "Solent and South Downs",
    "THAMES - HERTS AND NORTH LONDON": "Hertfordshire and North London",
    "THAMES - KENT AND SOUTH LONDON": "Kent and South London",
    "THAMES - WEST THAMES": "West Thames",
}

SEED = 20261110


def ensure_dirs():
    for p in (PROC, TAB, FIG, PROV):
        p.mkdir(parents=True, exist_ok=True)


def size_class(area_km2):
    return pd.cut(pd.Series(area_km2), SIZE_BREAKS, labels=SIZE_LABELS, right=False)


def blocks(e, n, m):
    """Square BNG grid-cell label, identical to Pilot 01 (easting // m, northing // m)."""
    ex = pd.Series(np.floor(np.asarray(e, float) / m).astype(int)).astype(str)
    ny = pd.Series(np.floor(np.asarray(n, float) / m).astype(int)).astype(str)
    return (ex + "_" + ny).values


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (c - h, c + h)


def dump_json(obj, path):
    def conv(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return None if np.isnan(o) else float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=conv)


def log(msg):
    print(msg, flush=True)
