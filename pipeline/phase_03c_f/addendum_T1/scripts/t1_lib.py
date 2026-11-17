"""Phase 3C-F addendum T1: shared paths and helpers.

Imports the Phase 3C-F library read-only (bytecode writing disabled). Any Phase 3C-F function that writes to L.ROOT is used only
after `redirect()`, which points L.ROOT at this addendum, so no completed Phase 3C-F output can be overwritten.
"""
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
ADD = Path(__file__).resolve().parents[1]                 # phase_03c_f/addendum_T1
P3F = ADD.parent                                           # phase_03c_f
sys.path.insert(0, str(P3F / "scripts"))
import p3f_lib as L  # noqa: E402

import yaml  # noqa: E402

T1CFG = yaml.safe_load(open(ADD / "config" / "t1_settings.yaml")) if (ADD / "config" / "t1_settings.yaml").exists() else {}
ORIG_TAG = "main_R3"
SEG_TAG = "T1_segment"
COMB_TAG = "T1_combined"
NCH = 4


def redirect():
    """Send every Phase 3C-F writer (L.ROOT/diagnostics, tables, figures) to the addendum."""
    L.ROOT = ADD
    return L


def scratch():
    return L.scratch_dir()


def orig_csvs():
    return sorted((scratch() / f"fit_{ORIG_TAG}").glob("*_[1-4].csv"), key=lambda p: int(p.stem.split("_")[-1]))


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def csv_config(path):
    """Header configuration and adaptation block of a CmdStan CSV. Layout: config comments, column header, adaptation comments,
    draws. Reads only up to the first draw row."""
    cfg, step, metric, lines, seen_header = {}, None, None, [], False
    with open(path) as f:
        for ln in f:
            if ln.startswith("#"):
                lines.append(ln)
            elif not seen_header:
                seen_header = True
            else:
                break
    for i, ln in enumerate(lines):
        m = re.match(r"#\s+(\w+) = (.*)", ln)
        if m:
            cfg.setdefault(m.group(1), m.group(2).replace(" (Default)", "").strip())
        if ln.startswith("# Step size ="):
            step = float(ln.split("=")[1])
        if ln.startswith("# Diagonal elements of inverse mass matrix"):
            metric = np.array([float(x) for x in lines[i + 1].lstrip("#").split(",")])
    return cfg, step, metric


def last_draw(path):
    """Header and final row (as strings, full printed precision) of a CmdStan CSV, read from the end of the file."""
    header = None
    with open(path) as f:
        for ln in f:
            if not ln.startswith("#"):
                header = ln.strip().split(",")
                break
    with open(path, "rb") as f:
        f.seek(0, 2); size = f.tell(); block = min(size, 1 << 22)
        f.seek(size - block); tail = f.read().decode().splitlines()
    rows = [r for r in tail if r and not r.startswith("#")]
    return header, rows[-1].split(",")


def save_json(obj, path):
    Path(path).write_text(json.dumps(obj, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
