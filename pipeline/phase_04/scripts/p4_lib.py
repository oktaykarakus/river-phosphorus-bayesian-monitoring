"""Track A Phase 4: shared paths and helpers.

Imports the Phase 3C-F library (and through it Phase 3C-R / 3C / 3B / 3A) READ-ONLY with bytecode writing disabled.
`redirect()` points every Phase 3C-F writer (L.ROOT/diagnostics, tables, figures) at phase_04, so no earlier output can be
overwritten. Draws are read from the authoritative scratch archive (main_R3 + T1 combined, 4 x 5,000).
"""
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
P4 = Path(__file__).resolve().parents[1]
TA = P4.parent
P3F = TA / "phase_03c_f"
sys.path.insert(0, str(P3F / "scripts"))
import p3f_lib as L  # noqa: E402
import yaml  # noqa: E402

CFG = yaml.safe_load(open(P4 / "config" / "p4_settings.yaml"))
TAG = "T1_combined"
THRESH = [0.05, 0.1, 0.2]
B = L.B; C = L.C


def redirect():
    L.ROOT = P4
    for d in ["diagnostics", "tables", "figures"]:
        (P4 / d).mkdir(exist_ok=True)
    return L


def scratch() -> Path:
    p = Path(os.path.expanduser(CFG["scratch_dir"])); p.mkdir(parents=True, exist_ok=True)
    return p


def load_post(n=None):
    """Authoritative posterior (dict of arrays, 20,000 draws, chain-major). n: evenly thinned subset of n draws."""
    D = dict(np.load(L.scratch_dir() / f"{TAG}_draws.npz"))
    if n is None:
        return D
    S = len(D["lp__"]); idx = np.round(np.linspace(0, S - 1, n)).astype(int)
    return {k: (v[idx] if (np.ndim(v) >= 1 and len(v) == S) else v) for k, v in D.items()}


def model_config():
    return json.load(open(P3F / "addendum_T1/diagnostics/T1_combined_sampling_diagnostics.json"))["model_config"]


def sha(p, buf=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while (b := f.read(buf)):
            h.update(b)
    return h.hexdigest()


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(obj, open(path, "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))


def band(x, lo=.025, hi=.975):
    x = np.asarray(x).ravel()
    return dict(median=float(np.median(x)), lo95=float(np.quantile(x, lo)), hi95=float(np.quantile(x, hi)), mean=float(x.mean()), sd=float(x.std()))
