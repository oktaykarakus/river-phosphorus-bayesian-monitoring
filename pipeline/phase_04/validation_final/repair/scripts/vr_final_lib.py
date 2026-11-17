"""Shared helpers for the W1 final attempt (VR_03 protocol). Imports the Phase 4B library read-only; every writer is redirected
to validation_final/repair/ and ~/track_a_scratch/phase_04_vr. Overrides are in memory only."""
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.dont_write_bytecode = True
VR = Path(__file__).resolve().parents[1]
VFDIR = VR.parent
sys.path.insert(0, str(VR / "scripts"))
sys.path.insert(0, str(VFDIR / "scripts"))
import vf_lib as V  # noqa: E402

CFG = yaml.safe_load(open(VR / "config" / "vr_w1_final.yaml"))
TEST = os.environ.get("VR_TEST") == "1"          # test mode: every output goes to scratch phase_04_vr/test, separate ledger
SCR = Path(os.path.expanduser(CFG["scratch_dir"])) / ("test" if TEST else ""); SCR.mkdir(parents=True, exist_ok=True)
ORIG = Path(os.path.expanduser("~/track_a_scratch/phase_04_vf"))
LEDGER = VR / "logs" / ("vr_test_ledger.jsonl" if TEST else "vr_cpu_ledger.jsonl"); LEDGER.parent.mkdir(exist_ok=True)
DIAG = (SCR / "diag") if TEST else (VR / "diagnostics")
TAB = (SCR / "tab") if TEST else (VR / "tables")
FOLDS_VR = TAB / "folds_vr"
for _d in (DIAG, TAB, FOLDS_VR):
    _d.mkdir(parents=True, exist_ok=True)
_R0 = resource.getrusage(resource.RUSAGE_SELF); _C0 = resource.getrusage(resource.RUSAGE_CHILDREN); _T0 = time.time()


def to_repair():
    """Point the Phase 4B library (and through it Phase 3C-F) at repair/ and the phase_04_vr scratch."""
    V.VF = VR
    V.CFG["scratch_dir"] = CFG["scratch_dir"]
    return V.redirect()


def r3_draws(fold, source):
    """source 'vf' = original Phase 4B fold fit; 'vr' = the W1 refit of this stage."""
    p = (SCR / f"{CFG['refit']['tag']}_draws.npz") if source == "vr" else (ORIG / f"vf_{fold}_R3_draws.npz")
    return dict(np.load(p))


def cpu_self_h():
    r = resource.getrusage(resource.RUSAGE_SELF); c = resource.getrusage(resource.RUSAGE_CHILDREN)
    return ((r.ru_utime - _R0.ru_utime) + (r.ru_stime - _R0.ru_stime) + (c.ru_utime - _C0.ru_utime) + (c.ru_stime - _C0.ru_stime)) / 3600


def ledger(job, cpu_h=None, **kw):
    rec = dict(job=job, cpu_h=round(cpu_self_h() if cpu_h is None else cpu_h, 4), wall_s=round(time.time() - _T0, 1),
               time=time.strftime("%Y-%m-%d %H:%M:%S"), **kw)
    with open(LEDGER, "a") as f:
        f.write(json.dumps(rec) + "\n")
    return rec


def ledger_total():
    return sum(json.loads(x)["cpu_h"] for x in open(LEDGER) if x.strip()) if LEDGER.exists() else 0.0


def remaining():
    return CFG["budget_total_cpu_h"] - ledger_total()


def save_json(obj, path):
    V.save_json(obj, path)
