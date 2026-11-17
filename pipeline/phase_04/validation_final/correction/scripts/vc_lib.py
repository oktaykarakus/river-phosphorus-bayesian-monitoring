"""Validation-correction stage: shared helpers (VC_00). Imports the Phase 4B library read-only and redirects EVERY writer to
validation_final/correction/ and ~/track_a_scratch/phase_04_vc. Overrides are in memory only; no frozen config file is edited."""
import json
import os
import resource
import sys
import time
from pathlib import Path

import yaml

sys.dont_write_bytecode = True
VC = Path(__file__).resolve().parents[1]
VFDIR = VC.parent
sys.path.insert(0, str(VFDIR / "scripts"))
import vf_lib as V  # noqa: E402

VCFG = yaml.safe_load(open(VC / "config" / "vc_settings.yaml"))
ORIG_SCRATCH = Path(os.path.expanduser(VCFG["original_scratch"]))
NEW_SCRATCH = Path(os.path.expanduser(VCFG["scratch_dir"])); NEW_SCRATCH.mkdir(parents=True, exist_ok=True)
LEDGER = VC / "logs" / "cpu_ledger.jsonl"
_T0 = time.time(); _R0 = resource.getrusage(resource.RUSAGE_SELF); _C0 = resource.getrusage(resource.RUSAGE_CHILDREN)


def to_correction():
    """Point the Phase 4B library (and through it Phase 3C-F) at the correction directories."""
    V.VF = VC
    V.CFG["scratch_dir"] = VCFG["scratch_dir"]
    return V.redirect()


def original_draws(tag):
    import numpy as np
    return dict(np.load(ORIG_SCRATCH / f"{tag}_draws.npz"))


def new_draws(tag):
    import numpy as np
    return dict(np.load(NEW_SCRATCH / f"{tag}_draws.npz"))


def cpu_used_self():
    r = resource.getrusage(resource.RUSAGE_SELF); c = resource.getrusage(resource.RUSAGE_CHILDREN)
    return ((r.ru_utime - _R0.ru_utime) + (r.ru_stime - _R0.ru_stime) + (c.ru_utime - _C0.ru_utime) + (c.ru_stime - _C0.ru_stime)) / 3600


def ledger(job, cpu_h=None, **kw):
    rec = dict(job=job, cpu_h=round(cpu_h if cpu_h is not None else cpu_used_self(), 4), wall_s=round(time.time() - _T0, 1),
               time=time.strftime("%Y-%m-%d %H:%M:%S"), **kw)
    with open(LEDGER, "a") as f:
        f.write(json.dumps(rec) + "\n")
    return rec


def ledger_total():
    if not LEDGER.exists():
        return 0.0
    return sum(json.loads(x)["cpu_h"] for x in open(LEDGER) if x.strip())


def save_json(obj, path):
    V.save_json(obj, path)
