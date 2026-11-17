"""Track A validation repair, VR_04: W1 final-attempt preflight (NO sampling, NO scoring).

1. Resumability of the terminated corrective refit vc_W1_R3 (D7): what each preserved CmdStan CSV contains (configuration,
   adaptation block, complete draw rows, truncated last row), and what CmdStan 2.40 does not keep (RNG state).
2. Input identity: a fresh refit with the agreed VC_00 settings rebuilds exactly the inputs of the morning run
   (inits byte-identical, same executable hash, same pilot metric and initial step, same seed/settings).
Writes repair/diagnostics/VR_w1_preflight.json."""
import hashlib
import json
import os
import re
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
VR = Path(__file__).resolve().parents[1]
VFDIR = VR.parent
sys.path.insert(0, str(VFDIR / "scripts"))
import vf_lib as V  # noqa: E402

PART = VFDIR / "correction/partial_W1_refit"
VCCFG = __import__("yaml").safe_load(open(VFDIR / "correction/config/vc_settings.yaml"))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def csv_info(p):
    cfg, step_block, metric, header, ncol_rows, last_complete = {}, None, None, None, [], None
    lines = open(p).read().splitlines()
    for i, ln in enumerate(lines):
        if ln.startswith("#"):
            m = re.match(r"#\s+(\w+) = (.*)", ln)
            if m:
                cfg.setdefault(m.group(1), m.group(2).replace(" (Default)", "").strip())
            if ln.startswith("# Step size ="):
                step_block = ln.split("=")[1].strip()
            if ln.startswith("# Diagonal elements"):
                metric = lines[i + 1].lstrip("#").split(",")
        elif header is None:
            header = ln.split(",")
        else:
            ncol_rows.append(len(ln.split(",")))
    H = len(header) if header else 0
    complete = [k for k, n in enumerate(ncol_rows) if n == H]
    out = dict(seed=cfg.get("seed"), id=cfg.get("id"), num_warmup=cfg.get("num_warmup"), num_samples=cfg.get("num_samples"),
               adapt_delta=cfg.get("delta"), sig_figs=cfg.get("sig_figs"), save_metric=cfg.get("save_metric"),
               adaptation_block_present=step_block is not None, step_size_block=step_block,
               inv_metric_length=len(metric) if metric else 0,
               inv_metric_sig_digits=(max(len(x.strip().replace("-", "").replace(".", "").lstrip("0").split("e")[0]) for x in metric) if metric else None),
               n_columns=H, n_data_rows=len(ncol_rows), n_complete_draws=len(complete),
               n_truncated_rows=len(ncol_rows) - len(complete))
    if complete:
        last = lines[[i for i, ln in enumerate(lines) if not ln.startswith("#")][1 + complete[-1]]].split(",")
        j = header.index("stepsize__"); d = header.index("divergent__")
        out["stepsize_column_last_complete"] = last[j]
        out["divergences_in_complete_draws"] = int(sum(float(lines[[i for i, ln in enumerate(lines) if not ln.startswith("#")][1 + k]].split(",")[d])
                                                        for k in complete))
    return out


def main():
    res = dict(resumability={}, input_identity={})
    for k in range(1, 5):
        res["resumability"][f"chain_{k}"] = csv_info(PART / f"track_a_m2rn_r3-20261116075950_{k}.csv")
    res["resumability"]["rng_state_saved"] = False
    res["resumability"]["note"] = ("CmdStan 2.40 writes no RNG state and has no checkpoint/restart; only seed + chain id are accepted. "
                                   "save_metric was off, so the adapted inverse metric exists only at the printed precision.")
    # input identity for a fresh refit (same function path as vc_01_refit_w1.py)
    V.CFG["fits"]["vr_W1_R3"] = dict(kind="fold", fold="W1", model="R3")
    fi = V.fit_inputs("vr_W1_R3")
    with tempfile.TemporaryDirectory() as td:
        new = V.r3_inits(fi, td)
        same_inits = [sha(p) == sha(PART / f"init_{k}.json") for k, p in enumerate(new, start=1)]
    m, step = V.r3_metric(fi)
    orig = json.load(open(VFDIR / "diagnostics/vf_W1_R3_sampling_diagnostics.json"))
    c1 = res["resumability"]["chain_1"]
    res["input_identity"] = dict(
        inits_byte_identical_to_morning_run=same_inits,
        executable_sha256_matches_vf_W1_R3=bool(V.sha(V.exe("R3")) == orig.get("executable_sha256")),
        N_train=int(fi["data"]["N"]), N_train_matches=bool(int(fi["data"]["N"]) == orig.get("N_train")),
        held_out_sites_match=bool(sorted(fi["held"]) == sorted(orig["held_out_sites"])),
        pilot_metric_length=int(len(m)), adapted_metric_length_chain1=c1["inv_metric_length"],
        initial_step=step, initial_step_matches_csv=bool(abs(step - float(open(PART / "track_a_m2rn_r3-20261116075950_1.csv").read().split("stepsize = ")[1].split()[0])) < 1e-6 * step),
        vc_settings=VCCFG["refit"])
    V.save_json(res, VR / "diagnostics/VR_w1_preflight.json")
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
