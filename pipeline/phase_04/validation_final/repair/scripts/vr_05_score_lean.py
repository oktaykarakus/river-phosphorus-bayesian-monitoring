"""VR_05: lean site-joint scoring of one (fold, model) over all posterior draws, or n thinned draws by the frozen rule
linspace(0, S0-1, n).astype(int). Frozen Phase 3C functions, unchanged (vr_02.site_lp; verified exact against stored values).
Processes draws in chunks; river-network structures computed once.
Usage: vr_05_score_lean.py FOLD MODEL SOURCE N   (MODEL m2rn|m1s2; SOURCE vf|vr for m2rn; N = 0 for all draws)
Output: scratch phase_04_vr/score_lp_{N or all}_{fold}_{model}_{source}.npz (lp, site, pick, S0); ledger entry."""
import sys
from pathlib import Path
import time

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import vr_final_lib as K
from vr_02_bench_rescore import site_lp

V = K.V
F, C, R, B = V.F, V.C, V.R, V.B


def main(fold, model, source, n):
    meas, sites, sy = V.L.load_tables()
    held = V.heldout_sites(fold, sites)
    if model == "m1s2":
        D = V.comparator_draws(fold); mc = C.model_config(tu=0)
    else:
        D = K.r3_draws(fold, source); mc = R.model_config("R3")
    S0 = B.ndraws(D)
    pick = np.arange(S0) if n == 0 else np.linspace(0, S0 - 1, n).astype(int)
    per_chain = np.bincount(pick // (S0 // 4), minlength=4)
    assert (per_chain == per_chain[0]).all(), f"thinned draws not balanced over chains: {per_chain}"
    rn = C.rn_structures(sites) if mc.get("tu", 0) else None
    t0 = time.time(); out = []; names = None
    step = K.CFG["scoring"]["chunk_draws"]
    for a in range(0, len(pick), step):
        lp, nm = site_lp(D, meas, sites, sy, mc, held, pick[a:a + step], rn=rn)
        names = nm if names is None else names
        assert list(nm) == list(names)
        out.append(lp)
        print(f"{fold} {model} {a + len(pick[a:a + step])}/{len(pick)} draws, {time.time() - t0:.0f} s", flush=True)
    lp = np.hstack(out)
    lab = "all" if n == 0 else str(n)
    np.savez_compressed(K.SCR / f"score_lp_{lab}_{fold}_{model}_{source}.npz", lp=lp, site=names.astype(str), pick=pick, S0=S0)
    K.ledger(f"lean scoring {fold} {model} {source} ({len(pick)} draws)", n_draws=int(len(pick)))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]))
