"""Post-hoc sampler-sensitivity check for R3 fold fits that exceed the frozen divergence limit (VF_01 §3).

Declared 2026-11-15 ~18:30, after the W1 fit showed 1.75% divergences (frozen limit 0.1%) and BEFORE any W1 score existed.
The divergences concentrate in the lower tail of sigma_u (divergent draws: median sigma_u 0.012; posterior median 0.10).
Question: do held-out scores depend on where the sampler is in sigma_u? If not, under-exploration of that tail cannot move
the scores materially. This does NOT change the frozen adequacy verdict; the fit still fails the divergence rule.

For each fold f with an R3 fit:
  - the 400 scoring draws (same even thinning as the frozen scorer) are split at the median of sigma_u (and at its lower quartile);
  - the held-out ELPD of every site is recomputed within each subset (log-mean-exp of the captured per-draw log densities);
  - shift = |ELPD_diff(lower subset) - ELPD_diff(upper subset)| summed over sites, where ELPD_diff = M2-RN - M1-S2
    (M1-S2 is fixed, so the shift is the M2-RN ELPD shift);
  - reference: the fold's catchment-clustered SE of the ELPD difference (tables/VF_validation_folds.csv).
  Declared threshold: shift < 0.20 x clustered SE -> "scores insensitive to the sigma_u tail".
  Also reported: Spearman correlation of the per-draw total held-out log density with log sigma_u, and the same split
  by "within 20 iterations of a divergence" vs the rest (a direct, if noisy, check of the neighbourhood of the problem).
Output: diagnostics/VF_sampler_sensitivity.json, tables/VF_sampler_sensitivity.csv
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import spearmanr

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import vf_lib as V  # noqa: E402

THRESH = 0.20


def elpd(lp):
    return logsumexp(lp, axis=1) - np.log(lp.shape[1])


def main(folds):
    T = pd.read_csv(V.VF / "tables/VF_validation_folds.csv").set_index("group")
    rows = []
    for f in folds:
        p = V.scratch() / f"score_lp_{f}_m2rn.npz"
        if not p.exists():
            continue
        z = np.load(p); lp = z["lp"]                                    # sites x S
        D = np.load(V.scratch() / f"vf_{f}_R3_draws.npz")
        S0 = len(D["sigma_u"]); pick = np.linspace(0, S0 - 1, lp.shape[1]).astype(int)
        su = D["sigma_u"][pick]
        div = D["divergent__"].astype(bool)
        near = np.zeros(S0, bool)
        it = np.arange(S0) % (S0 // 4)
        for k in np.flatnonzero(div):
            sel = (np.arange(S0) // (S0 // 4) == k // (S0 // 4)) & (np.abs(it - it[k]) <= 20)
            near |= sel
        near = near[pick]
        se = float(T.loc[f, "se_clustered"])
        full = elpd(lp).sum()
        res = dict(fold=f, n_sites=int(lp.shape[0]), n_draws=int(lp.shape[1]), divergences=int(div.sum()), se_clustered=se,
                   spearman_total_lp_vs_log_sigma_u=float(spearmanr(lp.sum(0), np.log(su)).correlation))
        for lab, lo in [("median", su <= np.median(su)), ("q25", su <= np.quantile(su, .25)), ("near_divergence", near)]:
            if lo.sum() < 20 or (~lo).sum() < 20:
                res[f"shift_{lab}"] = None; continue
            a, b = elpd(lp[:, lo]).sum(), elpd(lp[:, ~lo]).sum()
            res[f"shift_{lab}"] = float(abs(a - b)); res[f"shift_{lab}_over_se"] = float(abs(a - b) / se)
            res[f"elpd_{lab}_lower"] = float(a - full); res[f"elpd_{lab}_rest"] = float(b - full); res[f"n_{lab}"] = int(lo.sum())
        res["insensitive"] = bool((res.get("shift_median_over_se") or 0) < THRESH)
        rows.append(res)
        print(json.dumps(res, indent=1))
    pd.DataFrame(rows).to_csv(V.VF / "tables/VF_sampler_sensitivity.csv", index=False)
    json.dump(dict(threshold_over_se=THRESH, folds=rows), open(V.VF / "diagnostics/VF_sampler_sensitivity.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1:] or ["W1", "W2", "C1", "C2"])
