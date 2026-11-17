"""VR_08 (stage S3): frozen 400-draw scoring of the W1 refit (M2-RN only; the W1 M1-S2 scores are stored) with the frozen
p3c_folds.heldout_scores (pointwise PIT / coverage for the C4 calibration test, and the 400-draw record), the PRE-DECLARED
400-draw Rao-Blackwell statistic and the AMENDED AM1 statistic on all draws (threshold 0.05 unchanged).
Exit code 0 = AM1 passes, 3 = fails. Writes repair/tables/folds_vr/W1_{site,pointwise,siteyear}.csv and
repair/diagnostics/VR_gate_S3.json."""
import json
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import vr_final_lib as K

V = K.V
F, C, R, B = V.F, V.C, V.R, V.B


def rb(D, sites, held, n=None):
    import arviz as az
    mc = R.model_config("R3")
    S0 = B.ndraws(D); n = S0 if n is None else min(n, S0)
    pk = np.linspace(0, S0 - 1, n).astype(int)
    Ds = {k: (np.asarray(v)[pk] if np.ndim(v) >= 1 and len(v) == S0 else v) for k, v in D.items()}
    hidx = np.flatnonzero(sites.site.isin(held).to_numpy())
    mu, sd = F.heldout_theta_conditional(Ds, sites, mc, hidx, C.rn_structures(sites))
    d = mu - np.asarray(Ds["theta"], float)[:, hidx]; S = d.shape[0]
    ess = np.array([float(az.ess(d[:, i].reshape(4, S // 4))) for i in range(d.shape[1])])
    z = d.mean(0) / (d.std(0, ddof=1) / np.sqrt(np.clip(ess, 10, S)))
    mx = float(np.abs(d.mean(0)).max())
    return dict(n_draws=int(S), max_abs_mean_diff=mx, passes_threshold=bool(mx <= K.CFG["adequacy_fold"]["rb_threshold"]),
                mean_z=float(z.mean()), sd_z=float(z.std(ddof=1)), max_abs_z=float(np.abs(z).max()))


def main():
    meas, sites, sy = V.L.load_tables()
    held = V.heldout_sites("W1", sites)
    D = K.r3_draws("W1", "vr")
    s, p, y = F.heldout_scores(D, meas, sites, sy, R.model_config("R3"), held, n_draws=V.CFG["scoring"]["n_draws"])
    out = K.FOLDS_VR
    for t, nm in [(s, "site"), (p, "pointwise"), (y, "siteyear")]:
        t.insert(0, "model", "m2rn"); t.insert(0, "fold", "W1"); t.to_csv(out / f"W1_{nm}.csv", index=False)
    res = dict(stage="S3", record_400=dict(n_sites=int(len(s)), elpd_m2rn_400=float(s.elpd_site.sum())),
               rb_predeclared_400=rb(D, sites, held, 400), rb_AM1_all_draws_AMENDED=rb(D, sites, held))
    res["pass"] = res["rb_AM1_all_draws_AMENDED"]["passes_threshold"]
    K.save_json(res, K.DIAG / "VR_gate_S3.json")
    K.ledger("S3 W1 refit: frozen 400-draw scoring + RB (400 pre-declared, all-draw AM1)")
    print(json.dumps(res, indent=1, default=float))
    sys.exit(0 if res["pass"] else 3)


if __name__ == "__main__":
    main()
