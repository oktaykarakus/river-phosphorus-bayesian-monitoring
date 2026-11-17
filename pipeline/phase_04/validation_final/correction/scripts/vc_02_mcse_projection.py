"""Correction stage, VC_00 §5: projected scoring MCSE / clustered SE with 1,600 draws for W1 and W2, from existing outputs only
(no scoring). Per held-out site and model: CV of the importance weights and their ESS from the stored Phase 4B per-draw site log
densities (400 draws; frozen MCSE formula reproduced and checked), times the ESS gain r_i = ESS(theta_i at the 1,600 thinned
indices) / ESS(theta_i at the 400 thinned indices) from the full posterior draws (central), or min(r_i, r_log_sigma_u)
(conservative). Usage: vc_02_mcse_projection.py [--w1 original|refit]"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import vc_lib as K  # noqa: E402

V = K.V
L = V.L


def pick(S0, n):
    return np.linspace(0, S0 - 1, min(n, S0)).astype(int)          # the frozen scorer's selection


def ess(x):
    import arviz as az
    S = len(x)
    return float(az.ess(np.asarray(x, float).reshape(4, S // 4)))


def site_terms(lp):
    w = np.exp(lp - lp.max()); S = len(w)
    try:
        e = ess(w)
    except Exception:
        e = float(S)
    e = max(min(e, S), 10.0)
    return float(w.std(ddof=1) / w.mean()), e


def project(fold, model, D, sites, n_new):
    lpz = np.load(K.ORIG_SCRATCH / f"score_lp_{fold}_{model}.npz")
    lp, site = lpz["lp"], lpz["site"].astype(str)
    stored = pd.read_csv(K.VFDIR / "tables/folds" / f"{fold}_site.csv")
    stored = stored[stored.model == model].set_index("site").mcse_elpd
    idx = pd.Series(np.arange(len(sites)), index=sites.site.astype(str))
    th = np.asarray(D["theta"], float); S0 = th.shape[0]
    p400, pnew = pick(S0, 400), pick(S0, n_new)
    ls = np.log(np.asarray(D["sigma_u"], float).ravel())
    r_su = ess(ls[pnew]) / ess(ls[p400])
    rows = []
    for k, s in enumerate(site):
        cv, e400 = site_terms(lp[k])
        q = th[:, idx[s]]
        r = ess(q[pnew]) / ess(q[p400])
        rows.append(dict(fold=fold, model=model, site=s, cv=cv, ess400=e400, mcse400=cv / np.sqrt(e400), mcse400_stored=stored.get(s, np.nan),
                         r_theta=r, r_log_sigma_u=r_su, n_new=len(pnew),
                         ess_new_central=min(e400 * r, len(pnew)), ess_new_cons=min(e400 * min(r, r_su), len(pnew)),
                         ess_new_ideal=min(e400 * len(pnew) / 400, len(pnew))))
    T = pd.DataFrame(rows)
    for k in ["central", "cons", "ideal"]:
        T[f"mcse_new_{k}"] = T.cv / np.sqrt(T[f"ess_new_{k}"])
    return T


def main(w1):
    meas, sites, sy = L.load_tables()
    VV = json.load(open(K.VFDIR / "diagnostics/VF_validation.json"))
    G = {g["group"]: g for g in VV["groups"]}
    out, tabs = {}, []
    for fold in ["W1", "W2"]:
        if fold == "W1" and w1 == "refit":
            Dr = K.new_draws("vc_W1_R3"); src = "refit vc_W1_R3 (CV and ESS400 from the original W1 per-draw scores: approximation, VC_00 §5)"
        else:
            Dr = K.original_draws(f"vf_{fold}_R3"); src = f"original vf_{fold}_R3"
        Dc = V.comparator_draws(fold)
        T = pd.concat([project(fold, "m2rn", Dr, sites, 1600), project(fold, "m1s2", Dc, sites, 1600)])
        tabs.append(T)
        se = G[fold]["se_clustered"]
        res = dict(r3_source=src, clustered_se_phase4b=se, mcse_over_se_phase4b=G[fold]["mcse_over_se"])
        chk = T.dropna(subset=["mcse400_stored"])
        res["max_abs_dev_reproduced_vs_stored_mcse400"] = float(np.abs(chk.mcse400 - chk.mcse400_stored).max())
        for k in ["400", "new_central", "new_cons", "new_ideal"]:
            col = "mcse400" if k == "400" else f"mcse_{k}"
            res[f"ratio_{k}"] = float(np.sqrt((T[col] ** 2).sum()) / se)
        for m in ["m2rn", "m1s2"]:
            t = T[T.model == m]
            res[f"{m}_share_of_mcse2_400"] = float((t.mcse400 ** 2).sum() / (T.mcse400 ** 2).sum())
            res[f"{m}_median_r_theta"] = float(t.r_theta.median()); res[f"{m}_r_log_sigma_u"] = float(t.r_log_sigma_u.iloc[0])
            res[f"{m}_median_ess400"] = float(t.ess400.median()); res[f"{m}_n_sites_ess400_lt_100"] = int((t.ess400 < 100).sum())
            res[f"{m}_top5_site_share_mcse2"] = float(np.sort(t.mcse400 ** 2)[-5:].sum() / (T.mcse400 ** 2).sum())
        rule = K.VCFG["projection_rule"]
        res["rescore"] = bool(res["ratio_new_central"] <= rule["rescore_if_central_le"] and res["ratio_new_cons"] < rule["and_conservative_lt"])
        out[fold] = res
    tag = "refit" if w1 == "refit" else "original"
    pd.concat(tabs).to_csv(K.VC / "tables" / f"VC_mcse_projection_sites_W1{tag}.csv", index=False)
    K.save_json(out, K.VC / "diagnostics" / f"VC_mcse_projection_W1{tag}.json")
    K.ledger(f"MCSE projection (W1 {tag}, W2)")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--w1", default="original", choices=["original", "refit"]); a = ap.parse_args()
    main(a.w1)
