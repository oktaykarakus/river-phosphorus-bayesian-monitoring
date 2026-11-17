"""Phase 3C-F: identifiability of the site-level variance allocation (sigma_u, sigma_B, sigma_TU) and of the tail-up range rho_TU.

Separates four explanations:
 * structural non-identifiability  -> expected Fisher information of (log su^2, log sB^2, log stu^2, log rho) is singular even when
   the site levels theta are known exactly ("complete data"), for the observed river / water-body design;
 * weak identification from data   -> information is non-singular with theta known but collapses once theta is observed with the
   measurement-level noise implied by each site's replication (Sigma + N, N_ii = approx. var(theta_i | own data));
 * parameterisation problems       -> compared across R0 / R1 / R3 posteriors (Phase 3C, 3C-R) and the main R3 fit;
 * sampler failure                 -> main-fit diagnostics (R-hat, ESS, E-BFMI, divergences) for the same quantities.
Fisher information for theta - m ~ MVN(0, Sigma(psi)): I_jk = 0.5 tr(Sigma^-1 dSigma_j Sigma^-1 dSigma_k), block-diagonal over the
233 R3 components (exact). Evaluated at posterior medians and at rho_TU in {50, 100, 200, 500, 1000, 3000} km.
Also: prior-posterior contraction (1 - var_post / var_prior) of shares, sigma_u, sigma_B, sigma_tu, log rho_TU from the main fit.
Usage: p3f_06_identifiability.py --tag main_R3
Outputs: diagnostics/identifiability_<tag>.json, tables/fisher_information_<tag>.csv
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import p3f_lib as L  # noqa: E402

B = L.B


def blocks(sites):
    r3 = L.R.r3_structures(sites); out = []
    for k in range(r3["n_blk"]):
        n = r3["blk_size"][k]; idx = np.array(r3["blk_sites"][r3["blk_start"][k] - 1:r3["blk_start"][k] - 1 + n]) - 1
        W = np.array(r3["wbs"][r3["wbs_start"][k] - 1:r3["wbs_start"][k] - 1 + n * n]).reshape(n, n).T
        pr = [(r3["pr_i"][p] - 1, r3["pr_j"][p] - 1, r3["pr_h"][p], r3["pr_lw"][p]) for p in range(r3["pr_start"][k] - 1, r3["pr_start"][k] - 1 + r3["pr_n"][k])]
        out.append((idx, W, pr))
    return out


def fisher(blks, su, sb, stu, rho, noise=None):
    """Expected information for psi = (log su^2, log sB^2, log stu^2, log rho)."""
    I = np.zeros((4, 4))
    for idx, W, pr in blks:
        n = len(idx)
        Ctu = np.eye(n); dC = np.zeros((n, n))
        for i, j, h, lw in pr:
            c = np.exp(-h / rho + 0.5 * lw); Ctu[i, j] = Ctu[j, i] = c
            dC[i, j] = dC[j, i] = c * h / rho                       # d/dlog rho of exp(-h/rho) = exp(-h/rho) * h / rho
        S = su ** 2 * np.eye(n) + sb ** 2 * W + stu ** 2 * Ctu
        if noise is not None:
            S = S + np.diag(noise[idx])
        Si = np.linalg.inv(S)
        dS = [su ** 2 * np.eye(n), sb ** 2 * W, stu ** 2 * Ctu, stu ** 2 * dC]
        M = [Si @ d for d in dS]
        for a in range(4):
            for b in range(a, 4):
                I[a, b] += 0.5 * np.sum(M[a] * M[b].T); I[b, a] = I[a, b]
    return I


def theta_noise(D, meas, sites, sy, mc):
    """Approximate var(theta_i | site i's own data): measurement SD sigma_i (site-specific, posterior median) and site-year shock
    sigma_om; site-year y with n_iy measurements contributes precision n_iy / (sigma_i^2 + n_iy sigma_om^2). Censoring ignored
    (so this is a lower bound on the noise)."""
    sg = np.median(B.sig_site(B_thin(D, 200), sites, mc), axis=0)
    som = float(np.median(B.get1(D, "sigma_om")))
    cnt = meas.groupby(["site_idx", "siteyear_idx"]).size().reset_index(name="n")
    cnt["prec"] = cnt.n / (sg[cnt.site_idx - 1] ** 2 + cnt.n * som ** 2)
    prec = cnt.groupby("site_idx").prec.sum().reindex(range(1, len(sites) + 1)).fillna(0).to_numpy()
    return 1 / np.maximum(prec, 1e-9)


def B_thin(D, n):
    S = B.ndraws(D); sel = np.linspace(0, S - 1, n).astype(int)
    return {k: (np.asarray(v)[sel] if (np.ndim(v) >= 1 and len(v) == S) else v) for k, v in D.items()}


def info_summary(I):
    ev = np.linalg.eigvalsh(I)
    cov = np.linalg.inv(I); sd = np.sqrt(np.diag(cov)); cor = cov / np.outer(sd, sd)
    return dict(eigenvalues=ev.tolist(), condition=float(ev.max() / ev.min()), sd_log_param=dict(zip(["log_su2", "log_sB2", "log_stu2", "log_rho"], sd.tolist())),
                correlation=np.round(cor, 3).tolist())


def contraction(D):
    """Prior-posterior contraction under the model priors (half-normal SDs from the Phase 3C config; rho_tu InvGamma)."""
    rng = np.random.default_rng(20261115)
    pri = L.C.CFG["priors"] if "priors" in L.C.CFG else {}
    out = {}
    for k in ["sigma_u", "sigma_B", "sigma_tu", "rho_tu", "sigma_site"]:
        if k in D:
            x = B.get1(D, k)
            out[k] = dict(post_median=float(np.median(x)), post_90=[float(np.quantile(x, .05)), float(np.quantile(x, .95))])
    if "share" in D:
        sh = np.asarray(D["share"])
        for i, n in enumerate(["share_u", "share_B", "share_tu"]):
            out[n] = dict(post_median=float(np.median(sh[:, i])), post_90=[float(np.quantile(sh[:, i], .05)), float(np.quantile(sh[:, i], .95))])
    return out, pri


def main(tag, where):
    meas, sites, sy = L.load_tables()
    mc = L.R.model_config("R3")
    D = L.load_draws(tag, where=where)
    med = {k: float(np.median(B.get1(D, k))) for k in ["sigma_u", "sigma_B", "sigma_tu", "rho_tu"]}
    blks = blocks(sites)
    noise = theta_noise(D, meas, sites, sy, mc)
    rows, res = [], {}
    for rho in [med["rho_tu"], 50, 100, 200, 500, 1000, 3000]:
        for lab, nz in [("complete (theta known)", None), ("observed (theta + noise)", noise)]:
            I = fisher(blks, med["sigma_u"], med["sigma_B"], med["sigma_tu"], rho, nz)
            s = info_summary(I); key = f"rho={rho:.0f}|{lab}"; res[key] = s
            rows.append(dict(rho_tu=rho, setting=lab, **{f"sd_{k}": v for k, v in s["sd_log_param"].items()}, condition=s["condition"],
                             min_eig=s["eigenvalues"][0], corr_su_stu=s["correlation"][0][2], corr_su_sB=s["correlation"][0][1], corr_sB_stu=s["correlation"][1][2],
                             corr_stu_rho=s["correlation"][2][3], info_log_rho=float(I[3, 3])))
    T = pd.DataFrame(rows); T.to_csv(L.ROOT / "tables" / f"fisher_information_{tag}.csv", index=False)
    con, pri = contraction(D)
    wb = B.hier_indices(sites)["wb"]; nwb = np.bincount(wb)
    design = dict(n_sites=len(sites), water_bodies=int((nwb > 0).sum()), wb_with_2plus_sites=int((nwb >= 2).sum()),
                  sites_in_multi_site_wb=int(nwb[nwb >= 2].sum()), n_blocks=len(blks), singleton_blocks=int(sum(len(b[0]) == 1 for b in blks)),
                  tu_pairs=int(sum(len(b[2]) for b in blks)), median_theta_noise_sd=float(np.median(np.sqrt(noise))),
                  theta_noise_sd_quartiles=np.quantile(np.sqrt(noise), [.25, .5, .75]).round(3).tolist())
    out = dict(tag=tag, posterior_medians=med, design=design, fisher=res, posterior=con)
    L.save_json(out, L.ROOT / "diagnostics" / f"identifiability_{tag}.json")
    pd.set_option("display.width", 250)
    print(json.dumps(design, indent=1)); print(T.round(3).to_string()); print(json.dumps(con, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="main_R3"); ap.add_argument("--where", default="3cf")
    a = ap.parse_args(); main(a.tag, a.where)
