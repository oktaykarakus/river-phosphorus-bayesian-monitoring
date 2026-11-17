"""T1 step 4a: combine the original main_R3 draws and the T1 segment per chain (all draws, in order; no thinning, no discarding)
and compute the sampling diagnostics of the combined 4 x 5,000 chains in the Phase 3C-F schema, so that the unchanged Phase 3C-F
gate code (p3f_04) can be applied to them.
Outputs (addendum only): scratch T1_combined_draws.npz; diagnostics/T1_combined_sampling_diagnostics.json;
tables/T1_combined_posterior_summary.csv
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import t1_lib as T  # noqa: E402

L = T.redirect()
import arviz as az  # noqa: E402

NCH = 4
LAT = ["theta", "z_om", "z_sig", "vB", "z_C", "z_A", "w_site", "z_e", "v_tu"]


def main():
    sc = T.scratch()
    A = np.load(sc / f"{T.ORIG_TAG}_draws.npz"); Bz = np.load(sc / f"{T.SEG_TAG}_draws.npz")
    keys = [k for k in A.files if k in Bz.files]
    missing = sorted(set(A.files) ^ set(Bz.files))
    comb = {}
    for k in keys:
        a = A[k]; b = Bz[k]
        na, nb = a.shape[0] // NCH, b.shape[0] // NCH
        comb[k] = np.concatenate([a.reshape(NCH, na, *a.shape[1:]), b.reshape(NCH, nb, *b.shape[1:])], axis=1).reshape(NCH * (na + nb), *a.shape[1:])
    n = comb["lp__"].shape[0] // NCH
    np.savez(sc / f"{T.COMB_TAG}_draws.npz", **comb)
    # posterior summary of the top-level parameters (same variable list and az.summary call as p3f_lib.summarise_fit)
    top = [v for v in L.TOP if v in comb and comb[v].size > 0]
    post = {v: comb[v].reshape(NCH, n, *comb[v].shape[1:]) for v in top}
    idata = az.from_dict(posterior=post)
    summ = az.summary(idata, var_names=top, round_to=6)
    summ.to_csv(T.ADD / "tables" / f"{T.COMB_TAG}_posterior_summary.csv")
    hyper = summ.loc[[i for i in summ.index if i.split("[")[0] in L.HYPER]]
    extra = {}
    for v in ["rho_tu", "rho", "sigma_u", "sigma_B", "sigma_tu", "sigma_site"]:
        if v in comb and comb[v].size > 0:
            lv = np.log(comb[v].reshape(NCH, n, -1)[..., 0])
            extra[f"log_{v}"] = dict(r_hat=float(az.rhat(lv)), ess_bulk=float(az.ess(lv, method="bulk")), ess_tail=float(az.ess(lv, method="tail")))
    lat_r, lat_b, lat_t = [], [], []
    for v in LAT:
        if v in comb and comb[v].size > 0:
            x = comb[v].reshape(NCH, n, -1)
            ds = az.from_dict(posterior={v: x})
            s = az.summary(ds, var_names=[v], kind="diagnostics")
            i = s["r_hat"].idxmax()
            lat_r.append((float(s["r_hat"].max()), str(i))); lat_b.append(float(s["ess_bulk"].min())); lat_t.append(float(s["ess_tail"].min()))
            del x, ds, s
    energy = comb["energy__"].reshape(NCH, n)
    ebfmi = [float(x) for x in az.bfmi(energy)]
    div = comb["divergent__"].reshape(NCH, n)
    tdep = comb["treedepth__"].reshape(NCH, n); lf = comb["n_leapfrog__"].reshape(NCH, n)
    so = json.load(open(T.P3F / "diagnostics" / "main_R3_sampling_diagnostics.json"))
    ss = json.load(open(T.ADD / "diagnostics" / f"{T.SEG_TAG}_sampling_diagnostics.json"))
    res = dict(tag=T.COMB_TAG, model_config=so["model_config"], composition=f"per chain: main_R3 draws 1-2500, then T1 segment draws 1-{n - 2500}",
               keys_not_in_both=missing, wall_s=so["wall_s"] + ss["wall_s"], cpu_s=so["cpu_s"] + ss["cpu_s"], cpu_h=(so["cpu_s"] + ss["cpu_s"]) / 3600,
               chains=NCH, iter_warmup=so["iter_warmup"], iter_sampling=n, adapt_delta=so["adapt_delta"], max_treedepth=so["max_treedepth"],
               seed=[so["seed"], ss["seed"]], I=so["I"], J=so["J"], N=so["N"],
               divergences=int(div.sum()), divergences_by_chain=div.sum(axis=1).astype(int).tolist(), divergence_share=float(div.mean()),
               treedepth_hits=int((tdep >= so["max_treedepth"]).sum()), mean_treedepth=float(tdep.mean()), mean_leapfrog=float(lf.mean()),
               leapfrog_by_chain=lf.mean(axis=1).round(1).tolist(), stepsize_by_chain=[float(x) for x in comb["stepsize__"].reshape(NCH, n)[:, -1]],
               ebfmi=ebfmi, hyper_max_rhat=float(hyper["r_hat"].max()), hyper_argmax_rhat=str(hyper["r_hat"].idxmax()),
               hyper_min_ess_bulk=float(hyper["ess_bulk"].min()), hyper_argmin_ess_bulk=str(hyper["ess_bulk"].idxmin()),
               hyper_min_ess_tail=float(hyper["ess_tail"].min()), hyper_argmin_ess_tail=str(hyper["ess_tail"].idxmin()),
               log_scale_diagnostics=extra, latent_max_rhat=max(lat_r)[0], latent_argmax_rhat=max(lat_r)[1],
               latent_min_ess_bulk=min(lat_b), latent_min_ess_tail=min(lat_t),
               ebfmi_note="E-BFMI = arviz bfmi per chain on the concatenated energy series; the single segment junction adds one term of 4,999")
    # E-BFMI check against arviz on the original part (formula identity)
    eo = comb["energy__"].reshape(NCH, n)[:, :2500]
    res["ebfmi_formula_check_vs_original"] = dict(this=[float(x) for x in az.bfmi(eo)], original=so["ebfmi"])
    res["ebfmi_segment_only"] = [float(x) for x in az.bfmi(comb["energy__"].reshape(NCH, n)[:, 2500:])]
    res["summary_index_matches_original"] = sorted(set(summ.index)) == sorted(set(l.split(",")[0] for l in open(T.P3F / "tables" / "main_R3_posterior_summary.csv").read().splitlines()[1:]))
    T.save_json(res, T.ADD / "diagnostics" / f"{T.COMB_TAG}_sampling_diagnostics.json")
    print(json.dumps({k: res[k] for k in ["divergences", "ebfmi", "hyper_max_rhat", "hyper_argmax_rhat", "hyper_min_ess_bulk", "hyper_argmin_ess_bulk",
                                          "hyper_min_ess_tail", "latent_max_rhat", "latent_argmax_rhat", "keys_not_in_both", "ebfmi_formula_check_vs_original"]}, indent=1))
    print(summ.loc[[i for i in summ.index if i.split("[")[0] in ["sigma_u", "share", "sigma_B", "sigma_tu", "rho_tu", "delta_bar", "sigma_om"]]].to_string())


if __name__ == "__main__":
    main()
