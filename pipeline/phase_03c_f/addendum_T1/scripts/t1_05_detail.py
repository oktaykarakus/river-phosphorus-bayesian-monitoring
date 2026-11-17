"""T1 step 4c: detailed diagnostics requested by the T1 brief, on the original (4 x 2,500), segment (4 x 2,500) and combined
(4 x 5,000) draws: behaviour near sigma_u = 0; network offset; tail-up variance and range; chain-wise and segment-wise summaries;
R-hat by chain length; an 8-chain display (each segment its own chain; not a gate); chain sensitivity of Delta_T (leave-one-chain-out).
Outputs: diagnostics/T1_detail.json, tables/T1_chain_segment_summary.csv, tables/T1_param_diagnostics_by_view.csv,
tables/T1_delta_T_comparison.csv, figures/fig_t1_sigma_u_trace_rank.png
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import t1_lib as T  # noqa: E402

L = T.redirect()
import arviz as az  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

NCH = 4; N0 = 2500
PARS = ["sigma_u", "share_u", "share_B", "share_tu", "sigma_B", "sigma_tu", "rho_tu", "sigma_site", "delta_bar", "sigma_om", "sigma_C", "rho"]


def get(D, v):
    if v.startswith("share_"):
        return np.asarray(D["share"])[:, {"share_u": 0, "share_B": 1, "share_tu": 2}[v]]
    return np.asarray(L.B.get1(D, v))


def diag(x, logit=False):
    x = np.asarray(x)
    return dict(r_hat=float(az.rhat(x)), ess_bulk=float(az.ess(x, method="bulk")), ess_tail=float(az.ess(x, method="tail")))


def runs_below(x, thr):
    r, out = 0, []
    for b in x < thr:
        if b:
            r += 1
        elif r:
            out.append(r); r = 0
    if r:
        out.append(r)
    return out


def main():
    sc = T.scratch()
    D = T.L.load_draws(T.COMB_TAG)
    n = len(D["lp__"]) // NCH
    dT = dict(np.load(sc / f"{T.COMB_TAG}_delta_T_draws.npz"))
    dT0 = dict(np.load(sc / f"{T.ORIG_TAG}_delta_T_draws.npz"))
    X = {v: get(D, v).reshape(NCH, n) for v in PARS}
    for c in ["0.05", "0.1", "0.2"]:
        X[f"Delta_T_{c}"] = np.asarray(dT[c]).reshape(NCH, n)
    # consistency: the first 2,500 draws of each combined chain are exactly the original draws
    same = bool(np.array_equal(X["Delta_T_0.1"][:, :N0], np.asarray(dT0["0.1"]).reshape(NCH, N0)))
    views = {"original (4x2500)": lambda a: a[:, :N0], "T1 segment (4x2500)": lambda a: a[:, N0:], "combined (4x5000)": lambda a: a,
             "8-chain display (segments as chains)": lambda a: np.concatenate([a[:, :N0], a[:, N0:]], axis=0)}
    rows = []
    for v, a in X.items():
        for lab, f in views.items():
            y = f(np.log(a) if v.startswith(("sigma", "rho", "share")) else a)
            rows.append(dict(parameter=v, scale="log" if v.startswith(("sigma", "rho", "share")) else "natural", view=lab, **diag(y)))
    PD = pd.DataFrame(rows); PD.to_csv(T.ADD / "tables" / "T1_param_diagnostics_by_view.csv", index=False)
    # chain x segment summaries
    cs = []
    for k in range(NCH):
        for seg, sl in [("original", slice(0, N0)), ("T1", slice(N0, n))]:
            r = dict(chain=k + 1, segment=seg)
            for v in PARS + ["Delta_T_0.05", "Delta_T_0.1", "Delta_T_0.2"]:
                x = X[v][k, sl]
                r[f"{v}_median"] = float(np.median(x))
                if v in ("sigma_u", "rho_tu", "sigma_tu", "delta_bar"):
                    r[f"{v}_q05"] = float(np.quantile(x, .05)); r[f"{v}_q95"] = float(np.quantile(x, .95))
            su = X["sigma_u"][k, sl]; rb = runs_below(su, 0.05)
            r.update(p_sigma_u_lt_0_05=float((su < 0.05).mean()), p_sigma_u_lt_0_02=float((su < 0.02).mean()),
                     excursions_below_0_05=len(rb), longest_excursion=int(max(rb) if rb else 0),
                     ess_bulk_log_sigma_u_single_chain=float(az.ess(np.log(su)[None], method="bulk")))
            cs.append(r)
    CS = pd.DataFrame(cs); CS.to_csv(T.ADD / "tables" / "T1_chain_segment_summary.csv", index=False)
    # sigma_u near zero: local R-hat, sampler behaviour at small sigma_u, R-hat by length
    A = np.log(X["sigma_u"])
    qs = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
    loc = {}
    for lab, f in list(views.items())[:3]:
        a = f(A); qv = np.quantile(a, qs)
        loc[lab] = {str(q): float(az.rhat((a <= v).astype(float))) for q, v in zip(qs, qv)}
    su = X["sigma_u"].ravel(); low = su < 0.05
    acc = np.asarray(D["accept_stat__"]); lf = np.asarray(D["n_leapfrog__"]); dv = np.asarray(D["divergent__"]); td = np.asarray(D["treedepth__"])
    near0 = dict(p_lt_0_05_pooled=float(low.mean()), p_lt_0_02_pooled=float((su < 0.02).mean()), q01=float(np.quantile(su, .01)), q05=float(np.quantile(su, .05)),
                 min=float(su.min()),
                 sampler_when_sigma_u_lt_0_05=dict(n=int(low.sum()), mean_accept=float(acc[low].mean()), mean_leapfrog=float(lf[low].mean()),
                                                   divergences=int(dv[low].sum()), treedepth10=int((td[low] >= 10).sum())),
                 sampler_when_sigma_u_ge_0_05=dict(n=int((~low).sum()), mean_accept=float(acc[~low].mean()), mean_leapfrog=float(lf[~low].mean()),
                                                   divergences=int(dv[~low].sum()), treedepth10=int((td[~low] >= 10).sum())),
                 local_rhat=loc)
    traj = [dict(draws_per_chain=m, rhat=float(az.rhat(A[:, :m])), ess_bulk=float(az.ess(A[:, :m], method="bulk")), ess_tail=float(az.ess(A[:, :m], method="tail")))
            for m in range(500, n + 1, 500)]
    loo_rhat = {f"without_chain_{k + 1}": float(az.rhat(np.delete(A, k, axis=0))) for k in range(NCH)}
    # Delta_T: combined vs original, chain sensitivity
    dtab, loo = [], []
    for c in ["0.05", "0.1", "0.2"]:
        x = X[f"Delta_T_{c}"]; x0 = x[:, :N0]; xs = x[:, N0:]
        def band(z):
            z = np.asarray(z).ravel(); return dict(median=float(np.median(z)), lo95=float(np.quantile(z, .025)), hi95=float(np.quantile(z, .975)),
                                                   p_gt0=float((z > 0).mean()), sd=float(z.std()))
        dtab.append(dict(threshold=float(c), **{f"combined_{k}": v for k, v in band(x).items()}, **{f"original_{k}": v for k, v in band(x0).items()},
                         **{f"segment_{k}": v for k, v in band(xs).items()}))
        for k in range(NCH):
            b = band(np.delete(x, k, axis=0)); loo.append(dict(threshold=float(c), without_chain=k + 1, **b))
    DT = pd.DataFrame(dtab); DT.to_csv(T.ADD / "tables" / "T1_delta_T_comparison.csv", index=False)
    LO = pd.DataFrame(loo); LO.to_csv(T.ADD / "tables" / "T1_delta_T_leave_one_chain_out.csv", index=False)
    out = dict(first_2500_identical_to_original=same, draws_per_chain=n, near_sigma_u_zero=near0, rhat_log_sigma_u_by_length=traj,
               rhat_log_sigma_u_leave_one_chain_out=loo_rhat,
               key_diagnostics_by_view=PD[PD.parameter.isin(["sigma_u", "share_u", "delta_bar", "sigma_tu", "rho_tu", "share_tu", "sigma_B", "Delta_T_0.1"])].round(5).to_dict("records"),
               delta_T=DT.round(5).to_dict("records"),
               delta_T_leave_one_chain_out_range={c: dict(median=[float(LO[LO.threshold == float(c)]["median"].min()), float(LO[LO.threshold == float(c)]["median"].max())],
                                                          p_gt0_min=float(LO[LO.threshold == float(c)].p_gt0.min()), lo95_min=float(LO[LO.threshold == float(c)].lo95.min()))
                                                  for c in ["0.05", "0.1", "0.2"]})
    T.save_json(out, T.ADD / "diagnostics" / "T1_detail.json")
    print(json.dumps({k: out[k] for k in ["first_2500_identical_to_original", "near_sigma_u_zero", "rhat_log_sigma_u_by_length", "rhat_log_sigma_u_leave_one_chain_out",
                                          "delta_T_leave_one_chain_out_range"]}, indent=1))
    print(PD[PD.parameter.isin(["sigma_u", "share_u", "delta_bar", "sigma_tu", "rho_tu", "sigma_B", "Delta_T_0.1"])].round(4).to_string())
    print(CS[["chain", "segment", "sigma_u_median", "sigma_u_q05", "p_sigma_u_lt_0_05", "excursions_below_0_05", "longest_excursion", "delta_bar_median",
              "sigma_tu_median", "rho_tu_median", "Delta_T_0.1_median"]].round(4).to_string())
    print(DT.round(4).to_string())
    # figure
    cols = ["#2a78d6", "#d06a2b", "#2f9e6e", "#7a4fb5"]
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 3.4), gridspec_kw=dict(width_ratios=[2.4, 1]))
    for k in range(NCH):
        ax[0].plot(X["sigma_u"][k], color=cols[k], lw=.35, alpha=.8, label=f"chain {k + 1}")
    ax[0].axvline(N0, color="#55534e", ls="--", lw=1); ax[0].text(N0 + 30, ax[0].get_ylim()[1] * .95, "T1 segment →", fontsize=8, va="top")
    ax[0].set_ylabel("σ_u"); ax[0].set_xlabel("post-warmup draw"); ax[0].legend(frameon=False, fontsize=7, ncol=4, loc="lower right")
    rk = pd.Series(A.ravel()).rank().to_numpy().reshape(NCH, n); bins = np.linspace(0, NCH * n, 21)
    for k in range(NCH):
        h, _ = np.histogram(rk[k], bins=bins); ax[1].step(bins[:-1], h, where="post", color=cols[k], lw=1)
    ax[1].axhline(n / 20, color="#9a9893", ls=":", lw=1); ax[1].set_xlabel("pooled rank of log σ_u (combined)"); ax[1].set_ylabel("draws per bin")
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(T.ADD / "figures" / "fig_t1_sigma_u_trace_rank.png", dpi=200); plt.close(fig)


if __name__ == "__main__":
    main()
