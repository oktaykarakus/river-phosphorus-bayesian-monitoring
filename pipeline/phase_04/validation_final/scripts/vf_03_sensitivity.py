"""Phase 4B A4 / A5: RSN-only river-length robustness and the M2-I network-interaction sensitivity (VF_00 §5.2-5.4, pre-declared).

--what rsn : B-bar(c) on G_S with the frozen A3 code (phase_04/scripts/p4_03b_river_length.py, run unchanged with its reader and
             writers redirected to validation_final/rsn_only) on 500 thinned draws of vf_RSN_R3; P^R_T(c) with the frozen A2 code on
             2,000 draws; shared-parameter shifts vs the authoritative posterior. Criterion |dB-bar(0.10)| > 0.03 (P2B_04 §3.3).
--what m2i : Delta_T generalised to network-specific covariate effects on all 4,000 draws of vf_M2I_R3 (frozen A2 code):
             Delta_T = mean_T [pi(level + delta + iota_l z_l + iota_e z_e) - pi(level)]; criterion: median moves > 0.5 primary SD or
             P(Delta_T > 0) < 0.9. Interaction posteriors with prior-to-posterior contraction; Delta_T by size class / overflow tercile.
Outputs: tables/VF_rsn_only_*.csv, tables/VF_m2i_*.csv, diagnostics/VF_rsn_only.json, diagnostics/VF_m2i.json
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtr

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import vf_lib as V  # noqa: E402

L = V.redirect()
P = V.P; B = V.B; R = V.R
sys.path.insert(0, str(V.P4 / "scripts"))
import p4_02_selection_sensitivity as A2  # noqa: E402  (frozen A2 code)

L = V.redirect()                                    # p4_02 import re-points L.ROOT at phase_04; point it back
PARS = ["alpha", "gamma", "delta_bar", "sigma_site", "sigma_u", "sigma_B", "sigma_tu", "rho_tu", "sigma_A", "sigma_C", "sigma_w", "rho", "sigma_om", "sigma",
        "lambda", "tau_sig", "sigma_e"]


def rhat_ess(x, chains=4):
    import arviz as az
    x = np.asarray(x, float); n = len(x) // chains
    a = x[: n * chains].reshape(chains, n)
    return float(az.rhat(a)), float(az.ess(a, method="bulk"))


def sens_adequacy(tag, target):
    d = json.load(open(V.VF / "diagnostics" / f"{tag}_sampling_diagnostics.json"))
    g = V.CFG["adequacy"]["sensitivity"]
    lr = d.get("log_scale_diagnostics") or {}
    worst = max([d["hyper_max_rhat"]] + [v["r_hat"] for v in lr.values()])
    chk = dict(hyper_rhat=worst <= g["max_rhat_hyper"], divergences=d["divergence_share"] <= g["max_divergence_share"], ebfmi=min(d["ebfmi"]) >= g["min_ebfmi"],
               target_rhat=all(t[0] <= g["max_rhat_target"] for t in target.values()), target_ess=all(t[1] >= g["min_ess_target"] for t in target.values()))
    return dict(checks=chk, pass_all=bool(all(chk.values())), hyper_max_rhat=d["hyper_max_rhat"], hyper_argmax_rhat=d["hyper_argmax_rhat"], worst_incl_log_scale=worst,
                divergences=d["divergences"], min_ebfmi=min(d["ebfmi"]), hyper_min_ess_bulk=d["hyper_min_ess_bulk"], cpu_h=d["cpu_h"], target=target)


def primary_post(n):
    """Authoritative main-fit posterior from the Phase 3C-F scratch (V.redirect() points L.scratch_dir elsewhere)."""
    src = Path(os.environ.get("TRACK_A_SCRATCH_3CF", os.path.expanduser(L.CFG["scratch_dir"]))) / f"{P.TAG}_draws.npz"
    D = dict(np.load(src)); S = len(D["lp__"]); idx = np.round(np.linspace(0, S - 1, n)).astype(int)
    return {k: (v[idx] if (np.ndim(v) >= 1 and len(v) == S) else v) for k, v in D.items()}


def par_compare(Dv, Dp):
    rows = []
    for k in PARS:
        if k in Dv and k in Dp and np.size(Dv[k]):
            a, b = B.get1(Dv, k), B.get1(Dp, k)
            rows.append(dict(parameter=k, primary_median=float(np.median(b)), primary_sd=float(b.std()), refit_median=float(np.median(a)), refit_sd=float(a.std()),
                             shift_in_primary_sd=float((np.median(a) - np.median(b)) / b.std())))
    for j in range(np.asarray(Dv["beta"]).shape[1]):
        a = np.asarray(Dv["beta"], float)[:, j]
        b = np.asarray(Dp["beta"], float)[:, j] if j < np.asarray(Dp["beta"]).shape[1] else None
        rows.append(dict(parameter=f"beta[{j + 1}]", primary_median=float(np.median(b)) if b is not None else np.nan, primary_sd=float(b.std()) if b is not None else np.nan,
                         refit_median=float(np.median(a)), refit_sd=float(a.std()), shift_in_primary_sd=float((np.median(a) - np.median(b)) / b.std()) if b is not None else np.nan))
    return pd.DataFrame(rows)


def rsn():
    t0 = time.time()
    tag = "vf_RSN_R3"
    meas, sites, sy = L.load_tables(); mc = R.model_config("R3")
    Dfull = V.load_draws(tag)
    Dp = primary_post(4000)
    # ---- P^R_T with the frozen A2 code (2,000 draws), all thresholds and the primary psi grid
    D2 = V.thin(Dfull, 2000)
    S = A2.setup(D2, sites, sy, mc, P.CFG["A2_selection"]["gh_nodes"])
    prim = pd.read_csv(V.P4 / "tables/A2_PR_by_psi.csv")
    rows, target = [], {}
    for c in V.THRESH:
        for psi in P.CFG["A2_selection"]["psi_primary"]:
            pr = A2.pi_T(S, -psi, c)
            p0 = prim[(prim.threshold == c) & (np.isclose(prim.psi, psi))].iloc[0]
            rows.append(dict(threshold=c, psi=psi, **{f"PR_rsn_{k}": v for k, v in V.band(pr).items()}, PR_primary_median=p0.PR_median,
                             PR_primary_lo95=p0.PR_lo95, PR_primary_hi95=p0.PR_hi95, diff_median=float(np.median(pr) - p0.PR_median)))
            if psi == 0:
                target[f"PR_T_{c}"] = rhat_ess(pr)
    PRt = pd.DataFrame(rows); PRt.to_csv(V.VF / "tables/VF_rsn_only_PR_T.csv", index=False)
    # ---- B-bar with the frozen A3 code, run unchanged with redirected reader / writers
    import p4_03b_river_length as A3
    out = V.VF / "rsn_only"; (out / "tables").mkdir(parents=True, exist_ok=True); (out / "diagnostics").mkdir(exist_ok=True)
    sc = V.scratch() / "rsn_only"; sc.mkdir(exist_ok=True)
    for f in ["gs_links.parquet", "gs_tu_blanket.npz"]:
        if not (sc / f).exists():
            os.symlink(P.scratch() / f, sc / f)
    P.P4 = out
    P.scratch = lambda: sc
    P.load_post = lambda n=None: V.thin(Dfull, n) if n else Dfull
    A3.main()
    Bv = pd.read_csv(out / "tables/A3_river_length.csv"); Bp = pd.read_csv(V.TA / "phase_04/tables/A3_river_length.csv")
    PSv = pd.read_csv(out / "tables/A3_river_length_psi.csv"); PSp = pd.read_csv(V.TA / "phase_04/tables/A3_river_length_psi.csv")
    cmp_ = []
    for c in V.THRESH:
        for q, var in [("Bbar", "primary (tail-up conditional)"), ("B_2023", "primary (tail-up conditional)"), ("B_2024", "primary (tail-up conditional)"),
                       ("Bbar", "sensitivity: river term marginalised (no tail-up conditioning)")]:
            a = Bv[(Bv.threshold == c) & (Bv.quantity == q) & (Bv.variant == var)].iloc[0]; b = Bp[(Bp.threshold == c) & (Bp.quantity == q) & (Bp.variant == var)].iloc[0]
            cmp_.append(dict(threshold=c, quantity=q, variant=var.split(" (")[0].split(":")[0], rsn_only_median=a["median"], rsn_only_lo95=a.lo95, rsn_only_hi95=a.hi95,
                             primary_median=b["median"], primary_lo95=b.lo95, primary_hi95=b.hi95, diff=a["median"] - b["median"]))
        for psi in P.CFG["A2_selection"]["psi_primary"]:
            a = PSv[(PSv.threshold == c) & np.isclose(PSv.psi, psi)].iloc[0]; b = PSp[(PSp.threshold == c) & np.isclose(PSp.psi, psi)].iloc[0]
            cmp_.append(dict(threshold=c, quantity=f"Bbar_psi_{psi:+.1f}", variant="psi curve (7-node)", rsn_only_median=a["median"], rsn_only_lo95=a.lo95,
                             rsn_only_hi95=a.hi95, primary_median=b["median"], primary_lo95=b.lo95, primary_hi95=b.hi95, diff=a["median"] - b["median"]))
    CMP = pd.DataFrame(cmp_); CMP.to_csv(V.VF / "tables/VF_rsn_only_Bbar.csv", index=False)
    PC = par_compare(Dfull, Dp); PC.to_csv(V.VF / "tables/VF_rsn_only_parameters.csv", index=False)
    d10 = CMP[(CMP.threshold == 0.1) & (CMP.quantity == "Bbar") & (CMP.variant == "primary")].iloc[0]
    res = dict(adequacy=sens_adequacy(tag, target), criterion="|dBbar(0.10)| > 0.03 => material (P2B_04 §3.3)", dBbar_0_10=float(d10["diff"]),
               material=bool(abs(d10["diff"]) > 0.03), seconds=time.time() - t0,
               note="B-bar target R-hat / ESS are reported from the A3 run's 500 thinned draws in rsn_only/diagnostics")
    V.save_json(res, V.VF / "diagnostics/VF_rsn_only.json")
    pd.set_option("display.width", 220)
    print(CMP.round(4).to_string()); print(PRt.round(4).to_string()); print(PC.round(3).to_string()); print(json.dumps(res, indent=1, default=str))


def pi_units(S, shift, c):
    sh = np.asarray(shift, float); sh = sh[:, None] if sh.ndim == 1 else sh
    out = 0.0
    for wg, hh in S["h"]:
        out = out + wg * ndtr((S["L"] + sh + hh - np.log(c)) / S["sd"])
    return out


def m2i():
    t0 = time.time()
    tag = "vf_M2I_R3"
    meas, sites, sy = L.load_tables(); mc = R.model_config("R3")
    D = V.load_draws(tag)
    beta = np.asarray(D["beta"], float); K0 = 6
    iota = beta[:, K0:]
    D6 = dict(D, beta=beta[:, :K0])                                    # RSN level: no interaction columns
    S = A2.setup(D6, sites, sy, mc, P.CFG["A2_selection"]["gh_nodes"])
    Tm = sy.in_target_T.to_numpy() == 1; sidx = sy.site_idx.to_numpy()[Tm] - 1
    zl = sites.z_logarea.to_numpy()[sidx]; ze = sites.z_edm.to_numpy()[sidx]
    dl = B.get1(D, "delta_bar")
    shift = dl[:, None] + iota[:, :1] * zl[None, :] + iota[:, 1:2] * ze[None, :]
    prim = pd.read_csv(V.TA / "phase_03c_f/addendum_T1/tables/T1_combined_delta_T.csv")
    w = S["w"]
    size = sites.size_cls.astype(str).to_numpy()[sidx]
    terc = pd.qcut(sites.z_edm.to_numpy()[sidx], 3, labels=["low", "mid", "high"], duplicates="drop").astype(str)
    rows, sub, target = [], [], {}
    for c in V.THRESH:
        pm = pi_units(S, shift, c); p0 = pi_units(S, 0.0, c); pc = pi_units(S, dl, c)
        dT = (pm - p0) @ w; dT_common = (pc - p0) @ w
        pr = prim[np.isclose(prim.threshold, c)].iloc[0]
        target[f"Delta_T_{c}"] = rhat_ess(dT)
        rows.append(dict(threshold=c, **{f"DT_m2i_{k}": v for k, v in V.band(dT).items()}, DT_m2i_p_gt0=float((dT > 0).mean()),
                         **{f"DT_m2i_common_shift_{k}": v for k, v in V.band(dT_common).items()},
                         DT_primary_median=pr["median"], DT_primary_lo95=pr.lo95, DT_primary_hi95=pr.hi95, DT_primary_sd=pr.sd,
                         shift_in_primary_sd=float((np.median(dT) - pr["median"]) / pr.sd),
                         material=bool(abs(np.median(dT) - pr["median"]) > 0.5 * pr.sd or (dT > 0).mean() < 0.9)))
        for lab, grp in [("size", size), ("overflow_tercile", terc)]:
            for gv in sorted(set(grp)):
                m_ = grp == gv
                ww = w[m_] / w[m_].sum()
                x = (pm[:, m_] - p0[:, m_]) @ ww
                sub.append(dict(threshold=c, by=lab, group=gv, n_units=int(m_.sum()), **V.band(x), p_gt0=float((x > 0).mean())))
    DT = pd.DataFrame(rows); DT.to_csv(V.VF / "tables/VF_m2i_delta_T.csv", index=False)
    SB = pd.DataFrame(sub); SB.to_csv(V.VF / "tables/VF_m2i_delta_T_subgroups.csv", index=False)
    io = []
    for j, nm in enumerate(V.CFG["m2i"]["columns"]):
        x = iota[:, j]; r_, e_ = rhat_ess(x)
        io.append(dict(parameter=nm, **V.band(x), p_gt0=float((x > 0).mean()), prior_sd=V.CFG["m2i"]["prior_sd"],
                       contraction=float(1 - x.var() / V.CFG["m2i"]["prior_sd"] ** 2), rhat=r_, ess_bulk=e_))
    IO = pd.DataFrame(io); IO.to_csv(V.VF / "tables/VF_m2i_interactions.csv", index=False)
    PC = par_compare(D, primary_post(4000)); PC.to_csv(V.VF / "tables/VF_m2i_parameters.csv", index=False)
    res = dict(adequacy=sens_adequacy(tag, target), criterion="material if |median shift| > 0.5 primary SD or P(Delta_T > 0) < 0.9",
               material_any=bool(DT.material.any()), seconds=time.time() - t0)
    V.save_json(res, V.VF / "diagnostics/VF_m2i.json")
    pd.set_option("display.width", 220)
    print(DT.round(4).T.to_string()); print(IO.round(4).to_string()); print(SB.round(4).to_string()); print(PC.round(3).to_string())
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--what", required=True, choices=["rsn", "m2i"]); a = ap.parse_args()
    {"rsn": rsn, "m2i": m2i}[a.what]()
