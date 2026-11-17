"""Shared library for Track A Phase 3C-R (river-network model correction).

Imports the frozen Phase 3C library (phase_03c/scripts/p3c_lib.py, which imports Phase 3B / 3A) READ-ONLY for data tables,
river-network structures, posterior reconstruction, Delta_T, fold definitions and held-out scores. Nothing in any earlier
phase is written: Phase 3C's run_fit / get_model (which write to phase_03c) are never called; this module has its own.
Adds: the Phase 3C-R Stan program (stan/track_a_m2rn_r.stan, candidates R0 / R1 / R2), a fit runner writing only to
phase_03c_r and its scratch directory, and candidate-aware draw loading.
"""
from __future__ import annotations

import json
import os
import resource
import shutil
import sys
sys.dont_write_bytecode = True   # never write __pycache__ into frozen phase trees
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]          # phase_03c_r
TA = ROOT.parent
P3C = TA / "phase_03c"
sys.path.insert(0, str(P3C / "scripts"))
import p3c_lib as C   # noqa: E402  frozen Phase 3C library
import p3c_folds as F  # noqa: E402  frozen Phase 3C folds / scores

B = C.B; A = C.A
CFG = yaml.safe_load(open(ROOT / "config" / "phase3cr_settings.yaml"))
CFG3C = C.CFG
STAN_FILE = ROOT / "stan" / "track_a_m2rn_r.stan"
P3C_EXE = P3C / "stan" / "track_a_m2rn"              # frozen Phase 3C executable (run, never rebuilt)
THRESH = C.THRESH
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

CANDIDATES = {"R0": dict(site_nc=0, use_wb=1), "R1": dict(site_nc=1, use_wb=1), "R2": dict(site_nc=1, use_wb=0),
              "R3": dict(site_nc=0, use_wb=1, marginal=1)}
STAN_FILE_R3 = ROOT / "stan" / "track_a_m2rn_r3.stan"


def scratch_dir() -> Path:
    p = Path(os.environ.get("TRACK_A_SCRATCH_3CR", os.path.expanduser(CFG["scratch_dir"])))
    p.mkdir(parents=True, exist_ok=True)
    return p


def p3c_scratch() -> Path:
    return Path(os.path.expanduser(CFG["phase3c_scratch"]))


load_tables = C.load_tables
save_json = C.save_json
band = B.band


def model_config(cand="R1", tu=1, **kw):
    mc = C.model_config(tu=tu, **kw)
    mc.update(CANDIDATES[cand]); mc["candidate"] = cand
    return mc


def r3_structures(sites):
    """Blocks of the marginal site-level covariance (connected components of same river tree OR same water body), the
    same-water-body indicators per block and the flow-connected (tail-up) pairs with river distance and log path weight."""
    rn = C.rn_structures(sites)
    wb = B.hier_indices(sites)["wb"]; tree = rn["tree"]; I = len(sites)
    par = np.arange(I)

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]; x = par[x]
        return x
    for key in (tree, wb):
        first = {}
        for i, k in enumerate(key):
            if k in first:
                a, b = find(i), find(first[k])
                if a != b:
                    par[a] = b
            else:
                first[k] = i
    root = np.array([find(i) for i in range(I)])
    labels, comp = np.unique(root, return_inverse=True)
    blocks = [np.flatnonzero(comp == k) for k in range(len(labels))]
    ci, cc, hs, lw = C.tu_chains(rn)
    blk_sites, blk_size, blk_start, wbs, wbs_start, pr_i, pr_j, pr_h, pr_lw, pr_start, pr_n = [], [], [], [], [], [], [], [], [], [], []
    pos = 1; wpos = 1; ppos = 1
    loc = np.empty(I, int); bid = np.empty(I, int)
    for k, b in enumerate(blocks):
        loc[b] = np.arange(len(b)); bid[b] = k
    pairs_by_blk = {}
    for i, c, h, l in zip(ci, cc, hs, lw):
        assert bid[i] == bid[c]
        pairs_by_blk.setdefault(bid[i], []).append((loc[i] + 1, loc[c] + 1, h, l))
    for k, b in enumerate(blocks):
        n = len(b)
        blk_sites += (b + 1).tolist(); blk_size.append(n); blk_start.append(pos); pos += n
        Wm = (wb[b][:, None] == wb[b][None, :]).astype(float)
        wbs += Wm.T.reshape(-1).tolist(); wbs_start.append(wpos); wpos += n * n
        pl = pairs_by_blk.get(k, [])
        pr_start.append(ppos); pr_n.append(len(pl)); ppos += len(pl)
        for a_, c_, h, l in pl:
            pr_i.append(int(a_)); pr_j.append(int(c_)); pr_h.append(float(h)); pr_lw.append(float(min(l, 0.0)))
    single = [int(b[0]) + 1 for b in blocks if len(b) == 1]
    return dict(n_blk=len(blocks), blk_size=blk_size, blk_start=blk_start, blk_sites=blk_sites, n_wbs=len(wbs), wbs_start=wbs_start, wbs=wbs,
                n_pr=len(pr_i), pr_start=pr_start, pr_n=pr_n, pr_i=pr_i, pr_j=pr_j, pr_h=pr_h, pr_lw=pr_lw, n_single=len(single), single_sites=single)


def build_data(meas, sites, sy, mc, J_chk=50):
    d = C.build_data(meas, sites, sy, mc, J_chk=J_chk)
    if mc.get("marginal", 0):
        d.update(r3_structures(sites))
    else:
        d.update(site_nc=int(mc.get("site_nc", 0)), use_wb=int(mc.get("use_wb", 1)))
    return d


# ====================================================================== fitting (Phase 3C-R outputs only)
TOP = ["sigma_site", "share", "alpha", "beta", "gamma", "harm", "delta_bar", "tau_delta", "tau_s", "Cs", "sigma_e", "sigma_u", "sigma_om", "sigma", "lambda", "tau_sig",
       "sigma_A", "sigma_C", "sigma_B", "sigma_w", "rho", "sigma_tu", "rho_tu", "sigma_td", "rho_td"]
HYPER = ["sigma_site", "share", "sigma_u", "sigma_B", "sigma_tu", "rho_tu", "sigma_A", "sigma_C", "sigma_w", "rho", "sigma_om", "sigma", "lambda", "tau_sig", "sigma_e",
         "tau_s", "delta_bar", "alpha", "beta", "gamma", "harm"]


def get_model(stan_file=STAN_FILE):
    import cmdstanpy
    stan_file = Path(stan_file)
    exe = stan_file.with_suffix("")
    if exe.exists() and exe.stat().st_mtime < stan_file.stat().st_mtime:
        exe.unlink()                                  # macOS: a recompiled-in-place executable is killed (rc -9)
    return cmdstanpy.CmdStanModel(stan_file=str(stan_file))


def get_p3c_model():
    import cmdstanpy
    return cmdstanpy.CmdStanModel(exe_file=str(P3C_EXE))


def run_fit(tag, data, mc, iter_warmup=500, iter_sampling=500, chains=4, adapt_delta=0.95, max_treedepth=10, inits=None, seed=None,
            keep_csv=False, extra_diag=None, parallel_chains=None, frozen_p3c_program=False, metric=None, step_size=None):
    import cmdstanpy, arviz as az
    outdir = scratch_dir() / f"fit_{tag}"
    if outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True)
    model = get_p3c_model() if frozen_p3c_program else get_model(STAN_FILE_R3 if mc.get("marginal", 0) else STAN_FILE)
    if frozen_p3c_program:
        data = {k: v for k, v in data.items() if k not in ("site_nc", "use_wb")}
    seed = seed or CFG3C["seeds"]["stan"]
    r0 = resource.getrusage(resource.RUSAGE_CHILDREN); t0 = time.time()
    fit = model.sample(data=data, chains=chains, parallel_chains=parallel_chains or chains, iter_warmup=iter_warmup, iter_sampling=iter_sampling,
                       adapt_delta=adapt_delta, max_treedepth=max_treedepth, seed=seed, output_dir=str(outdir), show_progress=False, inits=inits, sig_figs=9,
                       **({"metric": metric} if metric is not None else {}), **({"step_size": step_size} if step_size is not None else {}))
    wall = time.time() - t0; r1 = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu = (r1.ru_utime - r0.ru_utime) + (r1.ru_stime - r0.ru_stime)
    mv = fit.method_variables()
    idata = az.from_cmdstanpy(posterior=fit)
    post = idata.posterior
    top = [v for v in TOP if v in post and post[v].size > 0]
    summ = az.summary(idata, var_names=top, round_to=6)
    # log rho_tu diagnostics (the sampler works on log rho_tu)
    extra_rows = {}
    for v in ["rho_tu", "rho"]:
        if v in post and post[v].size > 0:
            lv = np.log(post[v].values.reshape(post.sizes["chain"], post.sizes["draw"]))
            extra_rows[f"log_{v}"] = dict(r_hat=float(az.rhat(lv)), ess_bulk=float(az.ess(lv, method="bulk")), ess_tail=float(az.ess(lv, method="tail")))
    lat_vars = [v for v in ["theta", "z_om", "z_sig", "vB", "z_C", "z_A", "w_site", "z_e", "z_tu", "v_tu", "theta_par", "share"] if v in post and post[v].size > 0]
    lat = az.summary(idata, var_names=lat_vars, kind="diagnostics")
    hyper = summ.loc[[i for i in summ.index if i.split("[")[0] in HYPER]]
    res = dict(tag=tag, model_config=mc, frozen_p3c_program=frozen_p3c_program, wall_s=wall, cpu_s=cpu, cpu_h=cpu / 3600,
               peak_child_rss_mb=r1.ru_maxrss / 1e6, chains=chains, iter_warmup=iter_warmup, iter_sampling=iter_sampling, adapt_delta=adapt_delta,
               max_treedepth=max_treedepth, seed=seed, I=data["I"], J=data["J"], N=data["N"],
               divergences=int(mv["divergent__"].sum()), divergences_by_chain=np.asarray(mv["divergent__"]).sum(axis=0).astype(int).tolist(),
               divergence_share=float(np.asarray(mv["divergent__"]).mean()),
               treedepth_hits=int((mv["treedepth__"] >= max_treedepth).sum()), mean_treedepth=float(mv["treedepth__"].mean()),
               mean_leapfrog=float(mv["n_leapfrog__"].mean()), leapfrog_by_chain=np.asarray(mv["n_leapfrog__"]).mean(axis=0).round(1).tolist(),
               stepsize_by_chain=np.atleast_2d(mv["stepsize__"])[0].round(5).tolist(),
               ebfmi=[float(x) for x in az.bfmi(idata)],
               top_max_rhat=float(summ["r_hat"].max()), top_argmax_rhat=str(summ["r_hat"].idxmax()),
               top_min_ess_bulk=float(summ["ess_bulk"].min()), top_argmin_ess_bulk=str(summ["ess_bulk"].idxmin()),
               top_min_ess_tail=float(summ["ess_tail"].min()), top_argmin_ess_tail=str(summ["ess_tail"].idxmin()),
               hyper_max_rhat=float(hyper["r_hat"].max()), hyper_argmax_rhat=str(hyper["r_hat"].idxmax()),
               hyper_min_ess_bulk=float(hyper["ess_bulk"].min()), hyper_argmin_ess_bulk=str(hyper["ess_bulk"].idxmin()),
               hyper_min_ess_tail=float(hyper["ess_tail"].min()), hyper_argmin_ess_tail=str(hyper["ess_tail"].idxmin()),
               log_range_diagnostics=extra_rows,
               latent_max_rhat=float(lat["r_hat"].max()), latent_argmax_rhat=str(lat["r_hat"].idxmax()),
               latent_min_ess_bulk=float(lat["ess_bulk"].min()), latent_min_ess_tail=float(lat["ess_tail"].min()),
               gradient_evals_total_sampling=int(np.asarray(mv["n_leapfrog__"]).sum()))
    res["ms_per_gradient_approx"] = 1000 * cpu / max(1, (iter_warmup + iter_sampling) / iter_sampling * res["gradient_evals_total_sampling"])
    if extra_diag:
        res.update(extra_diag)
    cw = {}
    for v in ["alpha", "delta_bar", "sigma_u", "sigma_om", "sigma", "sigma_w", "rho", "sigma_B", "sigma_C", "sigma_A", "sigma_tu", "rho_tu"]:
        if v in post and post[v].size > 0:
            cw[v] = post[v].values.reshape(post.sizes["chain"], post.sizes["draw"], -1)[..., 0].mean(axis=1).round(4).tolist()
    res["chain_means"] = cw
    summ.to_csv(ROOT / "tables" / f"{tag}_posterior_summary.csv")
    draws = {}
    for v in list(post.data_vars):
        arr = post[v].values
        draws[v] = arr.reshape(arr.shape[0] * arr.shape[1], *arr.shape[2:])
    for v in ["divergent__", "treedepth__", "n_leapfrog__", "lp__", "energy__", "stepsize__"]:
        draws[v] = np.asarray(mv[v]).T.reshape(-1)
    np.savez(scratch_dir() / f"{tag}_draws.npz", **draws)
    save_json(res, ROOT / "diagnostics" / f"{tag}_sampling_diagnostics.json")
    if not keep_csv:
        shutil.rmtree(outdir, ignore_errors=True)
    return res, draws, idata


def load_draws(tag, where="3cr"):
    base = {"3cr": scratch_dir(), "3c": p3c_scratch(), "3b": C.ref_scratch()}[where]
    D = dict(np.load(base / f"{tag}_draws.npz"))
    if "sigma_B" not in D or np.size(D["sigma_B"]) == 0:          # R2: no water-body effect -> sigma_B = 0, vB = 0
        D["sigma_B"] = np.zeros(len(D["alpha"]))
    return D


def thin(D, n=1000):
    S = B.ndraws(D); sel = np.arange(0, S, max(1, S // n))
    return {k: (np.asarray(v)[sel] if np.ndim(v) >= 1 and len(v) == S else v) for k, v in D.items()}


def chain_index(D, chains=4):
    S = B.ndraws(D)
    return np.repeat(np.arange(chains), S // chains)


def screen_verdict(res, gates=None):
    g = gates or CFG["screen_gates"]
    st = np.array(res["stepsize_by_chain"])
    stuck = any(st[k] < 0.5 * np.median(np.delete(st, k)) for k in range(len(st))) if len(st) > 1 else False
    lr = res.get("log_range_diagnostics", {}).get("log_rho_tu", {})
    checks = dict(divergences=res["divergence_share"] <= g["max_divergence_share"],
                  rhat=max(res["hyper_max_rhat"], lr.get("r_hat", 1.0)) <= g["max_rhat_hyper"],
                  ess=min(res["hyper_min_ess_bulk"], lr.get("ess_bulk", 1e9)) >= g["min_bulk_ess_hyper"],
                  ebfmi=min(res["ebfmi"]) >= g["min_ebfmi_each_chain"], no_stuck_chain=not stuck)
    return dict(checks=checks, pass_all=bool(all(checks.values())))
