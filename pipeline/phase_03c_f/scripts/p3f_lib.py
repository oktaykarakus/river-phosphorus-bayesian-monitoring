"""Shared library for Track A Phase 3C-F (computational closure and model-selection decision).

Imports the Phase 3C-R library (phase_03c_r/scripts/p3r_lib.py, which imports Phase 3C / 3B / 3A) READ-ONLY for data tables,
R3 data structures, Delta_T, PPC and fold scores. Nothing in an earlier phase is written: p3r_lib.run_fit (which writes to
phase_03c_r) is never called; this module has its own runner writing only to phase_03c_f and its scratch directory.
The Stan program is a byte-identical copy of the Phase 3C-R R3 program, compiled inside phase_03c_f/stan.
"""
from __future__ import annotations

import os
import resource
import shutil
import sys
sys.dont_write_bytecode = True
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]          # phase_03c_f
TA = ROOT.parent
sys.path.insert(0, str(TA / "phase_03c_r" / "scripts"))
import p3r_lib as R  # noqa: E402  Phase 3C-R library (read-only)

C = R.C; B = R.B; F = R.F
CFG = yaml.safe_load(open(ROOT / "config" / "phase3cf_settings.yaml"))
STAN_FILE = ROOT / "stan" / "track_a_m2rn_r3.stan"
THRESH = R.THRESH
save_json = R.save_json
load_tables = R.load_tables
HYPER = R.HYPER
TOP = R.TOP


def scratch_dir() -> Path:
    p = Path(os.environ.get("TRACK_A_SCRATCH_3CF", os.path.expanduser(CFG["scratch_dir"])))
    p.mkdir(parents=True, exist_ok=True)
    return p


def r3_data():
    meas, sites, sy = load_tables()
    mc = R.model_config("R3")
    return meas, sites, sy, mc, R.build_data(meas, sites, sy, mc)


def get_model():
    return R.get_model(STAN_FILE)


def load_draws(tag, where="3cf"):
    if where == "3cf":
        return dict(np.load(scratch_dir() / f"{tag}_draws.npz"))
    return R.load_draws(tag, where=where)


def summarise_fit(fit, tag, data, mc, cpu, wall, iter_warmup, iter_sampling, chains, adapt_delta, max_treedepth, seed, extra=None):
    """Diagnostics dict (Phase 3C-R definitions), posterior summary table and draws archive (phase_03c_f only)."""
    import arviz as az
    mv = fit.method_variables()
    idata = az.from_cmdstanpy(posterior=fit)
    post = idata.posterior
    top = [v for v in TOP if v in post and post[v].size > 0]
    summ = az.summary(idata, var_names=top, round_to=6)
    extra_rows = {}
    for v in ["rho_tu", "rho", "sigma_u", "sigma_B", "sigma_tu", "sigma_site"]:
        if v in post and post[v].size > 0:
            lv = np.log(post[v].values.reshape(post.sizes["chain"], post.sizes["draw"], -1)[..., 0])
            extra_rows[f"log_{v}"] = dict(r_hat=float(az.rhat(lv)), ess_bulk=float(az.ess(lv, method="bulk")), ess_tail=float(az.ess(lv, method="tail")))
    lat_vars = [v for v in ["theta", "z_om", "z_sig", "vB", "z_C", "z_A", "w_site", "z_e", "v_tu"] if v in post and post[v].size > 0]
    lat = az.summary(idata, var_names=lat_vars, kind="diagnostics")
    hyper = summ.loc[[i for i in summ.index if i.split("[")[0] in HYPER]]
    res = dict(tag=tag, model_config=mc, wall_s=wall, cpu_s=cpu, cpu_h=cpu / 3600, chains=chains, iter_warmup=iter_warmup,
               iter_sampling=iter_sampling, adapt_delta=adapt_delta, max_treedepth=max_treedepth, seed=seed, I=data["I"], J=data["J"], N=data["N"],
               divergences=int(mv["divergent__"].sum()), divergences_by_chain=np.asarray(mv["divergent__"]).sum(axis=0).astype(int).tolist(),
               divergence_share=float(np.asarray(mv["divergent__"]).mean()),
               treedepth_hits=int((mv["treedepth__"] >= max_treedepth).sum()), mean_treedepth=float(mv["treedepth__"].mean()),
               mean_leapfrog=float(mv["n_leapfrog__"].mean()), leapfrog_by_chain=np.asarray(mv["n_leapfrog__"]).mean(axis=0).round(1).tolist(),
               stepsize_by_chain=np.atleast_2d(mv["stepsize__"])[0].round(5).tolist(),
               ebfmi=[float(x) for x in az.bfmi(idata)],
               hyper_max_rhat=float(hyper["r_hat"].max()), hyper_argmax_rhat=str(hyper["r_hat"].idxmax()),
               hyper_min_ess_bulk=float(hyper["ess_bulk"].min()), hyper_argmin_ess_bulk=str(hyper["ess_bulk"].idxmin()),
               hyper_min_ess_tail=float(hyper["ess_tail"].min()), hyper_argmin_ess_tail=str(hyper["ess_tail"].idxmin()),
               log_scale_diagnostics=extra_rows,
               latent_max_rhat=float(lat["r_hat"].max()), latent_argmax_rhat=str(lat["r_hat"].idxmax()),
               latent_min_ess_bulk=float(lat["ess_bulk"].min()), latent_min_ess_tail=float(lat["ess_tail"].min()),
               gradient_evals_sampling=int(np.asarray(mv["n_leapfrog__"]).sum()))
    if extra:
        res.update(extra)
    summ.to_csv(ROOT / "tables" / f"{tag}_posterior_summary.csv")
    draws = {}
    for v in list(post.data_vars):
        arr = post[v].values
        draws[v] = arr.reshape(arr.shape[0] * arr.shape[1], *arr.shape[2:])
    for v in ["divergent__", "treedepth__", "n_leapfrog__", "lp__", "energy__", "stepsize__", "accept_stat__"]:
        draws[v] = np.asarray(mv[v]).T.reshape(-1)
    np.savez(scratch_dir() / f"{tag}_draws.npz", **draws)
    save_json(res, ROOT / "diagnostics" / f"{tag}_sampling_diagnostics.json")
    return res, draws, idata


def run_fit(tag, data, mc, iter_warmup, iter_sampling, chains=4, adapt_delta=0.98, max_treedepth=10, inits=None, seed=None,
            inv_metric=None, step_size=None, save_warmup=False, refresh=25, summarise=True, extra=None):
    """CmdStan NUTS run of the phase_03c_f R3 program. CSVs are kept in scratch (metric / step size retained)."""
    outdir = scratch_dir() / f"fit_{tag}"
    if outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True)
    model = get_model()
    seed = seed or CFG["model"]["seeds"]["stan"]
    kw = {}
    if inv_metric is not None:
        kw.update(metric="diag_e", inv_metric={"inv_metric": list(map(float, inv_metric))})
    if step_size is not None:
        kw["step_size"] = step_size
    r0 = resource.getrusage(resource.RUSAGE_CHILDREN); t0 = time.time()
    fit = model.sample(data=data, chains=chains, parallel_chains=chains, iter_warmup=iter_warmup, iter_sampling=iter_sampling,
                       adapt_delta=adapt_delta, max_treedepth=max_treedepth, seed=seed, output_dir=str(outdir), show_progress=False,
                       inits=inits, sig_figs=9, save_warmup=save_warmup, refresh=refresh, **kw)
    wall = time.time() - t0; r1 = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu = (r1.ru_utime - r0.ru_utime) + (r1.ru_stime - r0.ru_stime)
    if not summarise:
        return fit, cpu, wall
    return summarise_fit(fit, tag, data, mc, cpu, wall, iter_warmup, iter_sampling, chains, adapt_delta, max_treedepth, seed, extra)


def read_cmdstan_csv(path):
    """Read a (possibly partial) CmdStan CSV: draws DataFrame, adapted step size and inverse metric (if present)."""
    import pandas as pd
    step, metric, lines = None, None, open(path).read().splitlines()
    for i, ln in enumerate(lines):
        if ln.startswith("# Step size ="):
            step = float(ln.split("=")[1])
        if ln.startswith("# Diagonal elements of inverse mass matrix:"):
            metric = np.array([float(x) for x in lines[i + 1].lstrip("# ").split(",")])
    body = [ln for ln in lines if not ln.startswith("#")]
    header = body[0].split(","); rows = [r for r in body[1:] if r.count(",") == len(header) - 1]
    from io import StringIO
    df = pd.read_csv(StringIO("\n".join([body[0]] + rows)))
    return df, step, metric
