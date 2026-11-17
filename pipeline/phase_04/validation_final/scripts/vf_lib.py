"""Track A Phase 4B (final validation): shared library.

Imports the Phase 4 library (and through it Phase 3C-F / 3C-R / 3C / 3B / 3A) READ-ONLY, bytecode writing disabled.
All writers are redirected: L.ROOT -> validation_final, L.scratch_dir -> ~/track_a_scratch/phase_04_vf. The authoritative
R3 executable and the frozen Phase 3C executable are RUN, never rebuilt (CmdStanModel(exe_file=...)).

Fit inputs per tag (config/vf_settings.yaml):
  fold      W1 / W2 / C1: frozen Phase 3C definitions (p3c_folds.heldout_sites); C2: frozen phase_04/validation/fold_assignments_C2_C3.csv.
            All sites, site-years and river-network nodes are kept; ONLY the held-out sites' measurements are removed.
  rsn_only  all MS measurements removed (MS sites kept as nodes without data: exactly equivalent to dropping them).
  m2i       full data; X gains ms * z_logarea and ms * z_edm with N(0, 0.25^2) priors (Phase 2B M2-I). Same program.
R3 inits: the four over-dispersed initial values of the authoritative main fit, with every latent quantity of a site or site-year
WITHOUT data in the fit reset (theta_i -> alpha + X_i beta of that init; z_om, z_sig -> 0), so no value informed by a held-out
measurement enters a fold fit. Metric / step size: the authoritative pilot metric (tuning only), windowed adaptation engaged.
"""
from __future__ import annotations

import json
import os
import resource
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.dont_write_bytecode = True
VF = Path(__file__).resolve().parents[1]
P4 = VF.parent
TA = P4.parent
sys.path.insert(0, str(P4 / "scripts"))
import p4_lib as P  # noqa: E402

L = P.L                       # phase_03c_f library
R = L.R; C = L.C; B = L.B; F = L.F; A = B.A
CFG = yaml.safe_load(open(VF / "config" / "vf_settings.yaml"))
THRESH = [0.05, 0.1, 0.2]
save_json = P.save_json; band = P.band; sha = P.sha


def scratch() -> Path:
    p = Path(os.path.expanduser(CFG["scratch_dir"])); p.mkdir(parents=True, exist_ok=True)
    return p


def redirect():
    """Point every Phase 3C-F writer at validation_final and its scratch."""
    L.ROOT = VF
    L.scratch_dir = scratch
    for d in ["diagnostics", "tables", "figures"]:
        (VF / d).mkdir(exist_ok=True)
    return L


def exe(model):
    return (P4 / CFG["exe_r3" if model == "R3" else "exe_m1s2"]).resolve()     # paths in the config are relative to phase_04


def heldout_sites(fold, sites):
    if fold in ("W1", "W2", "C1"):
        return F.heldout_sites(fold, sites)
    fa = pd.read_csv(P4 / "validation" / "fold_assignments_C2_C3.csv")
    mcs = set(fa[fa.fold == fold].mancat)
    return sorted(sites.site[sites.mancat.isin(mcs)])


def m2i_design(sites):
    X = A.site_design(sites); ms = sites.ms.to_numpy().astype(float)
    return np.column_stack([X, ms * X[:, 0], ms * X[:, 2]])


def fit_inputs(tag):
    """Return dict(data, mc, held, nodata_sites, nodata_siteyears, meas_train, model, settings)."""
    spec = CFG["fits"][tag]
    meas, sites, sy = L.load_tables()
    held = []
    if spec["kind"] == "fold":
        held = heldout_sites(spec["fold"], sites)
        mtr = F.training_measurements(meas, held)
    elif spec["kind"] == "rsn_only":
        mtr = meas[meas.network == "RSN"].copy()
    else:
        mtr = meas.copy()
    if spec["model"] == "R3":
        mc = R.model_config("R3")
        data = R.build_data(mtr, sites, sy, mc)
    else:
        mc = C.model_config(tu=0)
        data = C.build_data(mtr, sites, sy, mc)
    if spec["kind"] == "m2i":
        X = m2i_design(sites)
        data["K"] = int(X.shape[1]); data["X"] = X.tolist()
        data["beta_sd"] = list(map(float, data["beta_sd"])) + [CFG["m2i"]["prior_sd"]] * 2
        mc = dict(mc, m2i=1)
    has = set(mtr.site_idx.unique())
    nodata_sites = np.array([i for i in range(1, len(sites) + 1) if i not in has])
    has_sy = set(mtr.siteyear_idx.unique())
    nodata_sy = np.array([j for j in range(1, len(sy) + 1) if j not in has_sy])
    st = CFG["r3_sampling" if spec["model"] == "R3" else "m1s2_sampling"]
    return dict(data=data, mc=mc, held=held, nodata_sites=nodata_sites, nodata_siteyears=nodata_sy, meas_train=mtr,
                model=spec["model"], kind=spec["kind"], settings=st, sites=sites, sy=sy, meas=meas)


def r3_inits(fi, out_dir):
    """Main-fit over-dispersed inits with all no-data latents reset (see module docstring). Returns list of json paths."""
    inp = Path(os.path.expanduser(CFG["inputs_main"]))
    X = np.asarray(fi["data"]["X"], float)
    paths = []
    for k in range(1, 5):
        it = json.load(open(inp / f"init_{k}.json"))
        beta = np.asarray(it["beta"], float)
        if X.shape[1] > len(beta):
            beta = np.concatenate([beta, np.zeros(X.shape[1] - len(beta))])
            it["beta"] = beta.tolist()
        th = np.asarray(it["theta"], float); zo = np.asarray(it["z_om"], float); zs = np.asarray(it["z_sig"], float)
        ns = fi["nodata_sites"] - 1; nj = fi["nodata_siteyears"] - 1
        if len(ns):
            th[ns] = it["alpha"] + X[ns] @ beta
            zs[ns] = 0.0
        if len(nj):
            zo[nj] = 0.0
        it.update(theta=th.tolist(), z_om=zo.tolist(), z_sig=zs.tolist())
        p = Path(out_dir) / f"init_{k}.json"; json.dump(it, open(p, "w")); paths.append(str(p))
    return paths


def r3_metric(fi):
    inp = Path(os.path.expanduser(CFG["inputs_main"]))
    m = np.asarray(json.load(open(inp / "inv_metric.json"))["inv_metric"], float)
    K0 = 6
    extra = fi["data"]["K"] - K0
    if extra > 0:                                   # unconstrained order: alpha, beta[1:K], ...
        m = np.concatenate([m[:1 + K0], np.full(extra, float(np.exp(np.log(m[1:1 + K0]).mean()))), m[1 + K0:]])
    step = json.load(open(L.TA / "phase_03c_f" / "diagnostics" / "main_inputs.json"))["initial_step"]
    return m, float(step)


def get_model(model):
    import cmdstanpy
    return cmdstanpy.CmdStanModel(exe_file=str(exe(model)))


def run_fit(tag, iter_sampling=None, iter_warmup=None, chains=None, parallel_chains=None, summarise=True, seed=None, adapt_engaged=True,
            out_name=None):
    """Sample one validation fit (writes only to validation_final + its scratch). Returns (res, draws) or (fit, cpu, wall)."""
    redirect()
    fi = fit_inputs(tag)
    st = fi["settings"]
    name = out_name or tag
    outdir = scratch() / f"fit_{name}"
    if outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True)
    model = get_model(fi["model"])
    chains = chains or st["chains"]
    kw = {}
    if fi["model"] == "R3":
        inits = r3_inits(fi, outdir)[:chains]
        m, step = r3_metric(fi)
        kw.update(metric="diag_e", inv_metric={"inv_metric": m.tolist()}, step_size=step, inits=inits if chains > 1 else inits[0])
        seed = seed or CFG["seeds"]["stan_r3"]
    else:
        seed = seed or CFG["seeds"]["stan_m1s2"]
    iw = st["iter_warmup"] if iter_warmup is None else iter_warmup
    isamp = st["iter_sampling"] if iter_sampling is None else iter_sampling
    r0 = resource.getrusage(resource.RUSAGE_CHILDREN); t0 = time.time()
    fit = model.sample(data=fi["data"], chains=chains, parallel_chains=parallel_chains or st["parallel_chains"], iter_warmup=iw, iter_sampling=isamp,
                       max_treedepth=st["max_treedepth"], seed=seed, output_dir=str(outdir), show_progress=False,
                       sig_figs=9, refresh=25, **(dict(adapt_delta=st["adapt_delta"]) if adapt_engaged else dict(adapt_engaged=False)), **kw)
    wall = time.time() - t0; r1 = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu = (r1.ru_utime - r0.ru_utime) + (r1.ru_stime - r0.ru_stime)
    if not summarise:
        return fit, cpu, wall, fi
    extra = dict(kind=fi["kind"], fold=CFG["fits"][tag].get("fold"), n_held_out_sites=len(fi["held"]), held_out_sites=fi["held"],
                 n_nodata_sites=int(len(fi["nodata_sites"])), N_train=int(fi["data"]["N"]), parallel_chains=parallel_chains or st["parallel_chains"],
                 inits=("main-fit over-dispersed inits, no-data latents reset" if fi["model"] == "R3" else "random (Stan default)"),
                 metric=("authoritative pilot metric, windowed adaptation" if fi["model"] == "R3" else "Stan default adaptation"),
                 executable=str(exe(fi["model"])), executable_sha256=sha(exe(fi["model"])))
    res, draws, idata = L.summarise_fit(fit, tag, fi["data"], fi["mc"], cpu, wall, iw, isamp, chains, st["adapt_delta"], st["max_treedepth"], seed, extra)
    return res, draws


def load_draws(tag):
    return dict(np.load(scratch() / f"{tag}_draws.npz"))


def comparator_draws(fold):
    if fold in CFG["existing_comparators"]:
        return dict(np.load(os.path.expanduser(CFG["existing_comparators"][fold])))
    return load_draws(f"vf_{fold}_m1s2")


def thin(D, n):
    S = len(D["lp__"]) if "lp__" in D else B.ndraws(D)
    idx = np.round(np.linspace(0, S - 1, n)).astype(int)
    return {k: (np.asarray(v)[idx] if (np.ndim(v) >= 1 and len(v) == S) else v) for k, v in D.items()}
