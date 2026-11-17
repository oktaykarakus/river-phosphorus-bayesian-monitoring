"""Shared library for Track A Phase 3B (M0-H ... M1-S2).

Re-uses the validated Phase 3A data tables and annual-mean / standardisation code (phase_03a/scripts/p3a_lib.py,
imported read-only) and adds: hierarchy indices, the HSGP basis, a configurable Stan data builder, a fit runner that
writes CmdStan CSVs to local SSD scratch, posterior reconstruction, pointwise log-likelihood, the generalised
standardised contrast Delta_T, and the residual spatial / flow-connectivity statistics (P3A_06 §4).
Specification: phase_02b/specifications/model_specification.md (MS) + phase_03b/specifications/phase_02b_addendum_2026-11-12.md.
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
from scipy.special import ndtr, log_ndtr

ROOT = Path(__file__).resolve().parents[1]          # phase_03b
TA = ROOT.parent
P3A = TA / "phase_03a"
P2B = TA / "phase_02b"
sys.path.insert(0, str(P3A / "scripts"))
import p3a_lib as A  # noqa: E402  (frozen Phase 3A library: data tables, priors + override C2, annual mean helpers)

STAN_FILE = ROOT / "stan" / "track_a_m1.stan"
CFG = yaml.safe_load(open(ROOT / "config" / "phase3b_settings.yaml"))
THRESH = A.THRESH
REGIONS = A.REGIONS
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


def scratch_dir() -> Path:
    p = Path(os.environ.get("TRACK_A_SCRATCH", os.path.expanduser(CFG["scratch_dir"])))
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_model():
    import cmdstanpy
    exe = STAN_FILE.with_suffix("")
    if exe.exists() and exe.stat().st_mtime < STAN_FILE.stat().st_mtime:
        exe.unlink()                       # macOS kills an executable overwritten in place (Phase 3A gotcha)
    return cmdstanpy.CmdStanModel(stan_file=str(STAN_FILE))


def load_tables():
    return A.load_tables()                 # phase_03a/data_processed/{measurements,sites,siteyears}.parquet (frozen)


# ------------------------------------------------------------------ hierarchy indices
def hier_indices(sites: pd.DataFrame):
    """1-based contiguous WMA / management-catchment / water-body indices. Sites with no catchment or water body
    (11 / 7 in the primary data) get their OWN new level, i.e. an effect drawn from the population distribution,
    not a shared 'unassigned' level (P3A_06 §3)."""
    out = {}
    for col in ["wma", "mancat", "wb"]:
        v = sites[col].astype(object).where(sites[col].notna(), None).tolist()
        v = [x if (x is not None and str(x) not in ("", "nan", "None")) else f"__new_{col}_{s}" for x, s in zip(v, sites.site)]
        levels = sorted(set(v))
        mp = {lv: k + 1 for k, lv in enumerate(levels)}
        out[col] = np.array([mp[x] for x in v], dtype=int)
        out[f"{col}_levels"] = levels
        out[f"{col}_n_new"] = int(sum(str(x).startswith("__new_") for x in levels))
    return out


# ------------------------------------------------------------------ HSGP basis (MS-§4)
def hsgp_constants():
    sc = json.load(open(P2B / "specifications/standardisation_constants.json"))
    return sc


def hsgp_basis(x_km, y_km, m=None, c_b=None):
    """Separable 1-D Laplacian eigenfunctions on [-L, L], L = c_b * S (Riutort-Mayol et al. 2023, Stat. Comput.).
    phi_j(x) = L^{-1/2} sin(sqrt(lam_j)(x + L)),  sqrt(lam_j) = j pi / (2L)."""
    sc = hsgp_constants()
    cE, cN = sc["hsgp_centre_km"]
    SE, SN = sc["hsgp_half_range_km"]
    m = m or sc["hsgp_m"]; c_b = c_b or sc["hsgp_boundary_factor"]
    out = []
    for coord, cen, S, mm in [(np.asarray(x_km), cE, SE, m[0]), (np.asarray(y_km), cN, SN, m[1])]:
        Lb = c_b * S
        sl = np.arange(1, mm + 1) * np.pi / (2 * Lb)
        phi = np.sin(np.outer(coord - cen + Lb, sl)) / np.sqrt(Lb)
        out += [phi, sl]
    return out  # PhiE, sqrt_lamE, PhiN, sqrt_lamN


def hsgp_field(PhiE, PhiN, slE, slN, Z, sigma_w, rho):
    """Python reference of the Stan function hsgp_field. Z: (S, m1, m2); sigma_w, rho: (S,). Returns (S, I)."""
    LAM2 = slE[:, None] ** 2 + slN[None, :] ** 2
    spd = sigma_w[:, None, None] * np.sqrt(2 * np.pi) * rho[:, None, None] * np.exp(-0.25 * rho[:, None, None] ** 2 * LAM2[None])
    B = spd * Z
    return np.einsum("ij,sjk,ik->si", PhiE, B, PhiN, optimize=True)


# ------------------------------------------------------------------ Stan data
def model_config(name: str, **kw) -> dict:
    """Named model configurations. kw overrides any key."""
    base = dict(network_mode=1, regional_season=0, modifiers=[], n_hmod=0, hetero=1, hier=0, field=0, shock=0, wb_marginal=1, sigma_om_sd=CFG["sigma_om_priors"]["primary"])
    sc = CFG["seasonal_candidates"]
    if name in sc:
        base.update(regional_season=sc[name]["regional"], modifiers=sc[name]["modifiers"], n_hmod=sc[name]["harmonics_modulated"], shock=sc[name].get("shock", 0),
                    shock_centre=sc[name].get("shock_centre", 1))
    base.update(kw)
    return base


def seasonal_modifiers(sites, modifiers):
    cols = []
    for mname in modifiers:
        if mname == "ms":
            cols.append(sites.ms.to_numpy().astype(float))
        elif mname == "z_logarea":
            cols.append(sites.z_logarea.to_numpy().astype(float))
        else:
            raise ValueError(mname)
    return np.column_stack(cols) if cols else np.zeros((len(sites), 0))


def build_data(meas, sites, sy, mc: dict, J_chk=50, Q=24, priors=None):
    d = A.build_stan_data(meas, sites, sy, network_mode=mc["network_mode"], regional_season=mc["regional_season"], J_chk=J_chk, Q=Q,
                          priors=priors, centred_site=1, centred_siteyear=0, hetero_sigma=mc["hetero"])
    for k in ["use_interval", "log_lo", "log_hi", "prior_only", "centred_site", "centred_siteyear"]:
        d.pop(k, None)
    d["sigma_om_sd"] = float(mc["sigma_om_sd"])
    d["lambda_sd"] = CFG["hetero"]["lambda_sd"]; d["tau_sig_sd"] = CFG["hetero"]["tau_sig_sd"]
    Zs = seasonal_modifiers(sites, mc["modifiers"])
    d.update(P_s=int(Zs.shape[1]), Zs=Zs.tolist(), n_hmod=int(mc["n_hmod"] if Zs.shape[1] else 0), seas_mod_sd=CFG["seasonal_modifier_sd"],
             hetero_sigma=int(mc["hetero"]), use_shock=int(mc.get("shock", 0)), sigma_e_sd=CFG["shock_sd"], shock_centre=int(mc.get("shock_centre", 1)))
    hi = hier_indices(sites)
    d.update(use_hier=int(mc["hier"]), n_A=int(hi["wma"].max()), n_C=int(hi["mancat"].max()), n_B=int(hi["wb"].max()),
             wma=hi["wma"].tolist(), mancat=hi["mancat"].tolist(), wb=hi["wb"].tolist(), wb_marginal=int(mc.get("wb_marginal", 1)))
    pri = (priors or A.load_priors())["priors"]
    d.update(sigma_A_sd=pri["sigma_wma"]["sigma"], sigma_C_sd=pri["sigma_mancat"]["sigma"], sigma_B_sd=pri["sigma_waterbody"]["sigma"],
             sigma_w_sd=pri["sigma_field"]["sigma"], rho_a=pri["rho_field_km"]["alpha"], rho_b=pri["rho_field_km"]["beta"])
    if mc["field"]:
        PhiE, slE, PhiN, slN = hsgp_basis(sites.x_km.to_numpy(), sites.y_km.to_numpy())
    else:
        PhiE, slE, PhiN, slN = np.zeros((len(sites), 1)), np.ones(1), np.zeros((len(sites), 1)), np.ones(1)
    d.update(use_field=int(mc["field"]), m1=int(PhiE.shape[1]), m2=int(PhiN.shape[1]), PhiE=PhiE.tolist(), PhiN=PhiN.tolist(),
             sqrt_lamE=slE.tolist(), sqrt_lamN=slN.tolist())
    return d


# ------------------------------------------------------------------ fitting
SCALARS = ["alpha", "gamma", "sigma_u", "sigma_om", "sigma"]
TOP_CANDIDATES = ["alpha", "beta", "gamma", "harm", "delta_bar", "tau_delta", "tau_s", "Cs", "sigma_e", "sigma_u", "sigma_om", "sigma", "lambda", "tau_sig",
                  "sigma_A", "sigma_C", "sigma_B", "sigma_w", "rho"]


def run_fit(tag, data, mc, iter_warmup=None, iter_sampling=None, chains=None, adapt_delta=None, max_treedepth=None, inits=None, seed=None,
            keep_csv=False, save_draws_every=1, extra_diag=None, refresh=None):
    """Sample, write CSVs to scratch, compute convergence diagnostics, save draws (npz, scratch) and a summary (project)."""
    import cmdstanpy, arviz as az
    s = CFG["sampling"]
    chains = chains or s["chains"]
    outdir = scratch_dir() / f"fit_{tag}"
    if outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True)
    model = get_model()
    r0 = resource.getrusage(resource.RUSAGE_CHILDREN); t0 = time.time()
    fit = model.sample(data=data, chains=chains, parallel_chains=chains, iter_warmup=iter_warmup or s["iter_warmup"],
                       iter_sampling=iter_sampling or s["iter_sampling"], adapt_delta=adapt_delta or s["adapt_delta"],
                       max_treedepth=max_treedepth or s["max_treedepth"], seed=seed or CFG["seeds"]["stan"], output_dir=str(outdir),
                       show_progress=False, inits=inits, sig_figs=9)
    wall = time.time() - t0; r1 = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu = (r1.ru_utime - r0.ru_utime) + (r1.ru_stime - r0.ru_stime)
    mv = fit.method_variables()
    idata = az.from_cmdstanpy(posterior=fit)
    post = idata.posterior
    top = [v for v in TOP_CANDIDATES if v in post and post[v].size > 0]
    summ = az.summary(idata, var_names=top, round_to=6)
    lat_vars = [v for v in ["theta", "z_om", "z_sig", "z_B", "vB", "z_C", "z_A", "w_site", "z_e"] if v in post and post[v].size > 0]
    lat = az.summary(idata, var_names=lat_vars, kind="diagnostics")
    res = dict(tag=tag, model_config=mc, wall_s=wall, cpu_s=cpu, cpu_h=cpu / 3600, peak_child_rss_mb=r1.ru_maxrss / 1e6, chains=chains,
               iter_warmup=iter_warmup or s["iter_warmup"], iter_sampling=iter_sampling or s["iter_sampling"],
               adapt_delta=adapt_delta or s["adapt_delta"], seed=seed or CFG["seeds"]["stan"], I=data["I"], J=data["J"], N=data["N"],
               divergences=int(mv["divergent__"].sum()), divergences_by_chain=np.asarray(mv["divergent__"]).sum(axis=0).astype(int).tolist(),
               treedepth_hits=int((mv["treedepth__"] >= (max_treedepth or s["max_treedepth"])).sum()),
               mean_treedepth=float(mv["treedepth__"].mean()), mean_leapfrog=float(mv["n_leapfrog__"].mean()),
               stepsize_by_chain=np.atleast_2d(mv["stepsize__"])[0].round(5).tolist(), ebfmi=[float(x) for x in az.bfmi(idata)],
               top_max_rhat=float(summ["r_hat"].max()), top_argmax_rhat=str(summ["r_hat"].idxmax()),
               top_min_ess_bulk=float(summ["ess_bulk"].min()), top_argmin_ess_bulk=str(summ["ess_bulk"].idxmin()),
               top_min_ess_tail=float(summ["ess_tail"].min()),
               latent_max_rhat=float(lat["r_hat"].max()), latent_min_ess_bulk=float(lat["ess_bulk"].min()), latent_min_ess_tail=float(lat["ess_tail"].min()),
               gradient_evals_sampling=int(mv["n_leapfrog__"].sum()))
    res["ms_per_gradient_approx"] = 1000 * wall * chains / max(1, res["gradient_evals_sampling"] * ((iter_warmup or s["iter_warmup"]) + (iter_sampling or s["iter_sampling"])) / (iter_sampling or s["iter_sampling"]))
    if extra_diag:
        res.update(extra_diag)
    cw = {}
    for v in ["alpha", "delta_bar", "sigma_u", "sigma_om", "sigma", "sigma_w", "rho", "sigma_B", "sigma_C", "sigma_A"]:
        if v in post and post[v].size > 0:
            cw[v] = post[v].values.reshape(post.sizes["chain"], post.sizes["draw"], -1)[..., 0].mean(axis=1).round(4).tolist()
    res["chain_means"] = cw
    summ.to_csv(ROOT / "tables" / f"{tag}_posterior_summary.csv")
    draws = {}
    for v in list(post.data_vars):
        arr = post[v].values
        draws[v] = arr.reshape(arr.shape[0] * arr.shape[1], *arr.shape[2:])[::save_draws_every]
    for v in ["divergent__", "treedepth__", "n_leapfrog__", "lp__"]:
        draws[v] = np.asarray(mv[v]).T.reshape(-1)[::save_draws_every] if np.ndim(mv[v]) == 2 else np.asarray(mv[v]).reshape(-1)[::save_draws_every]
    np.savez(scratch_dir() / f"{tag}_draws.npz", **draws)
    A.save_json(res, ROOT / "diagnostics" / f"{tag}_sampling_diagnostics.json")
    if not keep_csv:
        shutil.rmtree(outdir, ignore_errors=True)
    return res, draws, idata


def load_draws(tag):
    D = dict(np.load(scratch_dir() / f"{tag}_draws.npz"))
    return D


def from_p3a(tag):
    """Adapter: Phase 3A M0/M0-H draws (phase_03a/data_processed/<tag>_draws.npz) -> Phase 3B draw dictionary."""
    meas, sites, sy = load_tables()
    P = np.load(P3A / "data_processed" / f"{tag}_draws.npz")
    X = A.site_design(sites); ms = sites.ms.to_numpy(); reg = sites.region_idx.to_numpy() - 1
    sm = P["alpha"][:, None] + P["beta"] @ X.T + ms[None, :] * P["delta_r"][:, reg]
    D = dict(alpha=P["alpha"], beta=P["beta"], gamma=P["gamma"], harm=P["harm"], delta_bar=P["delta_r"][:, :1], sigma_u=P["sigma_u"],
             sigma_om=P["sigma_om"], sigma=P["sigma"], theta=sm + P["u_site"].astype(float), z_om=P["omega"].astype(float) / P["sigma_om"][:, None],
             mu_chk=P["mu_chk"])
    if "lambda" in P.files:
        D["lambda"] = P["lambda"]; D["tau_sig"] = P["tau_sig"]; D["_sig_site"] = P["sig_site"].astype(float)
    return D


# ------------------------------------------------------------------ posterior reconstruction
def ndraws(D):
    return len(D["alpha"])


def get1(D, k):
    v = D.get(k)
    if v is None or np.size(v) == 0:
        return None
    v = np.asarray(v, dtype=float)
    return v.reshape(len(v), -1)[:, 0] if v.ndim > 1 and v.shape[1] == 1 else v


def site_harm(D, sites, mc, ms_override=None):
    """(S, I, 4) site harmonic coefficients c_i; ms_override in {None, 0, 1} sets the ms modifier for prediction."""
    S = ndraws(D); I = len(sites)
    C = np.repeat(np.asarray(D["harm"], float)[:, None, :], I, axis=1)
    reg = sites.region_idx.to_numpy() - 1
    if mc["regional_season"]:
        dev = get1(D, "tau_s")[:, None, None] * np.asarray(D["z_seas"], float)       # (S,7,2)
        C[:, :, 0] += dev[:, reg, 0]; C[:, :, 1] += dev[:, reg, 1]
    if mc["modifiers"]:
        s2 = sites.copy()
        if ms_override is not None:
            s2["ms"] = ms_override
        Zs = seasonal_modifiers(s2, mc["modifiers"])
        C[:, :, :mc["n_hmod"]] += np.einsum("ip,sph->sih", Zs, np.asarray(D["Cs"], float))
    return C


def log_seasonal_factor(C, Q=24):
    """log (1/Q) sum_q exp(Hq . c_i) for C (S, I, 4) -> (S, I)."""
    Hq = A.harmonic_basis(Q)
    s = np.einsum("sih,qh->siq", C, Hq)
    m = s.max(axis=2, keepdims=True)
    return m[..., 0] + np.log(np.exp(s - m).mean(axis=2))


def sig_site(D, sites, mc):
    if "_sig_site" in D:
        return D["_sig_site"]
    sg = get1(D, "sigma")
    if not mc["hetero"]:
        return np.repeat(sg[:, None], len(sites), axis=1)
    return sg[:, None] * np.exp(get1(D, "lambda")[:, None] * sites.z_logarea.to_numpy()[None, :] + get1(D, "tau_sig")[:, None] * np.asarray(D["z_sig"], float))


def site_components(D, sites, mc):
    """Return dict of (S, I) site-level terms: fixed (alpha + X beta), net (delta MS), vA, vC, vB, w, u (= theta - m)."""
    X = A.site_design(sites); ms = sites.ms.to_numpy(); reg = sites.region_idx.to_numpy() - 1
    S = ndraws(D); I = len(sites)
    out = dict(fixed=np.asarray(D["alpha"], float)[:, None] + np.asarray(D["beta"], float) @ X.T)
    if mc["network_mode"] == 1:
        dr = np.repeat(get1(D, "delta_bar")[:, None], 7, axis=1)
    elif mc["network_mode"] == 2:
        dr = get1(D, "delta_bar")[:, None] + get1(D, "tau_delta")[:, None] * np.asarray(D["z_delta"], float)
    else:
        dr = np.zeros((S, 7))
    out["delta_r"] = dr
    out["net"] = ms[None, :] * dr[:, reg]
    zero = np.zeros((S, I))
    if mc["hier"]:
        hi = hier_indices(sites)
        out["vA"] = get1(D, "sigma_A")[:, None] * np.asarray(D["z_A"], float)[:, hi["wma"] - 1]
        out["vC"] = get1(D, "sigma_C")[:, None] * np.asarray(D["z_C"], float)[:, hi["mancat"] - 1]
        if "vB" in D and np.size(D["vB"]):
            out["vB"] = np.asarray(D["vB"], float)[:, hi["wb"] - 1]          # exact conditional draws (wb_marginal)
        elif "z_B" not in D or not np.size(D["z_B"]):
            out["vB"] = zero                                                  # marginalised and not drawn: m excludes vB
        else:
            out["vB"] = get1(D, "sigma_B")[:, None] * np.asarray(D["z_B"], float)[:, hi["wb"] - 1]
    else:
        out["vA"] = out["vC"] = out["vB"] = zero
    if mc["field"]:
        out["w"] = np.asarray(D["w_site"], float) if "w_site" in D and np.size(D["w_site"]) else None
        if out["w"] is None:
            PhiE, slE, PhiN, slN = hsgp_basis(sites.x_km, sites.y_km)
            out["w"] = hsgp_field(PhiE, PhiN, slE, slN, np.asarray(D["z_w"], float), get1(D, "sigma_w"), get1(D, "rho"))
    else:
        out["w"] = zero
    m = out["fixed"] + out["net"] + out["vA"] + out["vC"] + out["vB"] + out["w"]
    out["m"] = m
    out["u"] = np.asarray(D["theta"], float) - m
    return out


def eta_obs(D, meas, sites, sy, mc, draws=None):
    """(S', N) latent log-median at each measurement."""
    idx = np.arange(ndraws(D)) if draws is None else np.asarray(draws)
    th = np.asarray(D["theta"], float)[idx]
    om = get1(D, "sigma_om")[idx, None] * np.asarray(D["z_om"], float)[idx]
    ga = get1(D, "gamma")[idx]
    yr = (sy.year_idx.to_numpy() == 2).astype(float); sidx = sy.site_idx.to_numpy() - 1
    sy_lp = th[:, sidx] + ga[:, None] * yr[None, :] + om
    sj = meas.siteyear_idx.to_numpy() - 1; si = meas.site_idx.to_numpy() - 1
    tau = meas.tau.to_numpy()
    Hn = np.column_stack([np.cos(2 * np.pi * tau), np.sin(2 * np.pi * tau), np.cos(4 * np.pi * tau), np.sin(4 * np.pi * tau)])
    Dsub = {k: (np.asarray(v)[idx] if (hasattr(v, "__len__") and np.ndim(v) >= 1 and len(v) == ndraws(D)) else v) for k, v in D.items()}
    C = site_harm(Dsub, sites, mc)                                   # (S',I,4)
    seas = np.einsum("nh,snh->sn", Hn, C[:, si, :])
    if mc.get("shock", 0):
        seas = seas + shock_values(Dsub)[:, obs_rm_index(meas, sites)]
    return sy_lp[:, sj] + seas


def shock_values(D):
    """(S, 168) regional month-by-year effects e = sigma_e z_e, index (region-1)*24 + (year-1)*12 + (month-1)."""
    if "e_rm" in D and np.size(D["e_rm"]):
        return np.asarray(D["e_rm"], float)                       # Stan transformed parameter (centred form when shock_centre = 1)
    return get1(D, "sigma_e")[:, None] * np.asarray(D["z_e"], float)   # Stage 3 S5 fit (uncentred, no e_rm saved)


def month_block(tau):
    return np.minimum(12, 1 + np.floor(12 * np.asarray(tau)).astype(int))


def obs_rm_index(meas, sites):
    reg = sites.region_idx.to_numpy()[meas.site_idx.to_numpy() - 1]
    return (reg - 1) * 24 + (meas.year.to_numpy() - 2023) * 12 + month_block(meas.tau.to_numpy()) - 1


def log_sf_siteyear(D, sites, sy, mc, ms_override=None, Q=24):
    """(S, J) log of the annual seasonal factor (1/Q) sum_q exp(H(tau_q).c_i + e_{r(i),t,m(q)}) for every site-year."""
    C = site_harm(D, sites, mc, ms_override=ms_override)              # (S,I,4)
    sidx = sy.site_idx.to_numpy() - 1
    Hq = A.harmonic_basis(Q)
    s = np.einsum("sih,qh->siq", C, Hq)[:, sidx, :]                    # (S,J,Q)
    if mc.get("shock", 0):
        tq = (np.arange(1, Q + 1) - 0.5) / Q; mq = month_block(tq)
        reg = sites.region_idx.to_numpy()[sidx]; yr = sy.year_idx.to_numpy()
        idx = ((reg - 1) * 24 + (yr - 1) * 12)[:, None] + (mq - 1)[None, :]   # (J,Q)
        s = s + shock_values(D)[:, idx]
    m = s.max(axis=2, keepdims=True)
    return m[..., 0] + np.log(np.exp(s - m).mean(axis=2))


def pointwise_loglik(D, meas, sites, sy, mc, chunk=250):
    """(S, N) pointwise log-likelihood (normal lpdf of log y; censored: log Phi). The Jacobian 1/y is omitted (constant)."""
    S = ndraws(D); N = len(meas)
    out = np.empty((S, N), dtype=np.float64)
    cen = meas.censored.to_numpy(); ly = meas.log_y.to_numpy(); ll_lim = meas.log_limit.to_numpy()
    si = meas.site_idx.to_numpy() - 1
    sgs = sig_site(D, sites, mc)
    for a in range(0, S, chunk):
        idx = np.arange(a, min(S, a + chunk))
        eta = eta_obs(D, meas, sites, sy, mc, draws=idx)
        sd = sgs[idx][:, si]
        z = (np.where(cen, ll_lim, ly)[None, :] - eta) / sd
        out[idx] = np.where(cen[None, :], log_ndtr(z), -0.5 * z ** 2 - np.log(sd) - 0.5 * np.log(2 * np.pi))
    return out


# ------------------------------------------------------------------ standardised contrast (MS-§7 + Addendum C1)
def delta_T(D, sites, sy, mc, n_gh=15, include_location=True, return_pi=False):
    """P^R_T, P^M_T, Delta_T over the frozen target T. For each T unit (i, t): log level under network k
    L_k = alpha + x_i beta + delta_r(i) 1{k=M} + [vA + vC + vB + w(s_i) at the unit's own levels/location]
          + gamma_t + log SF_i^(k) + sigma_new^2/2,
    with a NEW site effect u and site-year effect omega marginalised exactly (Gaussian), and sigma_new integrated by
    Gauss-Hermite (hetero). Seasonal factor SF depends on k only if the network modifies seasonality."""
    S = ndraws(D)
    comp = site_components(D, sites, mc)
    Tm = sy.in_target_T.to_numpy() == 1
    sidx = sy.site_idx.to_numpy()[Tm] - 1; yrt = (sy.year_idx.to_numpy()[Tm] == 2).astype(float)
    w = sy.target_weight.to_numpy()[Tm]; w = w / w.sum()
    loc = (comp["vA"] + comp["vC"] + comp["vB"] + comp["w"]) if include_location else 0 * comp["fixed"]
    base = comp["fixed"][:, sidx] + loc[:, sidx] + get1(D, "gamma")[:, None] * yrt[None, :]
    dT = comp["delta_r"][:, sites.region_idx.to_numpy()[sidx] - 1]
    lsfR = log_sf_siteyear(D, sites, sy, mc, ms_override=0 if "ms" in mc["modifiers"] else None)[:, Tm]
    lsfM = log_sf_siteyear(D, sites, sy, mc, ms_override=1 if "ms" in mc["modifiers"] else None)[:, Tm]
    sd_re = np.sqrt(get1(D, "sigma_u") ** 2 + get1(D, "sigma_om") ** 2)[:, None]
    sg = get1(D, "sigma")
    if mc["hetero"]:
        ghx, ghw = A.gh_nodes(n_gh); lam = get1(D, "lambda"); tsg = get1(D, "tau_sig"); zt = sites.z_logarea.to_numpy()[sidx]
        nodes = [(wg, sg[:, None] * np.exp(lam[:, None] * zt[None, :] + tsg[:, None] * xg)) for xg, wg in zip(ghx, ghw)]
    else:
        nodes = [(1.0, np.repeat(sg[:, None], len(sidx), axis=1))]
    res = {}
    for c in THRESH:
        pR = np.zeros_like(base); pM = np.zeros_like(base)
        for wg, sn in nodes:
            h = 0.5 * sn ** 2
            pR += wg * ndtr((base + lsfR + h - np.log(c)) / sd_re)
            pM += wg * ndtr((base + lsfM + dT + h - np.log(c)) / sd_re)
        PR, PM = pR @ w, pM @ w
        res[c] = dict(PR=PR, PM=PM, D=PM - PR, RR=PM / PR)
        if return_pi:
            res[c].update(pR=pR, pM=pM)
    return res


def summarise_delta(res):
    rows = []
    for c, e in res.items():
        for nm in ["PR", "PM", "D", "RR"]:
            x = e[nm]
            rows.append(dict(threshold=c, quantity={"PR": "P^R_T", "PM": "P^M_T", "D": "Delta_T", "RR": "RR_T"}[nm], median=float(np.median(x)),
                             mean=float(np.mean(x)), sd=float(np.std(x)), lo95=float(np.quantile(x, .025)), hi95=float(np.quantile(x, .975)),
                             p_gt0=float((x > 0).mean()) if nm == "D" else None))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ residual spatial / flow statistics (P3A_06 §4)
class PairSets:
    def __init__(self, sites):
        xy = sites[["x_km", "y_km"]].to_numpy(); iu = np.triu_indices(len(sites), 1)
        d = np.sqrt(((xy[iu[0]] - xy[iu[1]]) ** 2).sum(1))
        near = d < 50
        self.a, self.b, self.d = iu[0][near], iu[1][near], d[near]
        pairs = pd.read_csv(P2B / "diagnostics/flow_connected_pairs.csv")
        sid = dict(zip(sites.site, range(len(sites))))
        conn = {}
        for x, y_, nk in zip(pairs.up_site, pairs.down_site, pairs.network_km):
            if x in sid and y_ in sid:
                i, j = sorted((sid[x], sid[y_])); conn[(i, j)] = nk
        self.fc = np.array([(i, j) in conn for i, j in zip(self.a, self.b)])
        self.net_km = np.array([conn.get((i, j), np.nan) for i, j in zip(self.a, self.b)])
        wb = sites.wb.to_numpy()
        self.same_wb = np.array([isinstance(wb[i], str) and wb[i] == wb[j] for i, j in zip(self.a, self.b)])
        self.k25 = (self.d < 25) & ~self.same_wb
        self.n_fc = int((self.k25 & self.fc).sum()); self.n_nc = int((self.k25 & ~self.fc).sum())

    def stats(self, r_draws, bands=((0, 5), (5, 10), (10, 20), (20, 40))):
        """r_draws (S, I). Returns per-draw correlogram and flow excess (connected - non-connected)."""
        out = {f"{lo}-{hi}": [] for lo, hi in bands}; exc, fcm, ncm = [], [], []
        for r in r_draws:
            z = (r - r.mean()) / r.std(); pr = z[self.a] * z[self.b]
            for lo, hi in bands:
                out[f"{lo}-{hi}"].append(pr[(self.d >= lo) & (self.d < hi)].mean())
            f_ = pr[self.k25 & self.fc].mean(); n_ = pr[self.k25 & ~self.fc].mean()
            exc.append(f_ - n_); fcm.append(f_); ncm.append(n_)
        return {k: np.array(v) for k, v in out.items()}, np.array(exc), np.array(fcm), np.array(ncm)


def band(x, lo=.05, hi=.95):
    x = np.asarray(x)
    return [float(np.quantile(x, lo)), float(np.median(x)), float(np.quantile(x, hi))]


save_json = A.save_json
