"""Shared library for Track A Phase 3A: data loading, Stan data construction, annual arithmetic mean and
standardised-estimand computations. Definitions follow phase_02b/specifications/model_specification.md (MS-§)."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
TA = ROOT.parent
P2B = TA / "phase_02b"
DATA = ROOT / "data_processed"
STAN_FILE = ROOT / "stan" / "track_a_measurement.stan"
THRESH = (0.05, 0.10, 0.20)
Q_DEFAULT = 24
REGIONS = ["Anglian", "Midlands", "NorthEast", "NorthWest", "SouthWest", "Southern", "Thames"]
X_COLS = ["z_logarea", "z_logarea2", "z_edm", "geo_Siliceous", "geo_Organic", "geo_Unknown"]
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


def get_model():
    """Compile (if needed) and return the CmdStanModel. On macOS an executable overwritten in place at the same path
    can be killed by the kernel (stale code-signature cache, return code -9), so a stale executable is DELETED first."""
    import cmdstanpy
    exe = STAN_FILE.with_suffix("")
    if exe.exists() and exe.stat().st_mtime < STAN_FILE.stat().st_mtime:
        exe.unlink()
    return cmdstanpy.CmdStanModel(stan_file=str(STAN_FILE))


def load_priors(apply_overrides=True):
    """Frozen Phase 2B priors, with the documented Phase 3A overrides (config/prior_overrides.yaml) applied."""
    pri = yaml.safe_load(open(P2B / "specifications/priors_and_settings.yaml"))
    ov = ROOT / "config" / "prior_overrides.yaml"
    if apply_overrides and ov.exists():
        for k, v in (yaml.safe_load(open(ov)) or {}).items():
            pri["priors"][k].update(v)
    return pri


def load_tables():
    meas = pd.read_parquet(DATA / "measurements.parquet")
    sites = pd.read_parquet(DATA / "sites.parquet")
    sy = pd.read_parquet(DATA / "siteyears.parquet")
    return meas, sites, sy


def site_design(sites: pd.DataFrame) -> np.ndarray:
    z = sites.z_logarea.to_numpy()
    X = np.column_stack([z, z ** 2, sites.z_edm.to_numpy(),
                         (sites.geology_idx == 2).astype(float), (sites.geology_idx == 3).astype(float), (sites.geology_idx == 4).astype(float)])
    return X


def prior_constants(pri=None):
    pri = pri or load_priors()
    p = pri["priors"]
    return dict(alpha_mu=p["alpha"]["mu"], alpha_sd=p["alpha"]["sigma"],
                beta_sd=[p["beta_logarea"]["sigma"], p["beta_logarea_sq"]["sigma"], p["beta_edm"]["sigma"]] + [p["beta_geology"]["sigma"]] * 3,
                gamma_sd=p["gamma_2024"]["sigma"], harm_sd=p["harmonics_national"]["sigma"], tau_s_sd=p["tau_season_region"]["sigma"],
                delta_sd=p["delta_bar"]["sigma"], tau_delta_sd=p["tau_delta"]["sigma"], sigma_u_sd=p["sigma_site"]["sigma"],
                sigma_om_sd=p["sigma_siteyear"]["sigma"], sigma_sd=p["sigma_resid"]["sigma"])


def build_stan_data(meas, sites, sy, network_mode=1, regional_season=0, prior_only=0, use_interval=0, J_chk=50, Q=Q_DEFAULT, priors=None, centred_site=1, centred_siteyear=0, hetero_sigma=0):
    """Indices must be 1-based contiguous in the supplied tables (re-indexed subsets are handled by reindex())."""
    unc = ~meas.censored.to_numpy(); cen = meas.censored.to_numpy()
    y = meas.value.to_numpy(); h = meas.half_width.to_numpy()
    lo = np.where(y - h > 0, np.log(np.maximum(y - h, 1e-300)), -50.0)
    d = dict(N=len(meas), N_unc=int(unc.sum()), N_cens=int(cen.sum()),
             unc_idx=(np.flatnonzero(unc) + 1).tolist(), cens_idx=(np.flatnonzero(cen) + 1).tolist(),
             log_y=meas.log_y.to_numpy()[unc].tolist(), log_L=meas.log_limit.to_numpy()[cen].tolist(),
             use_interval=int(use_interval), log_lo=lo[unc].tolist(), log_hi=np.log(y + h)[unc].tolist(),
             I=len(sites), J=len(sy), sy=meas.siteyear_idx.astype(int).tolist(),
             sy_site=sy.site_idx.astype(int).tolist(), sy_year=sy.year_idx.astype(int).tolist(),
             tau=meas.tau.to_numpy().tolist(), K=6, X=site_design(sites).tolist(), ms=sites.ms.astype(float).tolist(),
             region=sites.region_idx.astype(int).tolist(), network_mode=int(network_mode), regional_season=int(regional_season),
             prior_only=int(prior_only), centred_site=int(centred_site), centred_siteyear=int(centred_siteyear), hetero_sigma=int(hetero_sigma), lambda_sd=0.25, tau_sig_sd=0.5, J_chk=int(min(J_chk, len(sy))), Q=int(Q))
    d.update(prior_constants(priors))
    return d


def reindex(meas, sites, sy):
    """Re-index a subset to contiguous 1-based indices (sites -> site-years -> measurements)."""
    sites = sites.sort_values("site_idx").reset_index(drop=True).copy()
    smap = dict(zip(sites.site_idx, range(1, len(sites) + 1)))
    sy = sy[sy.site_idx.isin(smap)].sort_values("siteyear_idx").reset_index(drop=True).copy()
    ymap = dict(zip(sy.siteyear_idx, range(1, len(sy) + 1)))
    meas = meas[meas.siteyear_idx.isin(ymap)].sort_values("obs_idx").reset_index(drop=True).copy()
    sites["site_idx"] = sites.site_idx.map(smap); sy["site_idx"] = sy.site_idx.map(smap); sy["siteyear_idx"] = sy.siteyear_idx.map(ymap)
    meas["site_idx"] = meas.site_idx.map(smap); meas["siteyear_idx"] = meas.siteyear_idx.map(ymap)
    return meas, sites, sy


# ---------------------------------------------------------------- annual arithmetic mean (MS-§3)
def harmonic_basis(Q=Q_DEFAULT):
    t = (np.arange(1, Q + 1) - 0.5) / Q
    return np.column_stack([np.cos(2 * np.pi * t), np.sin(2 * np.pi * t), np.cos(4 * np.pi * t), np.sin(4 * np.pi * t)])


def annual_mean(base, harm, sigma, Q=Q_DEFAULT, short_term=True):
    """mu = (1/Q) sum_q exp(base + s(tau_q) + sigma^2/2).
    base: (..., ) array of draw-specific log levels; harm: (S,4) per-draw harmonic coefficients broadcast on axis 0;
    sigma: (S,). Shapes: base (S, M) -> returns (S, M)."""
    Hq = harmonic_basis(Q)                       # (Q,4)
    seas = harm @ Hq.T                           # (S,Q)
    # log-sum-exp over q for numerical stability
    a = base[:, :, None] + seas[:, None, :]      # (S,M,Q)
    m = a.max(axis=2, keepdims=True)
    lme = (m[..., 0] + np.log(np.exp(a - m).mean(axis=2)))
    if short_term:
        lme = lme + 0.5 * sigma[:, None] ** 2
    return np.exp(lme)


def log_annual_mean_seasonal_factor(harm, Q=Q_DEFAULT):
    """log of (1/Q) sum_q exp(s(tau_q)) per draw (S,), since base enters additively in the exponent."""
    Hq = harmonic_basis(Q); seas = harm @ Hq.T
    m = seas.max(axis=1, keepdims=True)
    return (m[:, 0] + np.log(np.exp(seas - m).mean(axis=1)))


def gh_nodes(n=7):
    """Gauss-Hermite nodes/weights for E[f(Z)], Z ~ N(0,1)."""
    x, w = np.polynomial.hermite.hermgauss(n)
    return x * np.sqrt(2.0), w / np.sqrt(np.pi)


def pi_exceed(loglevel_wo_re, sigma_u, sigma_om, c, n_gh=7):
    """pi = Pr(mu > c) marginal over new site effect u ~ N(0,sigma_u^2) and site-year effect om ~ N(0,sigma_om^2).
    log mu = loglevel_wo_re + u + om, so Pr(mu > c) = Phi((loglevel - log c) / sqrt(sigma_u^2 + sigma_om^2)) exactly
    (sum of independent normals); Gauss-Hermite version kept for verification of the general (non-Gaussian) case."""
    from scipy.special import ndtr
    s = np.sqrt(sigma_u ** 2 + sigma_om ** 2)
    return ndtr((loglevel_wo_re - np.log(c)) / s[:, None])


def pi_exceed_mc(loglevel_wo_re, sigma_u, sigma_om, c, n=200_000, seed=20261114):
    """Monte Carlo verification of pi_exceed for small inputs: draws u, om explicitly."""
    rng = np.random.default_rng(seed)
    zu, zo = rng.standard_normal(n), rng.standard_normal(n)
    out = np.empty_like(loglevel_wo_re)
    for s in range(loglevel_wo_re.shape[0]):
        lm = loglevel_wo_re[s][:, None] + sigma_u[s] * zu[None, :] + sigma_om[s] * zo[None, :]
        out[s] = (lm > np.log(c)).mean(axis=1)
    return out


def save_json(obj, path):
    def conv(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=conv)
