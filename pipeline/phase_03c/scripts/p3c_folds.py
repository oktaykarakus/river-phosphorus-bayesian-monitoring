"""Phase 3C held-out validation: fold definitions, leakage-free training data and held-out predictive scores (P3C_05).

Folds (pre-specified in config/phase3c_settings.yaml before any fold fit):
  W1, W2  water-body-blocked: all sites of a random 20% of water bodies (5-way partition stratified by region; groups 1, 2)
          -> tests prediction of NEW water bodies inside monitored river networks (what the T1 statistic measures).
  C1      management-catchment-blocked: the Phase 3B cat80 split (seed 20261120) -> tests prediction in UNMONITORED catchments.

Leakage control. Held-out sites remain in the Stan model as nodes WITHOUT measurements: all sites, site-years, water-body,
catchment, field and river-network nodes are kept, only the held-out measurements are removed, so every held-out latent is
drawn from its conditional distribution given the TRAINING data. Covariates (incl. the C1 SD covariate z_logarea) are not
outcomes. Standardisation constants are frozen Phase 2B constants.

Scores (per held-out site, Rao-Blackwellised per posterior draw):
  theta_i | rest ~ N(m_i + mu_v + mu_e, s_v^2 + s_e^2), where mu_e/s_e is the water-body conditional given the other sites
  of the water body and mu_v/s_v the tail-up full conditional given the other river-network nodes (Markov blanket);
  omega_it ~ N(0, sigma_om^2) per site-year; log sigma_i = log sigma + lambda z_i + tau_sig xi, xi ~ N(0,1).
  site-joint log predictive density: adaptive Gauss-Hermite over theta, GH over xi and over omega per year (censoring-aware);
  pointwise measurement log predictive density and PIT: closed form in theta/omega, GH over xi;
  site-year sample-mean exceedance (observed: Phase 2 mean_hi; replicate: measurements at the observed dates, censored at the
  observation's own limit or the precision rule A as in P3B_05): Brier and log scores, 90% interval coverage.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import log_ndtr, ndtr, logsumexp

import p3c_lib as L

B = L.B; A = L.A


# ====================================================================== folds
def heldout_sites(fold: str, sites: pd.DataFrame) -> list:
    cfg = L.CFG["validation"]["folds"][fold]
    if cfg["type"] == "waterbody_block":
        rng = np.random.default_rng(L.CFG["seeds"]["cv_folds"])
        wb = sites.wb.astype(object).where(sites.wb.notna(), "__new_" + sites.site).to_numpy()
        df = pd.DataFrame({"site": sites.site, "wb": wb, "region": sites.region})
        first_reg = df.groupby("wb").region.first()
        grp = {}
        for reg in sorted(first_reg.unique()):
            lv = sorted(first_reg[first_reg == reg].index)
            perm = rng.permutation(len(lv))
            for rank, k in enumerate(perm):
                grp[lv[k]] = rank % cfg["partition_k"] + 1
        return sorted(df.site[df.wb.map(grp) == cfg["group"]])
    if cfg["type"] == "mancat_block":
        rng = np.random.default_rng(L.CFG["seeds"]["subsets"])            # identical to phase_03b/scripts/p3b_fit.py cat80
        mc = sites.mancat.fillna("__none").unique(); rng.shuffle(mc)
        held, n = [], 0
        for c in mc:
            if n >= 0.2 * len(sites):
                break
            held.append(c); n += int((sites.mancat.fillna("__none") == c).sum())
        return sorted(sites.site[sites.mancat.fillna("__none").isin(held)])
    raise ValueError(fold)


def training_measurements(meas: pd.DataFrame, held: list) -> pd.DataFrame:
    return meas[~meas.site.isin(held)].copy()


# ====================================================================== conditional distribution of held-out site levels
def tu_conditional(rn, v, sigma, rho, i):
    """Full conditional N(mu, s^2) of v_i given all other TU node values, per draw. v: (S, I); sigma, rho: (S,)."""
    ptr, par, h, spw, ch = rn["tu_ptr"], rn["tu_par"], rn["tu_h"], np.sqrt(rn["tu_pw"]), rn["tu_child"]
    S = v.shape[0]

    def a_of(e):
        return np.exp(-h[e][None, :] / rho[:, None]) * spw[e][None, :]

    e_i = np.arange(ptr[i], ptr[i + 1])
    ai = a_of(e_i) if len(e_i) else np.zeros((S, 0))
    di = np.maximum(1 - (ai ** 2).sum(1), 1e-10)
    mu_own = (ai * v[:, par[e_i]]).sum(1) if len(e_i) else np.zeros(S)
    P = 1 / di; num = mu_own / di
    for c in np.unique(ch[par == i]):
        e_c = np.arange(ptr[c], ptr[c + 1]); ac = a_of(e_c)
        dc = np.maximum(1 - (ac ** 2).sum(1), 1e-10)
        k = par[e_c] == i
        a_ci = ac[:, k][:, 0]
        r_c = v[:, c] - (ac[:, ~k] * v[:, par[e_c][~k]]).sum(1)
        P = P + a_ci ** 2 / dc; num = num + a_ci * r_c / dc
    return num / P, sigma * np.sqrt(1 / P)


def heldout_theta_conditional(D, sites, mc, held_idx, rn=None):
    """Return (mu, sd) arrays (S, n_held) of theta_i | everything else in the draw, for held-out site indices (0-based)."""
    comp = L.site_components(D, sites, mc)
    mm = comp["m"] - comp["vB"]                     # Stan's m (excludes vB: water-body effects integrated out)
    if mc.get("tu", 0):
        v = np.asarray(D["v_tu"], float)
        mm_nov = mm - v
    else:
        mm_nov = mm
    e = np.asarray(D["theta"], float) - mm
    su2 = B.get1(D, "sigma_u") ** 2; sb2 = B.get1(D, "sigma_B") ** 2
    wb = B.hier_indices(sites)["wb"]
    S = e.shape[0]
    mu = np.zeros((S, len(held_idx))); sd = np.zeros((S, len(held_idx)))
    for k, i in enumerate(held_idx):
        others = np.flatnonzero((wb == wb[i]) & (np.arange(len(wb)) != i))
        n_o = len(others)
        se = e[:, others].sum(1) if n_o else np.zeros(S)
        mu_e = sb2 * se / (su2 + n_o * sb2)
        var_e = su2 + sb2 * su2 / (su2 + n_o * sb2)
        m_i = mm_nov[:, i] + mu_e
        var = var_e
        if mc.get("tu", 0):
            mv, sv = tu_conditional(rn, v, B.get1(D, "sigma_tu"), B.get1(D, "rho_tu"), i)
            m_i = m_i + mv; var = var + sv ** 2
        mu[:, k] = m_i; sd[:, k] = np.sqrt(var)
    return mu, sd


# ====================================================================== scores
def site_joint_logpred(m_mu, m_sd, s_om, ls, tsig, y, cen, lim, seas, yr, n_theta=24, n_om=13, n_sig=11):
    """log p(y_site | draw) integrating theta ~ N(m_mu, m_sd^2), omega_t ~ N(0, s_om^2) per site-year t (yr: 0..T-1),
    log sigma = ls + tsig * xi. Arrays over draws: m_mu, m_sd, s_om, ls, tsig (S,) or scalars; seas (S, n) or (n,).
    y, lim: log value / log limit (n,); cen (n,) bool. Returns (S,) or scalar."""
    scalar = np.ndim(m_mu) == 0
    m_mu, m_sd, s_om, ls, tsig = (np.atleast_1d(np.asarray(x, float)) for x in (m_mu, m_sd, s_om, ls, tsig))
    S = len(m_mu); seas = np.broadcast_to(np.atleast_2d(seas), (S, len(y)))
    xt, wt = A.gh_nodes(n_theta); xo, wo = A.gh_nodes(n_om); xs, ws = A.gh_nodes(n_sig)
    lwt, lwo, lws = np.log(wt), np.log(wo), np.log(ws)
    ycen = np.where(cen, lim - 0.5, y)
    s_eff2 = np.exp(2 * ls) + s_om ** 2
    prec = 1 / m_sd ** 2 + len(y) / s_eff2
    mhat = (m_mu / m_sd ** 2 + (ycen[None, :] - seas).sum(1) / s_eff2) / prec
    shat = 1.6 / np.sqrt(prec)
    th = mhat[:, None] + shat[:, None] * xt[None, :]                                   # (S,K)
    lw_th = lwt[None, :] + (-0.5 * ((th - m_mu[:, None]) / m_sd[:, None]) ** 2 - np.log(m_sd[:, None])) \
        - (-0.5 * xt[None, :] ** 2 - np.log(shat[:, None]))
    sg = np.exp(ls[:, None] + tsig[:, None] * xs[None, :])                             # (S,Ls)
    om = s_om[:, None] * xo[None, :]                                                    # (S,Jo)
    yv = np.where(cen, lim, y)
    total = np.zeros((S, n_theta, n_sig))
    for t in np.unique(yr):
        k = yr == t
        eta = th[:, :, None, None] + om[:, None, None, :] + 0 * sg[:, None, :, None]   # (S,K,Ls,Jo)
        z = (yv[k][None, None, None, None, :] - eta[..., None] - seas[:, k][:, None, None, None, :]) / sg[:, None, :, None, None]
        ll = np.where(cen[k], log_ndtr(z), -0.5 * z ** 2 - np.log(sg)[:, None, :, None, None] - 0.5 * np.log(2 * np.pi)).sum(-1)
        total += logsumexp(ll + lwo[None, None, None, :], axis=-1)
    out = logsumexp(total + lw_th[:, :, None] + lws[None, None, :], axis=(1, 2))
    return out[0] if scalar else out


def heldout_scores(D, meas, sites, sy, mc, held, n_draws=400, seed=None, n_rep=5):
    """Held-out predictive scores for the sites in `held` from a fold fit's draws D."""
    rng = np.random.default_rng(seed or L.CFG["seeds"]["prediction_mc"])
    S0 = B.ndraws(D); pick = np.linspace(0, S0 - 1, min(n_draws, S0)).astype(int)
    Ds = {k: (np.asarray(v)[pick] if np.ndim(v) >= 1 and len(v) == S0 else v) for k, v in D.items()}
    S = len(pick)
    held_idx = np.flatnonzero(sites.site.isin(held).to_numpy())
    rn = L.rn_structures(sites) if mc.get("tu", 0) else None
    mu, sd = heldout_theta_conditional(Ds, sites, mc, held_idx, rn)
    hm = meas[meas.site.isin(held)].copy()
    # seasonal + year + shock part of eta for held-out measurements: eta_obs minus (theta + omega)
    eta_all = B.eta_obs(Ds, hm, sites, sy, mc)                                          # (S, n_h) includes theta + omega
    th = np.asarray(Ds["theta"], float); om = B.get1(Ds, "sigma_om")[:, None] * np.asarray(Ds["z_om"], float)
    seas_all = eta_all - th[:, hm.site_idx.to_numpy() - 1] - om[:, hm.siteyear_idx.to_numpy() - 1]
    s_om = B.get1(Ds, "sigma_om"); lam = B.get1(Ds, "lambda"); tsg = B.get1(Ds, "tau_sig"); lsig = np.log(B.get1(Ds, "sigma"))
    zt = sites.z_logarea.to_numpy()
    xs, ws = A.gh_nodes(11)
    site_rows, pw_rows, sy_rows = [], [], []
    lim_rep_all = np.where(hm.censored, hm.limit, np.where(hm.decimals >= 4, 0.001, 0.01))
    for k, i in enumerate(held_idx):
        kk = (hm.site_idx.to_numpy() - 1) == i
        if not kk.any():
            continue
        sub = hm[kk]; seas = seas_all[:, kk]
        y = sub.log_y.to_numpy(); cen = sub.censored.to_numpy(); lim = sub.log_limit.to_numpy()
        syi = sub.siteyear_idx.to_numpy(); yr = pd.factorize(syi)[0]
        ls = lsig + lam * zt[i]
        lp = site_joint_logpred(mu[:, k], sd[:, k], s_om, ls, tsg, y, cen, lim, seas, yr)
        site_rows.append(dict(site=sites.site.iloc[i], network=sites.network.iloc[i], region=sites.region.iloc[i], n_meas=int(kk.sum()),
                              elpd_site=float(logsumexp(lp) - np.log(S)), mc_se_proxy=float(np.std(lp))))
        # pointwise (closed form in theta + omega, GH over xi)
        mean = mu[:, k][:, None] + seas                                                  # (S,n)
        lpp = []; cdf = []
        for xg, wg in zip(xs, ws):
            sg = np.exp(ls + tsg * xg)
            v = np.sqrt(sd[:, k] ** 2 + s_om ** 2 + sg ** 2)[:, None]
            zz = (np.where(cen, lim, y)[None, :] - mean) / v
            lpp.append(np.log(wg) + np.where(cen[None, :], log_ndtr(zz), -0.5 * zz ** 2 - np.log(v) - 0.5 * np.log(2 * np.pi)))
            cdf.append(wg * ndtr(zz))
        lpp = logsumexp(np.array(lpp), axis=0); F = np.array(cdf).sum(0).mean(0)          # (S,n), (n,)
        lpd = logsumexp(lpp, axis=0) - np.log(S)
        pit = np.where(cen, rng.uniform(0, 1, len(y)) * F, F)
        for j in range(len(y)):
            pw_rows.append(dict(site=sites.site.iloc[i], network=sites.network.iloc[i], lpd=float(lpd[j]), pit=float(pit[j]), censored=bool(cen[j])))
        # site-year sample-mean exceedance (replicates at the observed dates)
        limr = lim_rep_all[kk]
        reps = []
        for _ in range(n_rep):
            thr = mu[:, k] + sd[:, k] * rng.standard_normal(S)
            omr = s_om[:, None] * rng.standard_normal((S, yr.max() + 1))
            sgr = np.exp(ls + tsg * rng.standard_normal(S))
            yrep = np.exp(thr[:, None] + omr[:, yr] + seas + sgr[:, None] * rng.standard_normal((S, len(y))))
            yrep = np.where(yrep < limr[None, :], limr[None, :], yrep)
            reps.append(np.stack([yrep[:, yr == t].mean(1) for t in range(yr.max() + 1)], axis=1))
        reps = np.concatenate(reps, axis=0)                                               # (S*n_rep, T)
        for t, sj in enumerate(pd.unique(syi)):
            obs = float(sy.mean_hi.to_numpy()[sj - 1])
            row = dict(site=sites.site.iloc[i], network=sites.network.iloc[i], siteyear_idx=int(sj), obs_mean=obs,
                       in_target_T=int(sy.in_target_T.to_numpy()[sj - 1]),
                       lo90=float(np.quantile(reps[:, t], .05)), hi90=float(np.quantile(reps[:, t], .95)))
            for c in L.THRESH:
                p = float((reps[:, t] > c).mean()); o = float(obs > c)
                pc = min(max(p, 0.5 / len(reps)), 1 - 0.5 / len(reps))
                row[f"p_{c}"] = p; row[f"obs_{c}"] = o; row[f"brier_{c}"] = (p - o) ** 2
                row[f"logs_{c}"] = float(o * np.log(pc) + (1 - o) * np.log(1 - pc))
            sy_rows.append(row)
    return pd.DataFrame(site_rows), pd.DataFrame(pw_rows), pd.DataFrame(sy_rows)
