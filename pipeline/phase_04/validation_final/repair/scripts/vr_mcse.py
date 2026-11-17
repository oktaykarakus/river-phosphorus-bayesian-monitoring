"""Track A validation repair (VR): correlation-aware Monte Carlo error for held-out site-joint ELPD sums and model differences.

Pure numpy/scipy. Does not import the frozen scorer and does not depend on arviz (arviz is used only in the tests as an
independent cross-check).

Set-up (one fitted model)
-------------------------
Held-out sites i = 1..N; retained posterior draws s = 1..S, stored chain-major in `chains` equal-length chains.
lp[i, s] = log p(y_i | theta_s) is the site-joint log predictive density of site i's held-out measurements under draw s
(the frozen scorer has already integrated the site level, site-year shocks and the noise SD by quadrature).

The scored quantity, which is NOT changed here, is the plug-in estimate
    ELPD_hat = sum_i log( (1/S) sum_s w[i, s] ),   w = exp(lp).
No importance weighting is involved: w[i, .] is a plain Monte Carlo integrand whose posterior mean is p(y_i | y_train).

First-order (delta-method) Monte Carlo error
--------------------------------------------
log mu_hat_i - log mu_i ~= (1/S) sum_s psi[i, s],   psi = w / mu - 1   (mu estimated by mu_hat).
Summing over sites gives ONE draw-level series T_s = sum_i psi[i, s], so
    Var(ELPD_hat) ~= Var(T) / ESS_mean(T)
          = [ sum_i Var(psi_i) + sum_{i != j} Cov(psi_i, psi_j) ] / ESS_mean(T).
Every cross-site covariance (all sites share the same posterior draws) and the autocorrelation of T along the chains are
included. The frozen Phase 4B formula, sqrt(sum_i Var(psi_i) / ESS_i), sets every Cov(psi_i, psi_j) to zero.

Two models fitted by separate MCMC runs have independent Monte Carlo errors, so the error of their difference is
sqrt(mcse_A^2 + mcse_B^2). A draw-paired estimator Var(T_A - T_B)/ESS is justified only when the two models' draws are
coupled (common draws / common random numbers); `mcse_paired` exists for that case and for the tests.

Second order: E[log mu_hat_i] - log mu_i ~= -Var(psi_i) / (2 ESS_i) (Jensen). Reported as a bias diagnostic only.

Heavy tails: w[i, .] can be extremely right-skewed. Pareto k_hat of w[i, .] (Zhang & Stephens 2009 GPD fit with the weakly
informative prior used by loo/PSIS) diagnoses this. Variance is infinite for k > 0.5; the plain MC mean is unreliable for
k_hat >= min(1 - 1/log10(S), 0.7) (Vehtari, Simpson, Gelman, Yao & Gabry 2024, JMLR 25:72). The delta-method MCSE of a
site above that threshold is not trustworthy, so such sites are flagged and never certified by a MCSE value.
"""
from __future__ import annotations

import numpy as np

__all__ = ["chain_matrix", "ess_mean", "gpd_khat", "pareto_khat", "khat_threshold", "site_influence", "mc_summary",
           "mcse_independent_fits", "mcse_paired", "block_jackknife_elpd", "elpd_hat"]


# ----------------------------------------------------------------------------------------------------------- ESS
def chain_matrix(x, chains):
    x = np.asarray(x, float).ravel()
    if x.size % chains:
        raise ValueError(f"{x.size} draws do not split into {chains} equal chains")
    return x.reshape(chains, x.size // chains)


def _autocov(xc):
    """Biased (1/n) autocovariance of each chain (rows), lags 0..n-1, via FFT."""
    m, n = xc.shape
    z = xc - xc.mean(1, keepdims=True)
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    f = np.fft.rfft(z, nfft, axis=1)
    return np.fft.irfft(f * np.conj(f), nfft, axis=1)[:, :n] / n


def ess_mean(x, chains, cap=True, split=True):
    """Multi-chain effective sample size for the MEAN of x (not rank-normalised), Vehtari et al. (2021) §3.2:
    combined autocorrelation with the between-chain variance, Geyer's initial positive then monotone sequence.
    split=True halves every chain first (the published definition; arviz 'mean'); split=False uses whole chains.
    cap=True returns min(ESS, S) (conservative: antithetic chains are not credited)."""
    xc = chain_matrix(x, chains)
    if split:
        h = xc.shape[1] // 2
        xc = np.vstack([xc[:, :h], xc[:, xc.shape[1] - h:]])
    m, n = xc.shape
    S = m * n
    if n < 4 or not np.isfinite(xc).all():
        return np.nan
    if np.ptp(xc) < np.finfo(float).resolution:
        return float(S)
    acov = _autocov(xc)
    W = acov[:, 0].mean() * n / (n - 1)                        # mean within-chain variance (unbiased)
    var_plus = W * (n - 1) / n + (xc.mean(1).var(ddof=1) if m > 1 else 0.0)
    if var_plus <= 0:
        return float(S)
    rho = 1.0 - (W - acov.mean(0)) / var_plus                 # rho[0] = 1 exactly up to rounding
    rho[0] = 1.0
    # Geyer initial positive sequence on pairs Gamma_k = rho[2k] + rho[2k+1]
    npair = n // 2
    G = rho[0:2 * npair:2] + rho[1:2 * npair:2]
    K = npair
    for k in range(1, npair):
        if G[k] <= 0:
            K = k
            break
    extra = rho[2 * K] if (K < npair and rho[2 * K] > 0) else 0.0   # Geyer/Stan: positive even term of the first non-positive pair
    G = G[:K].copy()
    for k in range(1, K):                                     # initial monotone sequence
        if G[k] > G[k - 1]:
            G[k] = G[k - 1]
    tau = -1.0 + 2.0 * G.sum() + extra
    tau = max(tau, 1.0 / np.log10(S))
    e = S / tau
    return float(min(e, S)) if cap else float(e)


# ---------------------------------------------------------------------------------------------- Pareto k-hat
def gpd_khat(x):
    """Shape k of a generalised Pareto fitted to exceedances x >= 0 (Zhang & Stephens 2009 profile-posterior estimate,
    plus the weakly informative prior k <- (n k + 5) / (n + 10) used by loo/PSIS)."""
    x = np.sort(np.asarray(x, float))
    n = x.size
    prior = 3.0
    m = 30 + int(np.sqrt(n))
    xstar = x[int(np.floor(n / 4 + 0.5)) - 1]
    if x[-1] <= 0 or xstar <= 0:
        return -np.inf
    j = np.arange(1, m + 1)
    b = 1.0 / x[-1] + (1.0 - np.sqrt(m / (j - 0.5))) / (prior * xstar)
    kk = np.log1p(-b[:, None] * x[None, :]).mean(1)
    Lp = n * (np.log(-b / kk) - kk - 1.0)
    w = 1.0 / np.exp(Lp[None, :] - Lp[:, None]).sum(1)
    keep = w >= 10 * np.finfo(float).eps
    w = w[keep] / w[keep].sum()
    bpost = float((b[keep] * w).sum())
    k = float(np.log1p(-bpost * x).mean())
    return (n * k + 10 * 0.5) / (n + 10)


def pareto_khat(logw):
    """Right-tail Pareto k_hat of w = exp(logw) from the M = ceil(min(0.2 S, 3 sqrt(S))) largest values (PSIS rule)."""
    lw = np.asarray(logw, float).ravel()
    S = lw.size
    M = int(np.ceil(min(0.2 * S, 3.0 * np.sqrt(S))))
    srt = np.sort(lw - lw.max())
    cut = srt[-(M + 1)]
    tail = np.exp(srt[-M:]) - np.exp(cut)
    if not (tail > 0).any():
        return -np.inf
    return gpd_khat(tail)


def khat_threshold(S):
    """Sample-size-dependent reliability threshold of plain Monte Carlo / PSIS (Vehtari et al. 2024)."""
    return float(min(1.0 - 1.0 / np.log10(S), 0.7))


# ------------------------------------------------------------------------------------------ ELPD Monte Carlo
def elpd_hat(lp):
    """Per-site plug-in log mean density: log((1/S) sum_s exp(lp[i, s]))."""
    lp = np.atleast_2d(np.asarray(lp, float))
    mx = lp.max(1, keepdims=True)
    return (np.log(np.exp(lp - mx).mean(1, keepdims=True)) + mx).ravel()


def site_influence(lp):
    """Per-site elpd_hat (N,) and first-order influence psi (N, S) = w / mean(w) - 1 (scale-free, overflow-safe)."""
    lp = np.atleast_2d(np.asarray(lp, float))
    W = np.exp(lp - lp.max(1, keepdims=True))
    mu = W.mean(1, keepdims=True)
    return elpd_hat(lp), W / mu - 1.0


def mc_summary(lp, chains=4, idx=None, with_sites=True):
    """Monte Carlo error of sum_{i in idx} elpd_hat_i for ONE fitted model.

    Returns a dict with
      elpd            plug-in sum (the frozen point estimate, unchanged)
      mcse            correlation-aware first-order MCSE: sd(T) / sqrt(ESS_mean(T)), T_s = sum_i psi[i, s], with
                      ESS_mean(T) = min(split-chain ESS, whole-chain ESS) (conservative choice between the two variants)
      mcse_split/_whole  the same with each ESS variant
      ess_T, sd_T     its ingredients
      mcse_indep      the same, but with every cross-site covariance set to 0 (sqrt sum_i Var(psi_i)/ESS_i, ESS_i mean-type)
      cov_inflation   mcse^2 / mcse_indep^2
      jensen_bias     sum_i -Var(psi_i) / (2 ESS_i)   (expected bias of the plug-in log, second order)
      k_T             Pareto k_hat of the aggregate sum_i w_i / mu_i (positive)
      sites           per-site table (elpd, sd_psi, ess, mcse, k_hat) if with_sites
    """
    lp = np.atleast_2d(np.asarray(lp, float))
    if idx is not None:
        lp = lp[np.asarray(idx)]
    N, S = lp.shape
    e_i, psi = site_influence(lp)
    T = psi.sum(0)
    ess_split, ess_whole = ess_mean(T, chains, split=True), ess_mean(T, chains, split=False)
    essT = min(ess_split, ess_whole)                          # conservative: the smaller of the two published variants
    sdT = float(T.std(ddof=1))
    var_i = psi.var(1, ddof=1)
    ess_i = np.array([ess_mean(p, chains) for p in psi])
    ess_i = np.where(np.isfinite(ess_i), ess_i, S)
    mcse_i = np.sqrt(var_i / ess_i)
    out = dict(n_sites=int(N), n_draws=int(S), chains=int(chains), elpd=float(e_i.sum()), mcse=sdT / np.sqrt(essT),
               mcse_split=sdT / np.sqrt(ess_split), mcse_whole=sdT / np.sqrt(ess_whole),
               ess_T=float(essT), ess_T_split=float(ess_split), ess_T_whole=float(ess_whole), sd_T=sdT, mcse_indep=float(np.sqrt((mcse_i ** 2).sum())),
               jensen_bias=float(-0.5 * (mcse_i ** 2).sum()), k_T=float(pareto_khat(np.log(T + N))))
    out["cov_inflation"] = out["mcse"] ** 2 / out["mcse_indep"] ** 2 if out["mcse_indep"] > 0 else np.nan
    if with_sites:
        out["sites"] = dict(elpd=e_i, sd_psi=np.sqrt(var_i), ess=ess_i, mcse=mcse_i,
                            khat=np.array([pareto_khat(r) for r in lp]))
    return out


def mcse_independent_fits(*mcses):
    """MCSE of a sum/difference of ELPD totals from SEPARATE MCMC runs (independent Monte Carlo errors)."""
    return float(np.sqrt(np.sum(np.square(mcses))))


def mcse_paired(lpA, lpB, chains=4):
    """MCSE of ELPD_A - ELPD_B when draw s of A and draw s of B are COUPLED (common draws). Not valid for separate fits
    (there the pairing is arbitrary); used only where coupling holds by construction."""
    _, pa = site_influence(lpA)
    _, pb = site_influence(lpB)
    D = pa.sum(0) - pb.sum(0)
    return float(D.std(ddof=1) / np.sqrt(min(ess_mean(D, chains), ess_mean(D, chains, split=False))))


def block_jackknife_elpd(lp, chains=4, blocks_per_chain=5):
    """Delete-one-block jackknife SE of the NONLINEAR estimator sum_i log mean_s w[i, s] (no linearisation).
    Blocks are contiguous within chains, so autocorrelation shorter than a block and all cross-site dependence are kept.
    An independent check of the delta-method `mc_summary(...)['mcse']`."""
    lp = np.atleast_2d(np.asarray(lp, float))
    N, S = lp.shape
    n = S // chains
    if n % blocks_per_chain:
        raise ValueError("chain length must be divisible by blocks_per_chain")
    mx = lp.max(1, keepdims=True)
    W = np.exp(lp - mx)
    lab = np.repeat(np.arange(chains * blocks_per_chain), n // blocks_per_chain)
    Bk = chains * blocks_per_chain
    sums = np.stack([W[:, lab == b].sum(1) for b in range(Bk)], 1)   # (N, B)
    tot = sums.sum(1, keepdims=True)
    cnt = S - S // Bk
    est = (np.log((tot - sums) / cnt) + mx).sum(0)                    # (B,) leave-one-block-out ELPD
    return float(np.sqrt((Bk - 1) / Bk * ((est - est.mean()) ** 2).sum()))
