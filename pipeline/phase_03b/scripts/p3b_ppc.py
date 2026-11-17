"""Posterior predictive checks for Phase 3B (shared by the seasonal assessment and the spatial diagnostics).

Replicates measurements at the OBSERVED dates (MS-§8) for n_draws posterior draws, censors at the observation's own
limit when censored in the data, otherwise at the precision-based limit (rule A: 4 decimals -> 0.001, else 0.01 mg/L;
Phase 3A). Statistics:
  * measurement quantiles and censored fraction by network;
  * monthly mean log concentration profiles by network, by region x network and by size class x network
    (months inside the 90% predictive interval);
  * site-year sample arithmetic-mean exceedance shares at c = 0.05/0.10/0.20 by network, region x network, size x network.
"""
import numpy as np, pandas as pd
import p3b_lib as B

QS = [.01, .05, .25, .5, .75, .95, .99]
SIZE_ORDER = ["XS (<10 km²)", "S (10–100)", "M (100–1,000)", "L (≥1,000)"]


def ppc(D, meas, sites, sy, mc, n_draws=200, seed=None):
    rng = np.random.default_rng(seed or B.CFG["seeds"]["ppc"])
    S = B.ndraws(D); pick = np.linspace(0, S - 1, n_draws).astype(int)
    lim = np.where(meas.censored, meas.limit, np.where(meas.decimals >= 4, 0.001, 0.01))
    yobs = meas.value.to_numpy(); cobs = meas.censored.to_numpy()
    onet = meas.network.to_numpy(); month = meas.month.to_numpy(); si = meas.site_idx.to_numpy() - 1; sj = meas.siteyear_idx.to_numpy() - 1
    oreg = sites.region.to_numpy()[si]; osize = sites.size_cls.to_numpy()[si]
    sgs = B.sig_site(D, sites, mc)
    groups = {"network": onet, "region_network": np.char.add(np.char.add(oreg.astype(str), "|"), onet.astype(str)),
              "size_network": np.char.add(np.char.add(osize.astype(str), "|"), onet.astype(str))}
    syn = sy.network.to_numpy(); sreg = sites.region.to_numpy()[sy.site_idx.to_numpy() - 1]; ssize = sites.size_cls.to_numpy()[sy.site_idx.to_numpy() - 1]
    sy_groups = {"network": syn, "region_network": np.char.add(np.char.add(sreg.astype(str), "|"), syn.astype(str)),
                 "size_network": np.char.add(np.char.add(ssize.astype(str), "|"), syn.astype(str))}
    # observed statistics
    def monthly(vals, g):
        df = pd.DataFrame({"g": g, "m": month, "v": np.log(vals)})
        return df.groupby(["g", "m"]).v.mean()
    obs_month = {k: monthly(yobs, g) for k, g in groups.items()}
    obs_mean = sy.mean_hi.to_numpy()
    rep_month = {k: [] for k in groups}; rep_exc = {k: [] for k in sy_groups}; rep_q = {"RSN": [], "MS": []}; rep_cens = {"RSN": [], "MS": []}
    cnt = np.bincount(sj, minlength=len(sy))
    for b0 in range(0, len(pick), 50):
        idx = pick[b0:b0 + 50]
        eta = B.eta_obs(D, meas, sites, sy, mc, draws=idx)
        for k_, s in enumerate(idx):
            y = np.exp(eta[k_] + sgs[s][si] * rng.standard_normal(len(meas)))
            cen = y < lim; yv = np.where(cen, lim, y)
            for n in ["RSN", "MS"]:
                kk = onet == n
                rep_q[n].append(np.quantile(yv[kk], QS)); rep_cens[n].append(cen[kk].mean())
            for kname, g in groups.items():
                rep_month[kname].append(monthly(yv, g))
            smean = np.bincount(sj, weights=yv, minlength=len(sy)) / cnt
            for kname, g in sy_groups.items():
                df = pd.DataFrame({"g": g, **{f"c{c}": smean > c for c in B.THRESH}})
                rep_exc[kname].append(df.groupby("g").mean())
    out = {}
    for n in ["RSN", "MS"]:
        kk = onet == n; rq = np.array(rep_q[n]); oq = np.quantile(yobs[kk], QS)
        out[f"quantiles_{n}"] = {str(q): dict(observed=float(o), rep05=float(np.quantile(rq[:, i], .05)), rep95=float(np.quantile(rq[:, i], .95)),
                                               inside=bool(np.quantile(rq[:, i], .05) <= o <= np.quantile(rq[:, i], .95))) for i, (q, o) in enumerate(zip(QS, oq))}
        out[f"censored_{n}"] = dict(observed=float(cobs[kk].mean()), rep=B.band(rep_cens[n]))
    month_tables = {}
    for kname in groups:
        R = pd.concat(rep_month[kname], axis=1)
        lo, hi, med = R.quantile(.05, axis=1), R.quantile(.95, axis=1), R.median(axis=1)
        o = obs_month[kname].reindex(R.index)
        t = pd.DataFrame({"observed": o, "rep05": lo, "rep_median": med, "rep95": hi}); t["inside"] = (t.observed >= t.rep05) & (t.observed <= t.rep95)
        # standardised discrepancy (observed - median) / predictive SD
        t["z"] = (t.observed - med) / R.std(axis=1)
        month_tables[kname] = t
        g = t.reset_index().rename(columns={"level_0": "group", "g": "group"})
        out[f"monthly_{kname}"] = {grp: dict(months_inside=int(gg.inside.sum()), n_months=int(gg.inside.size), mean_abs_z=float(np.abs(gg.z).mean()))
                                   for grp, gg in g.groupby("group")}
    exc_tables = {}
    for kname, g in sy_groups.items():
        R = pd.concat(rep_exc[kname], keys=range(len(rep_exc[kname])))
        obs = pd.DataFrame({"g": g, **{f"c{c}": obs_mean > c for c in B.THRESH}}).groupby("g").mean()
        rows = []
        for grp in obs.index:
            r = R.xs(grp, level=1)
            for c in B.THRESH:
                x = r[f"c{c}"].to_numpy()
                rows.append(dict(group=grp, threshold=c, observed=float(obs.loc[grp, f"c{c}"]), rep05=float(np.quantile(x, .05)), rep_median=float(np.median(x)),
                                 rep95=float(np.quantile(x, .95)), n=int((g == grp).sum())))
        t = pd.DataFrame(rows); t["inside"] = (t.observed >= t.rep05) & (t.observed <= t.rep95)
        exc_tables[kname] = t
        out[f"samplemean_exceed_{kname}_inside"] = f"{int(t.inside.sum())}/{len(t)}"
    return out, month_tables, exc_tables
