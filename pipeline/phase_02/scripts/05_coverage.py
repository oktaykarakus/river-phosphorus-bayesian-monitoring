"""O4: monitoring-network coverage, RSN equal-probability diagnostics, river-size composition, clustering and
common support.

Inputs : data_processed/{links_gb_attributes.parquet, links_size.parquet, sites_matched.csv}, pilot site_years
         + site_covariates (frozen), RSN ARD official site list (frozen).
Outputs: tables/p2_03_*.csv|json, data_processed/site_years_p2.csv (site-years + spatial attributes + PS)
Run from phase_02/:  python -I scripts/05_coverage.py
"""
import sys, json
sys.path.insert(0, __import__("os").path.dirname(__file__))
from common import *  # noqa
from netmatch import match_points

import pyogrio
import shapely
from scipy.stats import chisquare, ks_2samp
import statsmodels.formula.api as smf

rng = np.random.default_rng(SEED)


def wq(x, w, q):
    o = np.argsort(x); x, w = np.asarray(x)[o], np.asarray(w)[o]
    c = np.cumsum(w) / w.sum()
    return np.interp(q, c, x)


def frame_table(E):
    """Return England frame links (F1 inland river) with size + WMA, used as river-length population."""
    return E[E.F1_inland_river & E.wma.notna()].copy()


def main():
    ensure_dirs()
    A = pd.read_parquet(PROC / "links_gb_attributes.parquet")
    Z = pd.read_parquet(PROC / "links_size.parquet")
    E = A[A.in_england].merge(Z.drop(columns="up_chan_km_mid"), on="id", how="left")
    F = frame_table(E)
    F["len_km"] = F.len_england_m / 1000
    S = pd.read_csv(PROC / "sites_matched.csv")
    sy = pd.read_csv(PILOT / "data/site_years.csv")
    cov = pd.read_csv(PILOT / "data/site_covariates.csv")
    out = {}

    # ---------------- 1. frame length by WMA x size, under frame variants
    pcols = [f"p_size_{k}" for k in range(4)]
    exp_len = F[pcols].mul(F.len_km, axis=0)
    exp_len["wma"] = F.wma.values
    by_wma_size = exp_len.groupby("wma")[pcols].sum().rename(columns=dict(zip(pcols, SIZE_LABELS)))
    by_wma_size["total_km"] = by_wma_size.sum(axis=1)
    by_wma_size.to_csv(TAB / "p2_03_frame_length_wma_size.csv")
    tot = F.len_km.sum()
    out["frame_F1_total_km"] = float(tot)
    out["frame_F1_size_share_expected"] = (exp_len[pcols].sum() / tot).set_axis(SIZE_LABELS).to_dict()
    out["frame_F1_size_share_point"] = (F.groupby("size_hat", observed=False).len_km.sum() / tot).to_dict()
    out["frame_F1_typology_share"] = (F.groupby(F.typology_size.fillna("none (coastal catchment)")).len_km.sum() / tot).to_dict()
    out["frame_F1_strahler_share"] = (F.groupby("strahler").len_km.sum() / tot).to_dict()

    # ---------------- 2. RSN sites (all ARD active+retired sites with GRTS design coordinates) on the frame
    rs = pd.read_csv(PILOT / "data/rsn_sites_official.csv").drop_duplicates("network_id")
    rs = rs[rs.waterbody_cat == "River"].reset_index(drop=True)
    G = pyogrio.read_dataframe(OS_GPKG, layer="watercourse_link", columns=["id"])
    geoms = G.geometry.values
    A["up_mid"] = A.up_chan_km_mid
    gm, _ = match_points(rs.grts_ngr_easting.values, rs.grts_ngr_northing.values, None, geoms, A)
    rs = pd.concat([rs, gm], axis=1)
    rs = rs[rs.link >= 0]
    rs["link_id"] = A.id.values[rs.link]
    rs = rs.merge(E[["id", "wma", "F1_inland_river", "area_hat_km2", "size_hat", "len_england_m"] + pcols], left_on="link_id", right_on="id", how="left")
    rsF = rs[rs.F1_inland_river == True]
    rs.drop(columns=[c for c in rs.columns if c.startswith("Unnamed")]).to_csv(PROC / "rsn_design_points_matched.csv", index=False)
    out["rsn_ard_river_sites"] = int(len(rs)); out["rsn_ard_on_F1"] = int(len(rsF))
    out["rsn_grts_match_class_counts"] = rs.cls.value_counts().to_dict()

    # 2a. Which truncation of the OS frame makes RSN design points look like an equal-probability-per-length sample?
    # The RSN sample is NOT truncated; only the frame is. Truncation is on upstream channel length (measured identically,
    # without model error, for frame links and design points). Points on channels absent from OS cannot enter (see 04).
    upF = F.up_chan_km_mid.values
    rs["up_pt"] = A.up_chan_km_top.values[rs.link.values] + rs.frac.values * A.chan_km.values[rs.link.values]
    rsF = rs[rs.F1_inland_river == True]
    up_r = np.log(np.maximum(rsF.up_pt.values, 0.05))
    rows = []
    for a in [0, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 8, 10]:
        Ft = F[upF >= a]
        p = Ft.len_km.values / Ft.len_km.sum()
        samp = np.log(np.maximum(Ft.up_chan_km_mid.values[rng.choice(len(Ft), 50000, p=p)], 0.05))
        ks = ks_2samp(up_r, samp)
        rows.append(dict(trunc_up_km=a, frame_km=Ft.len_km.sum(), frame_share_of_F1=Ft.len_km.sum() / F.len_km.sum(),
                         rsn_share_below=float((rsF.up_pt.values < a).mean()), ks_D=ks.statistic, ks_p=ks.pvalue,
                         frame_median_up_km=float(np.exp(np.median(samp))), rsn_median_up_km=float(np.exp(np.median(up_r))),
                         frame_median_area_hat=float(np.exp(wq(np.log(Ft.area_hat_km2), Ft.len_km, .5)))))
    trunc = pd.DataFrame(rows); trunc.to_csv(TAB / "p2_03_rsn_frame_truncation.csv", index=False)
    a_star = float(trunc.loc[trunc.ks_D.idxmin(), "trunc_up_km"])
    out["rsn_frame_truncation_best_up_km"] = a_star
    out["rsn_frame_truncation_best_frame_share"] = float(trunc.loc[trunc.ks_D.idxmin(), "frame_share_of_F1"])
    qrows = []
    for q in [.05, .1, .25, .5, .75, .9, .95]:
        qrows.append(dict(q=q, rsn_design_up_km=float(np.exp(np.quantile(up_r, q))), rsn_design_area_hat=float(rsF.area_hat_km2.quantile(q)),
                          frame_F1_up_km=float(wq(F.up_chan_km_mid, F.len_km, q)), frame_F1_area_hat=float(np.exp(wq(np.log(F.area_hat_km2), F.len_km, q))),
                          frame_trunc_up_km=float(wq(F[upF >= a_star].up_chan_km_mid, F[upF >= a_star].len_km, q))))
    pd.DataFrame(qrows).to_csv(TAB / "p2_03_size_quantiles_rsn_vs_frame.csv", index=False)
    rs["F_trunc"] = rs.F1_inland_river & (A.set_index("id").up_chan_km_mid.reindex(rs.link_id).values >= a_star)

    # 2b. RSN density per WMA vs length-proportional expectation (frame variants)
    drows = []
    for fname, tr in [("F1 all inland river", 0.0), (f"F1 trunc upstream>={a_star:g} km", a_star)]:
        Ft = F[upF >= tr]
        Lw = Ft.groupby("wma").len_km.sum()
        r = rsF if tr == 0 else rsF[rsF.up_pt >= tr]
        obs = r.groupby("wma").size().reindex(Lw.index).fillna(0)
        exp = Lw / Lw.sum() * obs.sum()
        chi = chisquare(obs.values, exp.values)
        for w in Lw.index:
            drows.append(dict(frame=fname, wma=w, frame_km=Lw[w], rsn_obs=obs[w], rsn_exp=exp[w], ratio=obs[w] / exp[w],
                              rsn_per_1000km=obs[w] / Lw[w] * 1000, chi2=chi.statistic, chi2_p=chi.pvalue))
    dens = pd.DataFrame(drows); dens.to_csv(TAB / "p2_03_rsn_uniformity_by_wma.csv", index=False)
    out["rsn_uniformity_chi2"] = dens.groupby("frame")[["chi2", "chi2_p"]].first().to_dict("index")
    # by size class (expected length with class probabilities) on truncated frame
    srow = []
    for fname, tr in [("F1 all inland river", 0.0), (f"F1 trunc upstream>={a_star:g} km", a_star)]:
        Ft = F[upF >= tr]
        exp_s = (Ft[pcols].mul(Ft.len_km, axis=0).sum() / Ft.len_km.sum()).values * len(rsF)
        obs_s = rsF[pcols].sum().values
        for k, lab in enumerate(SIZE_LABELS):
            srow.append(dict(frame=fname, size=lab, rsn_expected=exp_s[k], rsn_observed_probsum=obs_s[k], ratio=obs_s[k] / exp_s[k]))
    pd.DataFrame(srow).to_csv(TAB / "p2_03_rsn_uniformity_by_size.csv", index=False)
    out["rsn_ard_river_sites_total"] = int((pd.read_csv(PILOT / "data/rsn_sites_official.csv").drop_duplicates("network_id").waterbody_cat == "River").sum())
    out["rsn_ard_river_grts_unmatched_250m"] = out["rsn_ard_river_sites_total"] - out["rsn_ard_river_sites"]

    # ---------------- 3. monitoring density & size composition of RSN vs MS (sites with primary site-years)
    S["wma"] = S.pt_wma.fillna(S.wma_pilot)
    rows = []
    Lw = F.groupby("wma").len_km.sum()
    Lsize = pd.Series(exp_len[pcols].sum().values, index=SIZE_LABELS)
    for net in ["RSN", "MS", "MN_OTHER"]:
        d = S[S[f"primary_{net}"] & (S.cls != "U")]
        c = d.groupby("wma").size().reindex(Lw.index).fillna(0)
        for w in Lw.index:
            rows.append(dict(network=net, stratum_type="wma", stratum=w, sites=c[w], frame_km=Lw[w], sites_per_1000km=c[w] / Lw[w] * 1000))
        cs = d[pcols].sum().set_axis(SIZE_LABELS)
        for s in SIZE_LABELS:
            rows.append(dict(network=net, stratum_type="size", stratum=s, sites=cs[s], frame_km=Lsize[s], sites_per_1000km=cs[s] / Lsize[s] * 1000))
        rows.append(dict(network=net, stratum_type="england", stratum="all", sites=len(d), frame_km=tot, sites_per_1000km=len(d) / tot * 1000))
    pd.DataFrame(rows).to_csv(TAB / "p2_03_monitoring_density.csv", index=False)

    # size distributions by network (site-level)
    srows = []
    for net in ["RSN", "MS", "MN_OTHER"]:
        d = S[S[f"primary_{net}"] & (S.cls != "U")]
        srows.append(dict(network=net, sites=len(d), median_area=d.area_hat_km2.median(), q25=d.area_hat_km2.quantile(.25),
                          q75=d.area_hat_km2.quantile(.75), median_up_km=d.up_km.median(), median_strahler=d.link_strahler.median(),
                          **{f"share_{s}": float(d[f'p_size_{k}'].mean()) for k, s in enumerate(SIZE_LABELS)},
                          **{f"typology_{s}": float((d.size_typology == s).mean()) for s in SIZE_LABELS},
                          share_heavily_modified=np.nan))
    srows.append(dict(network="FRAME F1 (length-weighted)", sites=np.nan, median_area=float(np.exp(wq(np.log(F.area_hat_km2), F.len_km, .5))),
                      q25=float(np.exp(wq(np.log(F.area_hat_km2), F.len_km, .25))), q75=float(np.exp(wq(np.log(F.area_hat_km2), F.len_km, .75))),
                      **{f"share_{s}": float(v) for s, v in zip(SIZE_LABELS, exp_len[pcols].sum().values / tot)}))
    pd.DataFrame(srows).to_csv(TAB / "p2_03_size_by_network.csv", index=False)

    # agreement between size proxies at sites
    m = S[(S.cls != "U") & S.size_typology.notna() & (S.is_RSN | S.is_MS)]
    out["site_size_hat_vs_typology_agreement"] = float((m.size_hat.astype(str) == m.size_typology.astype(str)).mean())
    out["site_size_hat_vs_typology_confusion"] = pd.crosstab(m.size_typology, m.size_hat).to_dict()
    out["site_spearman_area_strahler"] = float(m[["area_hat_km2", "link_strahler"]].corr("spearman").iloc[0, 1])

    # ---------------- 4. clustering: NN distances vs uniform-on-frame simulation
    def nn(x, y):
        P = shapely.points(x, y); T = shapely.STRtree(P)
        i, j = T.query(P, predicate="dwithin", distance=200000)
        d = shapely.distance(P[i], P[j]); d[i == j] = np.inf
        return pd.Series(d).groupby(i).min().values
    Ft_geo = pyogrio.read_dataframe(PROC / "frame_links_england.gpkg", columns=["id", "F1_inland_river"])
    Ft_geo = Ft_geo[Ft_geo.F1_inland_river.astype(bool)].merge(F[["id", "area_hat_km2", "len_km"]], on="id")
    crow = []
    for net in ["RSN", "MS"]:
        d = S[S[f"primary_{net}"] & (S.cls != "U")]
        obs_nn = np.median(nn(d.easting.values, d.northing.values))
        Ft_geo["up"] = Ft_geo.id.map(A.set_index("id").up_chan_km_mid)
        for fname, Fg in [("F1", Ft_geo), (f"F1 trunc upstream>={a_star:g} km", Ft_geo[Ft_geo.up >= a_star])]:
            p = Fg.len_km.values / Fg.len_km.sum(); sims = []
            for b in range(99):
                k = rng.choice(len(Fg), len(d), p=p)
                pt = shapely.line_interpolate_point(Fg.geometry.values[k], rng.random(len(d)), normalized=True)
                sims.append(np.median(nn(shapely.get_x(pt), shapely.get_y(pt))))
            crow.append(dict(network=net, frame=fname, n=len(d), obs_median_nn_km=obs_nn / 1000, sim_median_nn_km=np.median(sims) / 1000,
                             sim_lo=np.quantile(sims, .025) / 1000, sim_hi=np.quantile(sims, .975) / 1000, ratio=obs_nn / np.median(sims)))
    pd.DataFrame(crow).to_csv(TAB / "p2_03_clustering_nn.csv", index=False)

    # ---------------- 5. site-years with attributes + common support
    keep = ["site", "cls", "dist", "ambiguous", "link_form", "link_up_chan_km_mid_osdir", "link_direction_repaired" if "link_direction_repaired" in S else "link_flag_direction_repaired", "up_km", "area_hat_km2", "la_hat", "size_hat", "size_hat_maxprob",
            "size_typology", "link_strahler", "link_typology_geology", "link_water_body_id", "pt_water_body_id", "opcat", "mancat",
            "pt_wma", "wma_pilot", "catchment_consistent"] + pcols
    P = sy[sy.network.isin(["RSN", "MS"]) & RULES["primary"](sy)].merge(S[keep], on="site", how="left").merge(cov, on="site", how="left")
    P["wma"] = P.area.map(AREA_TO_WMA)
    P["ms"] = (P.network == "MS").astype(int)
    P["log_area"] = P.la_hat
    P["log_edm5"] = np.log1p(P.edm_n_5km)
    P["geology"] = P.link_typology_geology.fillna("Unknown").replace({"Salt": "Unknown"})
    P["size_cls"] = P.size_hat.astype(str)
    out["site_years_primary"] = int(len(P)); out["site_years_unmatched"] = int((P.cls == "U").sum())
    Pm = P[P.cls != "U"].dropna(subset=["log_area", "log_edm5", "wma"]).copy()
    out["site_years_matched_missing_covariates"] = int(((P.cls != "U") & P[["log_area", "log_edm5", "wma"]].isna().any(axis=1)).sum())
    f_ps = "ms ~ C(wma) + log_area + I(log_area**2) + C(geology) + log_edm5 + C(year)"
    ps = smf.glm(f_ps, data=Pm, family=__import__("statsmodels.api").api.families.Binomial()).fit()
    Pm["ps"] = ps.predict(Pm)
    from sklearn.metrics import roc_auc_score
    out["ps_auc"] = float(roc_auc_score(Pm.ms, Pm.ps))
    out["ps_formula"] = f_ps
    for lo in [0.05, 0.1]:
        cs = Pm.ps.between(lo, 1 - lo)
        out[f"cs_{lo}_share_retained"] = {"RSN": float(cs[Pm.ms == 0].mean()), "MS": float(cs[Pm.ms == 1].mean())}
    # overlap of size distributions: share of each network's site-years whose log-area lies outside the other's 1-99% range
    for a, b, na, nb in [(0, 1, "RSN", "MS"), (1, 0, "MS", "RSN")]:
        xa = Pm[Pm.ms == a].log_area; xb = Pm[Pm.ms == b].log_area
        out[f"share_{na}_outside_{nb}_size_range_1_99"] = float(((xa < xb.quantile(.01)) | (xa > xb.quantile(.99))).mean())
    # standardised mean differences
    smd = {}
    for v in ["log_area", "log_edm5", "link_strahler"]:
        a, b = Pm[Pm.ms == 1][v], Pm[Pm.ms == 0][v]
        smd[v] = float((a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2))
    out["smd_ms_minus_rsn"] = smd
    # sparse cells WMA x size
    cells = Pm.groupby(["wma", "size_cls", "network"]).size().unstack(fill_value=0)
    cells.to_csv(TAB / "p2_03_cells_wma_size.csv")
    out["cells_total"] = int(len(cells)); out["cells_both_networks"] = int(((cells.RSN > 0) & (cells.MS > 0)).sum())
    out["cells_both_ge5"] = int(((cells.RSN >= 5) & (cells.MS >= 5)).sum())
    out["rsn_siteyears_in_cells_with_ms"] = float(cells.RSN[(cells.MS > 0)].sum() / cells.RSN.sum())
    out["ms_siteyears_in_cells_with_rsn"] = float(cells.MS[(cells.RSN > 0)].sum() / cells.MS.sum())
    # frame length in cells without any RSN / MS primary site
    fl = exp_len.groupby("wma")[pcols].sum().rename(columns=dict(zip(pcols, SIZE_LABELS))).stack().rename("km").reset_index().rename(columns={"level_1": "size_cls"})
    c2 = cells.reset_index()
    fl = fl.merge(c2, on=["wma", "size_cls"], how="left").fillna({"RSN": 0, "MS": 0})
    fl.to_csv(TAB / "p2_03_frame_cells_monitoring.csv", index=False)
    out["frame_km_in_cells_without_RSN"] = float(fl[fl.RSN == 0].km.sum()); out["frame_km_in_cells_without_MS"] = float(fl[fl.MS == 0].km.sum())
    out["frame_km_in_cells_without_either"] = float(fl[(fl.RSN == 0) & (fl.MS == 0)].km.sum())
    # sampling frequency by network x size
    fq = Pm.groupby(["network", "size_cls"]).n.describe()[["count", "mean", "50%"]]
    fq.to_csv(TAB / "p2_03_frequency_by_size.csv")
    P = P.merge(Pm[["site", "year", "network", "ps"]], on=["site", "year", "network"], how="left")
    P.to_csv(PROC / "site_years_p2.csv", index=False)
    dump_json(out, TAB / "p2_03_coverage.json")
    for k, v in out.items():
        log(f"{k}: {v}")
    print(trunc.round(3).to_string()); print(dens[dens.frame == dens.frame.iloc[0]].round(2).to_string())


if __name__ == "__main__":
    main()
