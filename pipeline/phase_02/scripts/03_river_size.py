"""O3: river-size characterisation.

Calibrates the network-derived upstream channel length (OS Open Rivers accumulation, 02_build_frame)
against OFFICIAL gauged catchment areas (NRFA station metadata) at English gauging stations, giving a
*modelled* upstream catchment-area estimate with predictive uncertainty for every link / site.

Variables compared (P2_03):
  up_chan_km      derived proxy   (topology accumulation, OS Open Rivers)
  strahler        derived proxy   (topology, OS Open Rivers)
  area_hat_km2    modelled        (log-log calibration on NRFA gauged areas; geology-specific)
  typology_size   official        (WFD Cycle 3 water-body typology size band; water-body level)

Outputs
  data_processed/nrfa_gauges_matched.csv
  tables/p2_03_area_calibration.csv, tables/p2_03_area_calibration.json
  data_processed/links_size.parquet  (England links: area_hat, size class probabilities)
Run from phase_02/:  python -I scripts/03_river_size.py
"""
import sys, json
sys.path.insert(0, __import__("os").path.dirname(__file__))
from common import *  # noqa
from netmatch import match_points

import geopandas as gpd
import pyogrio
import shapely
import statsmodels.formula.api as smf
from scipy.stats import norm


def load_links():
    A = pd.read_parquet(PROC / "links_gb_attributes.parquet")
    G = pyogrio.read_dataframe(OS_GPKG, layer="watercourse_link", columns=["id"])
    assert (G.id.values == A.id.values).all()
    return A, G.geometry.values


def upstream_at(links, link_idx, frac):
    l = links.iloc[link_idx]
    return l.up_chan_km_top.values + np.asarray(frac) * l.chan_km.values


def main():
    ensure_dirs()
    A, geoms = load_links()
    elig = (A.form.isin(["inlandRiver", "tidalRiver"]) | A.canal_carries_flow).values   # channels, not lake lines
    eidx = np.flatnonzero(elig)
    links_e = A.iloc[eidx].reset_index(drop=True)
    links_e["up_mid"] = links_e.up_chan_km_mid
    # ---- NRFA gauges in England
    d = json.load(open(NRFA_JSON))["data"]
    g = pd.DataFrame([{k: r.get(k) for k in ["id", "name", "river", "location", "easting", "northing", "catchment-area",
                                             "measuring-authority-id", "closed", "factors-affecting-runoff"]} for r in d])
    g = g.rename(columns={"catchment-area": "area_km2", "measuring-authority-id": "authority"})
    g = g[g.area_km2.notna() & g.easting.notna() & (g.area_km2 > 0)].reset_index(drop=True)
    W = gpd.read_file(WFD_GPKG, engine="pyogrio", columns=["water_body_id", "water_body_typology", "ea_area_name"])
    pts = gpd.GeoDataFrame(g, geometry=gpd.points_from_xy(g.easting, g.northing), crs=CRS)
    j = gpd.sjoin(pts, W, how="left", predicate="within"); j = j[~j.index.duplicated()]
    g["in_england"] = j.water_body_id.notna().values
    g = g[g.in_england & (g.id < 200000)].reset_index(drop=True)      # ids >= 200000: Northern Ireland (Irish Grid)
    labels = (g.river.fillna("") + " " + g.name.fillna("")).str.upper().tolist()
    m, _ = match_points(g.easting.values, g.northing.values, labels, geoms[eidx], links_e)
    g = pd.concat([g, m.add_prefix("m_")], axis=1)
    ok = g.m_link >= 0
    g.loc[ok, "up_km"] = upstream_at(links_e, g.loc[ok, "m_link"].values, g.loc[ok, "m_frac"].values)
    g.loc[ok, "strahler"] = links_e.strahler.values[g.loc[ok, "m_link"].values]
    g.loc[ok, "geology"] = links_e.typology_geology.values[g.loc[ok, "m_link"].values]
    g.loc[ok, "wma"] = links_e.wma.values[g.loc[ok, "m_link"].values]
    g.loc[ok, "rbd"] = links_e.river_basin_district.values[g.loc[ok, "m_link"].values]
    g.to_csv(PROC / "nrfa_gauges_matched.csv", index=False)

    # ---- calibration sample: class A/B matches, up_km > 0.5 km
    cal = g[g.m_cls.isin(["A", "B"]) & (g.up_km > 0.5)].copy()
    cal["geology"] = cal.geology.fillna("Unknown").replace({"Salt": "Unknown"})
    cal["la"] = np.log(cal.area_km2); cal["lu"] = np.log(cal.up_km)
    cal["ratio"] = cal.area_km2 / cal.up_km
    res = {}
    models = {"M1_loglog": "la ~ lu", "M2_loglog_geology": "la ~ lu + C(geology)",
              "M3_loglog_geology_rbd": "la ~ lu + C(geology) + C(rbd)"}
    # leave-one-RBD-out CV (spatially blocked) for model choice
    cvrows = []
    for name, f in models.items():
        fit = smf.ols(f, data=cal).fit()
        res[name] = dict(n=int(fit.nobs), r2=float(fit.rsquared), sigma=float(np.sqrt(fit.scale)), params=fit.params.to_dict())
        errs = []
        for r in cal.rbd.dropna().unique():
            tr, te = cal[cal.rbd != r], cal[cal.rbd == r]
            if name == "M3_loglog_geology_rbd":
                continue
            ft = smf.ols(f, data=tr).fit()
            te2 = te[te.geology.isin(tr.geology.unique())]
            errs.append(te2.la - ft.predict(te2))
        if errs:
            e = pd.concat(errs)
            res[name].update(cv_rmse_log=float(np.sqrt((e ** 2).mean())), cv_n=int(len(e)),
                             cv_within_factor2=float((np.abs(e) < np.log(2)).mean()))
    # robustness: exclude gauges flagged with artificial influences? report only
    # diagnostic: same model using the PUBLISHED OS flow direction (no repair), link-midpoint upstream length
    cal["lu_os"] = np.log(np.maximum(links_e.up_chan_km_mid_osdir.values[cal.m_link.values], 0.05))
    fo = smf.ols("la ~ lu_os + C(geology)", data=cal).fit()
    res["M2_with_published_OS_direction"] = dict(n=int(fo.nobs), r2=float(fo.rsquared), sigma=float(np.sqrt(fo.scale)))
    best = "M2_loglog_geology"
    fit = smf.ols(models[best], data=cal).fit()
    sig = float(np.sqrt(fit.scale))
    cal["la_hat"] = fit.predict(cal)
    cal["cls_obs"] = size_class(cal.area_km2).values
    cal["cls_hat"] = size_class(np.exp(cal.la_hat)).values
    res["chosen"] = best
    res["class_agreement_gauges"] = float((cal.cls_obs.astype(str) == cal.cls_hat.astype(str)).mean())
    res["confusion_gauges"] = pd.crosstab(cal.cls_obs, cal.cls_hat).to_dict()
    res["spearman_strahler_area"] = float(cal[["strahler", "area_km2"]].corr("spearman").iloc[0, 1])
    res["spearman_uplen_area"] = float(cal[["up_km", "area_km2"]].corr("spearman").iloc[0, 1])
    res["gauges_england"] = int(len(g)); res["gauges_matched_AB"] = int(g.m_cls.isin(["A", "B"]).sum())
    res["gauges_calibration"] = int(len(cal))
    res["drainage_density_km_per_km2_by_geology"] = (1 / cal.groupby("geology").ratio.median()).to_dict()
    res["note"] = ("area_hat is a MODELLED estimate: drainage density varies (chalk / permeable catchments have sparse "
                   "mapped channels); gauges over-represent medium/large rivers, so the calibration is extrapolated "
                   "for catchments < ~10 km2.")
    res["gauge_area_quantiles_km2"] = cal.area_km2.quantile([0, .05, .25, .5, .75, .95, 1]).to_dict()
    cal.to_csv(TAB / "p2_03_area_calibration_sample.csv", index=False)
    pd.DataFrame([{**{"model": k}, **{kk: vv for kk, vv in v.items() if kk != "params"}} for k, v in res.items()
                  if isinstance(v, dict) and "r2" in v]).to_csv(TAB / "p2_03_area_calibration.csv", index=False)
    dump_json(res, TAB / "p2_03_area_calibration.json")
    for k, v in res.items():
        log(f"{k}: {v}")

    # ---- apply to all England links (midpoint upstream length)
    E = A[A.in_england].copy()
    E["geology"] = E.typology_geology.fillna("Unknown").replace({"Salt": "Unknown"})
    E.loc[~E.geology.isin(cal.geology.unique()), "geology"] = "Unknown" if "Unknown" in cal.geology.unique() else "Calcareous"
    E["lu"] = np.log(np.maximum(E.up_chan_km_mid, 0.05))
    E["la_hat"] = fit.predict(E[["lu", "geology"]])
    E["area_hat_km2"] = np.exp(E.la_hat)
    br = np.log(np.array(SIZE_BREAKS[1:-1]))
    cdf = norm.cdf((br[None, :] - E.la_hat.values[:, None]) / sig)
    probs = np.diff(np.c_[np.zeros(len(E)), cdf, np.ones(len(E))], axis=1)
    for k, lab in enumerate(SIZE_LABELS):
        E[f"p_size_{k}"] = probs[:, k]
    E["size_hat"] = size_class(E.area_hat_km2.values).values
    E["size_hat_maxprob"] = probs.max(axis=1)
    E[["id", "up_chan_km_mid", "area_hat_km2", "la_hat", "size_hat", "size_hat_maxprob"] + [f"p_size_{k}" for k in range(4)]] \
        .to_parquet(PROC / "links_size.parquet", index=False)
    calib = dict(formula=models[best], params=fit.params.to_dict(), sigma=sig, geology_levels=sorted(cal.geology.unique()))
    dump_json(calib, PROC / "area_calibration_model.json")


if __name__ == "__main__":
    main()
