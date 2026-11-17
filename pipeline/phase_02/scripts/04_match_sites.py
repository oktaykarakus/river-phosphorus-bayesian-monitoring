"""O2: link RSN and routine (MS; MN_OTHER for context) monitoring locations to the OS Open Rivers network.

Inputs : pilot_01/data/site_years.csv (frozen), pilot_01/data/rsn_sites_official.csv (frozen),
         RSN ARD 'RSN Site Catchments' (frozen zip), phase_02 frame + calibration outputs.
Outputs: data_processed/sites_matched.csv           one row per unique sampling point
         data_processed/site_candidates.csv         all candidate links within 250 m
         tables/p2_02_matching_summary.csv, p2_02_threshold_sweep.csv, p2_02_matching.json
Run from phase_02/:  python -I scripts/04_match_sites.py
"""
import sys, json, zipfile
sys.path.insert(0, __import__("os").path.dirname(__file__))
from common import *  # noqa
from netmatch import match_points, R_MAX

import geopandas as gpd
import pyogrio
import shapely
from scipy.stats import norm


def main():
    ensure_dirs()
    A = pd.read_parquet(PROC / "links_gb_attributes.parquet")
    G = pyogrio.read_dataframe(OS_GPKG, layer="watercourse_link", columns=["id"])
    assert (G.id.values == A.id.values).all()
    geoms = G.geometry.values
    A["up_mid"] = A.up_chan_km_mid
    calib = json.load(open(PROC / "area_calibration_model.json"))

    sy = pd.read_csv(PILOT / "data/site_years.csv")
    sy = sy[sy.network.isin(["RSN", "MS", "MN_OTHER"])]
    prim = RULES["primary"](sy)
    S = (sy.assign(prim=prim).groupby("site")
           .agg(label=("label", "first"), easting=("easting", "first"), northing=("northing", "first"),
                area=("area", "first"), region=("region", "first"),
                networks=("network", lambda s: "|".join(sorted(set(s)))),
                n_site_years=("year", "size"), any_primary=("prim", "max"))
           .reset_index())
    for net in ["RSN", "MS", "MN_OTHER"]:
        S[f"is_{net}"] = S.networks.str.split("|").apply(lambda x: net in x)
        pr = set(sy[(sy.network == net) & prim].site)
        S[f"primary_{net}"] = S.site.isin(pr)
    # coordinate checks
    S["coord_missing"] = S.easting.isna() | S.northing.isna()
    S["coord_res_100m"] = (S.easting % 100 == 0) & (S.northing % 100 == 0)
    S["coord_res_10m"] = (S.easting % 10 == 0) & (S.northing % 10 == 0) & ~S.coord_res_100m
    S["coord_in_bng_range"] = S.easting.between(0, 700000) & S.northing.between(0, 1300000)
    S = S[~S.coord_missing].reset_index(drop=True)

    m, cand = match_points(S.easting.values, S.northing.values, S.label.tolist(), geoms, A)
    S = pd.concat([S, m], axis=1)
    cand["site"] = S.site.values[cand.pt]; cand["link_id"] = A.id.values[cand.link]
    cand["form"] = A.form.values[cand.link]; cand["up_mid_km"] = A.up_chan_km_mid.values[cand.link]
    cand["watercourse_name"] = A.watercourse_name.values[cand.link]
    cand.drop(columns="pt").to_csv(PROC / "site_candidates.csv", index=False)

    ok = S.link >= 0
    li = S.loc[ok, "link"].values
    L = A.iloc[li]
    chan = L.chan_km.values
    for c in ["id", "form", "watercourse_name", "strahler", "shreve", "water_body_id", "typology_size", "typology_geology",
              "wma", "flag_flowdir_missing", "flag_parallel_link", "flag_bifurcation_child", "flag_crosses_catchment",
              "flag_crossborder_basin", "split_ratio", "flag_direction_repaired", "canal_carries_flow", "up_chan_km_mid_osdir"]:
        S.loc[ok, "link_" + c] = L[c].values
    S.loc[ok, "up_km"] = L.up_chan_km_top.values + S.loc[ok, "frac"].values * chan
    S.loc[ok, "up_km_nosplit"] = L.up_chan_km_top_nosplit.values + S.loc[ok, "frac"].values * chan
    S.loc[ok, "alt_up_km"] = np.where(S.loc[ok, "alt_link"] >= 0, A.up_chan_km_mid.values[np.maximum(S.loc[ok, "alt_link"].values, 0)], np.nan)

    # modelled upstream area + size-class probabilities at the matched point
    geo = S.link_typology_geology.fillna("Unknown").replace({"Salt": "Unknown"})
    geo = geo.where(geo.isin(calib["geology_levels"]), "Unknown" if "Unknown" in calib["geology_levels"] else "Calcareous")
    p = calib["params"]
    la = p["Intercept"] + p["lu"] * np.log(np.maximum(S.up_km.astype(float), 0.05))
    for lev in calib["geology_levels"]:
        key = f"C(geology)[T.{lev}]"
        if key in p:
            la = la + p[key] * (geo == lev)
    S["la_hat"] = np.where(ok, la, np.nan)
    S["area_hat_km2"] = np.exp(S.la_hat)
    sig = calib["sigma"]
    br = np.log(np.array(SIZE_BREAKS[1:-1]))
    cdf = norm.cdf((br[None, :] - S.la_hat.values[:, None]) / sig)
    probs = np.diff(np.c_[np.zeros(len(S)), cdf, np.ones(len(S))], axis=1)
    for k in range(4):
        S[f"p_size_{k}"] = np.where(ok, probs[:, k], np.nan)
    S["size_hat"] = size_class(S.area_hat_km2.values).values
    S["size_hat_maxprob"] = np.where(ok, probs.max(axis=1), np.nan)
    S["size_typology"] = S.link_typology_size

    # point-in-polygon WFD catchment & WMA of the SITE point
    W = gpd.read_file(WFD_GPKG, engine="pyogrio", columns=["water_body_id", "water_body_type", "water_body_typology", "operational_catchment", "management_catchment"])
    Aw = gpd.read_file(WMA_GPKG, engine="pyogrio", columns=["long_name"])
    pts = gpd.GeoDataFrame(S[["site"]], geometry=gpd.points_from_xy(S.easting, S.northing), crs=CRS)
    j = gpd.sjoin(pts, W, how="left", predicate="within"); j = j[~j.index.duplicated()]
    S["pt_water_body_id"] = j.water_body_id.values; S["pt_wb_type"] = j.water_body_type.values
    S["pt_typology_size"] = j.water_body_typology.str.split(", ").str[1].map(TYPO_SIZE_MAP).values
    S["opcat"] = j.operational_catchment.values; S["mancat"] = j.management_catchment.values
    ja = gpd.sjoin(pts, Aw, how="left", predicate="within"); ja = ja[~ja.index.duplicated()]
    S["pt_wma"] = ja.long_name.values
    S["wma_pilot"] = S.area.map(AREA_TO_WMA)
    S["wma_consistent"] = S.pt_wma == S.wma_pilot
    S["catchment_consistent"] = S.pt_water_body_id == S.link_water_body_id

    # RSN: ARD waterbody and GRTS design point
    rs = pd.read_csv(PILOT / "data/rsn_sites_official.csv")
    rs["site"] = rs.api_notation
    with zipfile.ZipFile(RSN_ARD_ZIP) as z:
        sc = pd.read_csv(z.open("NCEA_River_Surveillance_Network_ARD_Open/RSN Site Catchments.csv"), encoding="utf-8-sig")
    rs = rs.merge(sc[["network_id", "waterbody_name"]], on="network_id", how="left")
    rs = rs.drop_duplicates("site")
    S = S.merge(rs[["site", "network_id", "waterbody_id", "waterbody_cat", "grts_ngr_easting", "grts_ngr_northing",
                    "replacement_site_flag", "site_status"]].rename(columns={"waterbody_id": "ard_waterbody_id"}),
                on="site", how="left")
    isr = S.network_id.notna()
    S["grts_to_wq_m"] = np.hypot(S.grts_ngr_easting - S.easting, S.grts_ngr_northing - S.northing)
    S["ard_wb_consistent_pt"] = np.where(isr, S.ard_waterbody_id == S.pt_water_body_id, np.nan)
    S["ard_wb_consistent_link"] = np.where(isr, S.ard_waterbody_id == S.link_water_body_id, np.nan)
    gpts = shapely.points(S.loc[isr, "grts_ngr_easting"].values, S.loc[isr, "grts_ngr_northing"].values)
    tree = shapely.STRtree(geoms)
    nn = tree.query_nearest(gpts, return_distance=True, all_matches=False)
    S.loc[isr, "grts_to_os_network_m"] = nn[1]

    # co-location between networks (RSN vs MS)
    for a, b in [("RSN", "MS")]:
        pa = S[S[f"is_{a}"]]; pb = S[S[f"is_{b}"]]
        t = shapely.STRtree(shapely.points(pb.easting.values, pb.northing.values))
        _, dd = t.query_nearest(shapely.points(pa.easting.values, pa.northing.values), return_distance=True)
        S.loc[pa.index, f"nn_{b}_m"] = dd
        _, dd2 = shapely.STRtree(shapely.points(pa.easting.values, pa.northing.values)).query_nearest(
            shapely.points(pb.easting.values, pb.northing.values), return_distance=True)
        S.loc[pb.index, f"nn_{a}_m"] = dd2
    S.to_csv(PROC / "sites_matched.csv", index=False)

    # ---------------- summaries
    rows, sweep = [], []
    near = cand.groupby("site").dist.min()
    for net in ["RSN", "MS", "MN_OTHER"]:
        for scope in ["all", "primary"]:
            d = S[S[f"is_{net}"] & (S[f"primary_{net}"] if scope == "primary" else True)]
            mt = d[d.cls != "U"]
            rows.append(dict(network=net, scope=scope, sites=len(d), matched=len(mt), match_rate=len(mt) / len(d),
                             cls_A=(d.cls == "A").sum(), cls_B=(d.cls == "B").sum(), cls_C=(d.cls == "C").sum(), cls_U=(d.cls == "U").sum(),
                             median_dist_m=mt.dist.median(), p90_dist_m=mt.dist.quantile(.9), p95_dist_m=mt.dist.quantile(.95),
                             p99_dist_m=mt.dist.quantile(.99), max_dist_m=mt.dist.max(),
                             ambiguous=int(d.ambiguous.sum()), resolved_by_name=int(d.resolved_by_name.sum()),
                             name_match=int(d.name_match.sum()), named_link=int(mt.link_watercourse_name.notna().sum()),
                             catchment_inconsistent=int((mt.catchment_consistent == False).sum()),
                             wma_inconsistent=int((d.wma_consistent == False).sum()),
                             on_inlandRiver=int((mt.link_form == "inlandRiver").sum()), on_tidal=int((mt.link_form == "tidalRiver").sum()),
                             on_lake=int((mt.link_form == "lake").sum()), on_canal=int((mt.link_form == "canal").sum()),
                             coord_res_100m=int(d.coord_res_100m.sum()), coord_res_10m=int(d.coord_res_10m.sum()),
                             size_ambiguous_alt_ratio_gt2=int(((d.ambiguous) & (d.size_ratio_alt > 2)).sum())))
            dn = near.reindex(d.site)
            for thr in [10, 25, 50, 75, 100, 150, 200, 250]:
                sweep.append(dict(network=net, scope=scope, threshold_m=thr, share_within=float((dn <= thr).mean())))
    pd.DataFrame(rows).to_csv(TAB / "p2_02_matching_summary.csv", index=False)
    pd.DataFrame(sweep).to_csv(TAB / "p2_02_threshold_sweep.csv", index=False)
    r = S[isr]
    info = dict(
        rsn_sites=int(isr.sum()),
        rsn_grts_to_wq_median_m=float(r.grts_to_wq_m.median()), rsn_grts_to_wq_p90_m=float(r.grts_to_wq_m.quantile(.9)),
        rsn_grts_to_wq_gt_500m=int((r.grts_to_wq_m > 500).sum()),
        rsn_grts_to_os_median_m=float(r.grts_to_os_network_m.median()), rsn_grts_to_os_p90_m=float(r.grts_to_os_network_m.quantile(.9)),
        rsn_grts_to_os_gt_100m=int((r.grts_to_os_network_m > 100).sum()),
        rsn_ard_wb_agree_point=float(r.ard_wb_consistent_pt.astype(float).mean()),
        rsn_ard_wb_agree_link=float(r[r.cls != "U"].ard_wb_consistent_link.astype(float).mean()),
        rsn_ms_colocated_lt_100m=int((S.nn_MS_m < 100).sum()), rsn_ms_colocated_lt_500m=int((S.nn_MS_m < 500).sum()),
        rsn_ms_same_link=int(len(set(S[S.is_RSN & (S.cls != "U")].link) & set(S[S.is_MS & (S.cls != "U")].link))),
        ms_and_mnother_same_point=int((S.is_MS & S.is_MN_OTHER).sum()),
        sites_outside_wfd=int(S.pt_water_body_id.isna().sum()),
    )
    dump_json(info, TAB / "p2_02_matching.json")
    print(pd.DataFrame(rows).T.to_string())
    for k, v in info.items():
        log(f"{k}: {v}")


if __name__ == "__main__":
    main()
