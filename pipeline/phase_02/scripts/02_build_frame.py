"""O1: construct the river-network frame from OS Open Rivers (2026-10, GB) + WFD Cycle 3 catchments.

Outputs
  data_processed/links_gb_attributes.parquet   all GB links, topology-derived attributes (no geometry)
  data_processed/frame_links_england.gpkg      England links with geometry + frame attributes
  tables/p2_02_frame_qa.json                   integrity / topology diagnostics
  tables/p2_02_frame_length_by_form.csv        length by form x frame
Run from phase_02/:  python -I scripts/02_build_frame.py

Design choices (documented in reports/P2_02):
  * Directed graph from start_node -> end_node, reversed for 'in opposite direction'; 52 links with
    null flow_direction keep digitisation order and are flagged.
  * Canal links are EXCLUDED from flow accumulation (they can join separate basins artificially) but
    kept in the geometry with form='canal'.
  * Upstream CHANNEL length counts inlandRiver + tidalRiver link length; lake centrelines carry flow
    (connectivity) but contribute 0 km of channel length.
  * Bifurcations: primary accumulation splits upstream length equally among outgoing links (mass
    conserving: no double counting when anabranches re-join); the 'no-split' variant (each branch
    inherits the full upstream total) is kept as a sensitivity attribute.
  * Strahler order: standard rule on the DAG; Shreve magnitude with equal splitting.
  * Catchment / area membership by link midpoint (shapely line_interpolate_point at 0.5 normalised);
    links whose geometry crosses a WFD catchment boundary are flagged and their England-clipped
    length is computed exactly where they cross the England boundary.
"""
import sys, time
sys.path.insert(0, __import__("os").path.dirname(__file__))
from common import *  # noqa

import geopandas as gpd
import networkx as nx
import pyogrio
import shapely


def accumulate(L, canal_active=None):
    """Return dict of per-link arrays: up_len_top (km of channel above the link's upstream end),
    up_len_bot (at downstream end), same for no-split variant, strahler, shreve, comp ids."""
    n = len(L)
    u = L["u"].values; v = L["v"].values
    canal_active = np.zeros(n, bool) if canal_active is None else np.asarray(canal_active, bool)
    chan_canal = canal_active & L.get("watercourse_name", pd.Series([""] * n)).fillna("").str.startswith("River").values
    chan = np.where(L.form.isin(["inlandRiver", "tidalRiver"]).values | chan_canal, L.length.values, 0.0) / 1000.0
    active = (L.form != "canal").values | canal_active
    G = nx.MultiDiGraph()
    G.add_nodes_from(np.unique(np.r_[u, v]))
    idx_active = np.flatnonzero(active)
    G.add_edges_from((u[i], v[i], {"i": int(i)}) for i in idx_active)
    C = nx.condensation(nx.DiGraph(G))          # SCCs collapsed (DiGraph sufficient for SCC structure)
    cmap = C.graph["mapping"]
    cu = np.array([cmap[x] for x in u]); cv = np.array([cmap[x] for x in v])
    order = list(nx.topological_sort(C))
    rank = np.empty(len(order), int); rank[order] = np.arange(len(order))
    internal = active & (cu == cv)
    # outgoing (external) active links per component
    out_links = {}
    for i in idx_active:
        if cu[i] != cv[i]:
            out_links.setdefault(cu[i], []).append(i)
    inflow = np.zeros(len(order)); inflow_ns = np.zeros(len(order)); shreve_in = np.zeros(len(order))
    has_in = np.zeros(len(order), bool)
    str_in = {}                                    # comp -> list of incoming strahler orders
    for i in np.flatnonzero(internal):             # internal SCC links: add their own length
        inflow[cu[i]] += chan[i]; inflow_ns[cu[i]] += chan[i]
    up_top = np.full(n, np.nan); up_bot = np.full(n, np.nan)
    up_top_ns = np.full(n, np.nan); up_bot_ns = np.full(n, np.nan)
    strahler = np.zeros(n, int); shreve = np.full(n, np.nan); split = np.zeros(n, bool)
    for c in order:
        outs = out_links.get(c, [])
        k = len(outs)
        ins = str_in.get(c, [])
        if ins:
            m = max(ins); so = m + 1 if ins.count(m) >= 2 else m
        else:
            so = 1
        sh = shreve_in[c] if has_in[c] else 1.0
        for i in outs:
            up_top[i] = inflow[c] / k; up_bot[i] = up_top[i] + chan[i]
            up_top_ns[i] = inflow_ns[c]; up_bot_ns[i] = up_top_ns[i] + chan[i]
            strahler[i] = so; shreve[i] = sh / k; split[i] = k > 1
            t = cv[i]
            inflow[t] += up_bot[i]; inflow_ns[t] += up_bot_ns[i]; shreve_in[t] += shreve[i]; has_in[t] = True
            str_in.setdefault(t, []).append(so)
    for i in np.flatnonzero(internal):
        c = cu[i]
        up_top[i] = inflow[c] - chan[i]; up_bot[i] = inflow[c]
        up_top_ns[i] = inflow_ns[c] - chan[i]; up_bot_ns[i] = inflow_ns[c]
        ins = str_in.get(c, [1]); strahler[i] = max(ins); shreve[i] = shreve_in[c] if has_in[c] else 1
    # weak components on the active graph
    wcc = np.full(n, -1)
    node_comp = {}
    comps = sorted(nx.weakly_connected_components(G), key=lambda s: min(s))   # deterministic ids
    for k_, comp in enumerate(comps):
        for x in comp:
            node_comp[x] = k_
    wcc[idx_active] = [node_comp[u[i]] for i in idx_active]
    n_scc = sum(1 for _, d in C.nodes(data=True) if len(d["members"]) > 1)
    # downstream-of-split flag propagates: any link whose upstream path includes a split
    return dict(up_top=up_top, up_bot=up_bot, up_top_ns=up_top_ns, up_bot_ns=up_bot_ns,
                strahler=strahler, shreve=shreve, at_split=split, internal_scc=internal,
                wcc=wcc, n_nontrivial_scc=n_scc, chan_km=chan)


def _is_bad(c, ii, oo):
    return (c == "none" and not (ii == 1 and oo == 1)) or (c == "junction" and (ii == 0 or oo == 0)) \
        or (c == "source" and ii > 0) or (c == "outlet" and oo > 0)


def node_anomalies(u, v, cat):
    """Nodes whose in/out degree contradicts the OS hydro_node_category:
    pseudo-node ('none') not 1-in/1-out; junction with no inflow or no outflow; source with inflow; outlet with outflow."""
    from collections import Counter
    o = Counter(u); i = Counter(v)
    return {n for n in set(u) | set(v) if _is_bad(cat.get(n, "none"), i.get(n, 0), o.get(n, 0))}


def repair_direction(L, cat, max_rounds=50):
    """Conservative local repair of digitisation-direction errors.
    A non-canal link is re-oriented only if doing so STRICTLY reduces the number of category-inconsistent
    endpoints (node_anomalies) at its two end nodes. Among improving flips touching the same anomaly the
    largest reduction wins; remaining ties are broken by the undirected network distance to the nearest OS
    'outlet' node (prefer the orientation that points towards it). Greedy, deterministic, iterated to a fixed
    point. Links whose flip would not improve consistency are never changed, and the published direction is
    retained in up_chan_km_mid_osdir / strahler_osdir for sensitivity analysis."""
    from collections import Counter
    u = L.u.values.copy(); v = L.v.values.copy()
    noncanal = (L.form != "canal").values
    Gu = nx.Graph()
    for a_, b_, w in zip(u[noncanal], v[noncanal], L.length.values[noncanal]):
        if Gu.has_edge(a_, b_):
            w = min(w, Gu[a_][b_]["weight"])
        Gu.add_edge(a_, b_, weight=w)
    dist = nx.multi_source_dijkstra_path_length(Gu, [n for n in Gu.nodes if cat.get(n) == "outlet"], weight="weight")
    o = Counter(u); i = Counter(v)
    inc = {}
    for k in np.flatnonzero(noncanal):
        inc.setdefault(u[k], []).append(k); inc.setdefault(v[k], []).append(k)
    bad = lambda n: _is_bad(cat.get(n, "none"), i.get(n, 0), o.get(n, 0))
    info = dict(anomalous_nodes_before=len(node_anomalies(u, v, cat)), rounds=[])
    flipped = np.zeros(len(L), bool)
    for r in range(max_rounds):
        nflip = 0
        for n in sorted(node_anomalies(u, v, cat)):
            if not bad(n):
                continue
            best = None
            for k in inc.get(n, []):
                a_, b_ = u[k], v[k]
                before = bad(a_) + bad(b_)
                o[a_] -= 1; i[b_] -= 1; o[b_] += 1; i[a_] += 1          # tentative flip a->b  =>  b->a
                after = bad(a_) + bad(b_)
                o[a_] += 1; i[b_] += 1; o[b_] -= 1; i[a_] -= 1
                gain = before - after
                if gain > 0:
                    toward = dist.get(a_, np.inf) < dist.get(b_, np.inf)   # flipped link would point to a_ (closer to outlet)
                    key = (gain, toward, -k)
                    if best is None or key > best[0]:
                        best = (key, k)
            if best is not None:
                k = best[1]; a_, b_ = u[k], v[k]
                o[a_] -= 1; i[b_] -= 1; o[b_] += 1; i[a_] += 1
                u[k], v[k] = b_, a_
                flipped[k] = ~flipped[k]; nflip += 1
        info["rounds"].append(dict(round=r + 1, flips=nflip))
        if nflip == 0:
            break
    info["links_reoriented"] = int(flipped.sum())
    info["km_reoriented"] = float(L.length.values[flipped].sum() / 1e3)
    info["anomalous_nodes_after"] = len(node_anomalies(u, v, cat))
    return u, v, info


def main():
    ensure_dirs()
    t0 = time.time()
    L = pyogrio.read_dataframe(OS_GPKG, layer="watercourse_link")
    nodes = pyogrio.read_dataframe(OS_GPKG, layer="hydro_node")
    log(f"links {len(L)}, nodes {len(nodes)}  ({time.time()-t0:.1f}s)")
    qa = dict(source="OS Open Rivers 2026-10 GB GeoPackage", n_links_gb=len(L), n_nodes_gb=len(nodes))

    # ---------------- integrity
    geom = L.geometry.values
    qa["invalid_geometries"] = int((~shapely.is_valid(geom)).sum())
    qa["empty_geometries"] = int(shapely.is_empty(geom).sum())
    qa["duplicate_ids"] = int(L.id.duplicated().sum())
    qa["duplicate_geometries"] = int(pd.Series(shapely.to_wkb(shapely.normalize(geom))).duplicated().sum())
    # NB: on a GeoDataFrame `L.length` is the GEOMETRY length; the OS attribute is the column L["length"]
    qa["length_attr_minus_geom_max_abs_m"] = float(np.abs(L["length"].values - shapely.length(geom)).max())
    qa["fictitious_true"] = int((L.fictitious.astype(int) == 1).sum())
    qa["form_counts"] = L.form.value_counts().to_dict()
    qa["flow_direction_counts"] = L.flow_direction.fillna("NULL").value_counts().to_dict()
    qa["hydro_node_category_counts"] = nodes.hydro_node_category.fillna("NULL").value_counts().to_dict()
    # node coordinates agree with link end points?
    nxy = dict(zip(nodes.id, zip(shapely.get_x(nodes.geometry.values), shapely.get_y(nodes.geometry.values))))
    sx = shapely.get_x(shapely.get_point(geom, 0)); sy_ = shapely.get_y(shapely.get_point(geom, 0))
    ex = shapely.get_x(shapely.get_point(geom, -1)); ey = shapely.get_y(shapely.get_point(geom, -1))
    snx = np.array([nxy[s][0] for s in L.start_node]); sny = np.array([nxy[s][1] for s in L.start_node])
    enx = np.array([nxy[s][0] for s in L.end_node]); eny = np.array([nxy[s][1] for s in L.end_node])
    qa["start_node_coord_mismatch_gt_1m"] = int((np.hypot(sx - snx, sy_ - sny) > 1).sum())
    qa["end_node_coord_mismatch_gt_1m"] = int((np.hypot(ex - enx, ey - eny) > 1).sum())

    # ---------------- direction
    rev = (L.flow_direction == "in opposite direction").values
    L["u"] = np.where(rev, L.end_node, L.start_node)
    L["v"] = np.where(rev, L.start_node, L.end_node)
    L["flag_flowdir_missing"] = L.flow_direction.isna()
    # accumulation with the published OS direction (kept as a sensitivity attribute)
    # canal links that carry river flow: river navigations / lock cuts by name, or canal links joining river
    # nodes at both ends (rule chosen against NRFA gauged areas, see tables/p2_03_canal_rule_experiment.csv)
    rn = set(L.loc[L.form == "inlandRiver", "start_node"]) | set(L.loc[L.form == "inlandRiver", "end_node"])
    L["canal_carries_flow"] = (L.form == "canal") & (L.watercourse_name.fillna("").str.contains(r"River|Navigation|Cut", regex=True)
                                                    | (L.start_node.isin(rn) & L.end_node.isin(rn)))
    qa["canal_links_carrying_flow"] = int(L.canal_carries_flow.sum())
    acc_os = accumulate(L, L.canal_carries_flow.values)
    L["up_chan_km_mid_osdir"] = (acc_os["up_top"] + acc_os["up_bot"]) / 2
    L["strahler_osdir"] = acc_os["strahler"]
    # conservative direction repair at topologically inconsistent nodes (see repair_direction docstring)
    cat = dict(zip(nodes.id, nodes.hydro_node_category.fillna("none")))
    u2, v2, rinfo = repair_direction(L, cat)
    L["flag_direction_repaired"] = (u2 != L.u.values)
    L["u"], L["v"] = u2, v2
    qa["direction_repair"] = rinfo
    pair = L.u + "|" + L.v
    L["flag_parallel_link"] = pair.duplicated(keep=False).values
    qa["parallel_links"] = int(L.flag_parallel_link.sum())

    acc = accumulate(L, L.canal_carries_flow.values)
    L["chan_km"] = acc["chan_km"]
    L["up_chan_km_top"] = acc["up_top"]; L["up_chan_km_bot"] = acc["up_bot"]
    L["up_chan_km_top_nosplit"] = acc["up_top_ns"]; L["up_chan_km_bot_nosplit"] = acc["up_bot_ns"]
    L["up_chan_km_mid"] = (L.up_chan_km_top + L.up_chan_km_bot) / 2
    L["strahler"] = acc["strahler"]; L["shreve"] = acc["shreve"]
    L["flag_bifurcation_child"] = acc["at_split"]; L["flag_in_cycle"] = acc["internal_scc"]
    L["component"] = acc["wcc"]
    qa["nontrivial_scc"] = int(acc["n_nontrivial_scc"])
    qa["bifurcation_child_links"] = int(acc["at_split"].sum())
    # downstream-of-bifurcation propagation is captured by the split/no-split ratio
    L["split_ratio"] = L.up_chan_km_bot / L.up_chan_km_bot_nosplit
    log(f"accumulation done ({time.time()-t0:.1f}s)")

    # ---------------- WFD catchments, WMAs, England mask
    W = gpd.read_file(WFD_GPKG, engine="pyogrio")[["water_body_id", "water_body_name", "water_body_type", "water_body_typology",
                                                    "water_body_hmd", "operational_catchment", "management_catchment",
                                                    "ea_area_name", "river_basin_district", "surveillance_water_body", "geometry"]]
    W["wb_area_km2"] = W.geometry.area / 1e6
    A = gpd.read_file(WMA_GPKG, engine="pyogrio")[["long_name", "code", "reg_name", "geometry"]].rename(columns={"long_name": "wma", "code": "wma_code", "reg_name": "wma_region"})
    qa["wfd_catchments"] = len(W); qa["wfd_invalid"] = int((~W.is_valid).sum())
    qa["wma_polygons"] = len(A); qa["wma_invalid"] = int((~A.is_valid).sum())
    W["geometry"] = W.geometry.make_valid(); A["geometry"] = A.geometry.make_valid()
    # overlaps between WFD catchments (should be a partition): sum of areas minus area of union
    t1 = time.time()
    eng = shapely.union_all(W.geometry.values)
    qa["wfd_sum_area_km2"] = float(W.geometry.area.sum() / 1e6)
    qa["wfd_union_area_km2"] = float(eng.area / 1e6)
    qa["wfd_overlap_area_km2"] = qa["wfd_sum_area_km2"] - qa["wfd_union_area_km2"]
    log(f"union done ({time.time()-t1:.1f}s)")

    mid = shapely.line_interpolate_point(geom, 0.5, normalized=True)
    Pm = gpd.GeoDataFrame(geometry=mid, crs=CRS)
    j = gpd.sjoin(Pm, W, how="left", predicate="within")
    j = j[~j.index.duplicated(keep="first")]
    for c in ["water_body_id", "water_body_name", "water_body_type", "water_body_typology", "water_body_hmd",
              "operational_catchment", "management_catchment", "ea_area_name", "river_basin_district",
              "surveillance_water_body", "wb_area_km2"]:
        L[c] = j[c].values
    ja = gpd.sjoin(Pm, A, how="left", predicate="within"); ja = ja[~ja.index.duplicated(keep="first")]
    L["wma"] = ja["wma"].values; L["wma_code"] = ja["wma_code"].values
    L["in_england"] = L.water_body_id.notna()
    # links crossing catchment boundaries (tree of polygons queried with each link: polygon side is prepared)
    t1 = time.time()
    ptree = shapely.STRtree(W.geometry.values)
    li, wi = ptree.query(geom, predicate="intersects")
    ncat = np.bincount(li, minlength=len(L))
    L["n_catchments_touched"] = ncat
    L["flag_crosses_catchment"] = ncat > 1
    log(f"catchment-crossing done ({time.time()-t1:.1f}s)")
    # exact England-clipped length for links not wholly inside one catchment: sum of pieces inside each
    # touched catchment (catchments form a partition, overlap checked above)
    eng_b = shapely.boundary(eng); shapely.prepare(eng_b)
    crosses = shapely.intersects(eng_b, geom)
    L["len_england_m"] = np.where(L.in_england, L.length, 0.0)
    sel = np.isin(li, np.flatnonzero(crosses))
    pieces = shapely.length(shapely.intersection(geom[li[sel]], W.geometry.values[wi[sel]]))
    clip = np.bincount(li[sel], weights=pieces, minlength=len(L))
    L.loc[crosses, "len_england_m"] = np.minimum(clip[crosses], L.length.values[crosses])
    log(f"england clip done ({time.time()-t1:.1f}s)")
    L["flag_crosses_england_boundary"] = crosses
    qa["links_crossing_england_boundary"] = int(crosses.sum())
    qa["england_len_midpoint_rule_km"] = float(L.loc[L.in_england, "length"].sum() / 1e3)
    qa["england_len_exact_clip_km"] = float(L.len_england_m.sum() / 1e3)
    qa["wma_vs_wfd_area_agreement"] = float((L.loc[L.in_england, "wma"] == L.loc[L.in_england, "ea_area_name"]).mean())
    qa["england_links_without_wma"] = int((L.in_england & L.wma.isna()).sum())

    # ---------------- components / boundary effects
    comp_len = L.groupby("component").length.sum() / 1e3
    # cross-border share uses inland-river links only: tidal/estuary links lie outside river-catchment polygons
    inl = L[L.form == "inlandRiver"]
    comp_eng_share = inl.groupby("component").len_england_m.sum() / inl.groupby("component").length.sum()
    qa["tidal_links_outside_wfd_river_catchments_km"] = float(L.loc[(L.form == "tidalRiver") & ~L.in_england, "length"].sum() / 1e3)
    L["component_len_km"] = L.component.map(comp_len)
    L["component_england_share"] = L.component.map(comp_eng_share)
    L["flag_crossborder_basin"] = L.in_england & (L.component_england_share < 0.995) & (L.component >= 0)
    eng_links = L[L.in_england]
    qa["england_components"] = int(eng_links.component.nunique())
    qa["england_links_in_components_lt_5km"] = int((eng_links.component_len_km < 5).sum())
    qa["england_len_in_components_lt_5km_km"] = float(eng_links.loc[eng_links.component_len_km < 5, "length"].sum() / 1e3)
    qa["england_len_crossborder_basins_km"] = float(eng_links.loc[eng_links.flag_crossborder_basin, "length"].sum() / 1e3)

    # ---------------- frame definitions
    L["F0_all_england"] = L.in_england
    L["F1_inland_river"] = L.in_england & (L.form == "inlandRiver")
    L["F2_river_wb_catchment"] = L.F1_inland_river & (L.water_body_type == "River")
    L["typology_size"] = L.water_body_typology.str.split(", ").str[1].map(TYPO_SIZE_MAP)
    L["typology_geology"] = L.water_body_typology.str.split(", ").str[2]
    L["typology_altitude"] = L.water_body_typology.str.split(", ").str[0]
    L["source_dataset"] = "OS Open Rivers 2026-10"

    rows = []
    for fr in ["F0_all_england", "F1_inland_river", "F2_river_wb_catchment"]:
        for form, g in L[L[fr]].groupby("form"):
            rows.append(dict(frame=fr, form=form, links=len(g), length_km=g.len_england_m.sum() / 1e3))
    pd.DataFrame(rows).to_csv(TAB / "p2_02_frame_length_by_form.csv", index=False)
    for fr in ["F0_all_england", "F1_inland_river", "F2_river_wb_catchment"]:
        qa[f"{fr}_links"] = int(L[fr].sum()); qa[f"{fr}_km"] = float(L.loc[L[fr], "len_england_m"].sum() / 1e3)
    qa["england_named_share_F1"] = float(L.loc[L.F1_inland_river, "watercourse_name"].notna().mean())
    qa["england_strahler_counts_F1"] = L[L.F1_inland_river].strahler.value_counts().sort_index().to_dict()

    attrs = L.drop(columns="geometry")
    attrs.to_parquet(PROC / "links_gb_attributes.parquet", index=False)
    E = gpd.GeoDataFrame(L[L.in_england | (L.len_england_m > 0)].copy(), geometry="geometry", crs=CRS)
    keep = ["id", "form", "watercourse_name", "length", "len_england_m", "up_chan_km_top", "up_chan_km_bot", "up_chan_km_mid",
            "strahler", "shreve", "water_body_id", "water_body_type", "typology_size", "wma", "F0_all_england",
            "F1_inland_river", "F2_river_wb_catchment", "flag_flowdir_missing", "flag_parallel_link", "flag_bifurcation_child",
            "flag_crosses_catchment", "flag_crossborder_basin", "component", "geometry"]
    E[keep].to_file(PROC / "frame_links_england.gpkg", driver="GPKG", engine="pyogrio")
    W.drop(columns="geometry").to_csv(PROC / "wfd_catchments_attributes.csv", index=False)
    qa["runtime_s"] = time.time() - t0
    dump_json(qa, TAB / "p2_02_frame_qa.json")
    for k, v in qa.items():
        log(f"{k}: {v}")


if __name__ == "__main__":
    main()
