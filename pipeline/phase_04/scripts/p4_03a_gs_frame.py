"""Phase 4 A3 step a: the restricted river frame G_S with link covariates, and the tail-up Markov blanket of every G_S link
midpoint with respect to the 1,832 model sites. Read-only on Phases 2-3; outputs to the Phase 4 scratch directory and
phase_04/diagnostics.

Rules (identical to Phase 2 / Phase 3C):
  * G_S = F1_inland_river links inside the 16 WMAs with la_hat in [0.5471933374977596, 7.62846787157966] (Phase 2 support bounds);
    weight = len_england_m.
  * Midpoint = line_interpolate_point(geometry, 0.5, normalized); EDM 2024 outlets within 5 km of the midpoint (Phase 2 code);
    geology = typology_geology, NaN/Salt -> Unknown; z-standardisation with the frozen constants.
  * Region: nearest model site in the same WMA.
  * Active links = form != canal or canal_carries_flow; downstream = first out-link at the link's end node (P2B rule; p3c_01).
  * AFV at a point = up_chan_km_top + f * chan_km (f = fraction from the upstream end); the link midpoint has f = 0.5.
  * Tail-up DAG with l inserted: parents P_l = sites whose downstream path reaches l's midpoint before any other site;
    child d = first site downstream of l's midpoint; co-parents Q_d = model TU parents of d not in P_l.
    a_pl = exp(-h/rho) sqrt(r_pl), r_pl = min(1, afv_p/afv_l) normalised at l if sum > 1;
    a_ld = exp(-h/rho) sqrt(min(1, afv_l/afv_d) / n_d), with n_d the model's normalisation at d (so a_pl a_ld = model a_pd).
Verification: (i) rebuilt next-downstream site and h_next vs phase_03c rn_sites; (iii) EDM counts / z values for sites.
Outputs: scratch gs_links.parquet, gs_tu_blanket.npz; diagnostics/A3_frame_build.json
"""
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import p4_lib as P  # noqa: E402

TA = P.TA
P2 = TA / "phase_02/data_processed"
STD = json.load(open(TA / "phase_02b/specifications/standardisation_constants.json"))
GEO = ["Calcareous", "Siliceous", "Organic", "Unknown"]


def zc(x, key_mu, key_sd):
    return (x - STD[key_mu]) / STD[key_sd]


def main():
    import pyogrio
    import shapely
    t0 = time.time(); res = {}
    cfg = P.CFG["A3_river_length"]
    sites = pd.read_parquet(TA / "phase_03a/data_processed/sites.parquet").sort_values("site_idx").reset_index(drop=True)
    rn = pd.read_csv(TA / "phase_03c/data_processed/rn_sites.csv").set_index("site").loc[sites.site].reset_index()
    Lk = pd.read_parquet(P2 / "links_gb_attributes.parquet",
                         columns=["id", "u", "v", "start_node", "end_node", "length", "form", "canal_carries_flow", "chan_km", "up_chan_km_top",
                                  "F1_inland_river", "wma", "len_england_m", "typology_geology", "management_catchment", "water_body_id"])
    Z = pd.read_parquet(P2 / "links_size.parquet", columns=["id", "la_hat"])
    # ---------------- G_S frame
    F = Lk[Lk.F1_inland_river & Lk.wma.notna()].merge(Z, on="id", how="left")
    lo, hi = 0.5471933374977596, 7.62846787157966
    G = F[F.la_hat.between(lo, hi)].reset_index(drop=True)
    res["gs_links"] = len(G); res["gs_km"] = float(G.len_england_m.sum() / 1e3)
    geo = pyogrio.read_dataframe(P2 / "frame_links_england.gpkg", columns=["id"])
    geo = geo.set_index("id").loc[G.id]
    mid = shapely.line_interpolate_point(geo.geometry.values, 0.5, normalized=True)
    G["x_km"] = shapely.get_x(mid) / 1000; G["y_km"] = shapely.get_y(mid) / 1000
    edm = pd.read_csv(TA / "pilot_01/data/edm_outlets_2024.csv").dropna(subset=["easting", "northing"])
    tree = shapely.STRtree(shapely.points(edm.easting.values, edm.northing.values))
    i, _ = tree.query(mid, predicate="dwithin", distance=cfg["edm_radius_m"])
    G["edm_n_5km"] = np.bincount(i, minlength=len(G))
    G["z_logarea"] = zc(G.la_hat, "log_area_mean", "log_area_sd")
    G["z_edm"] = zc(np.log1p(G.edm_n_5km), "log1p_edm5_mean", "log1p_edm5_sd")
    g = G.typology_geology.fillna("Unknown").replace({"Salt": "Unknown"}); g = g.where(g.isin(GEO), "Unknown")
    G["geology"] = g; G["geology_idx"] = g.map({k: j + 1 for j, k in enumerate(GEO)})
    # (iii) verification on sites: EDM counts and z values recomputed with the same code from site coordinates
    sp = shapely.points(sites.x_km.values * 1000, sites.y_km.values * 1000)
    j, _ = tree.query(sp, predicate="dwithin", distance=cfg["edm_radius_m"])
    cnt = np.bincount(j, minlength=len(sites))
    res["verify_site_edm_counts_exact_share"] = float((cnt == sites.edm_n_5km.values).mean())
    res["verify_site_z_edm_maxabs"] = float(np.abs(zc(np.log1p(sites.edm_n_5km), "log1p_edm5_mean", "log1p_edm5_sd") - sites.z_edm).max())
    res["verify_site_z_logarea_maxabs"] = float(np.abs(zc(sites.log_area, "log_area_mean", "log_area_sd") - sites.z_logarea).max())
    # region: nearest site in the same WMA
    G["region_idx"] = 0
    for w, gw in G.groupby("wma"):
        sw = sites[sites.wma == w]
        d2 = (gw.x_km.values[:, None] - sw.x_km.values[None, :]) ** 2 + (gw.y_km.values[:, None] - sw.y_km.values[None, :]) ** 2
        G.loc[gw.index, "region_idx"] = sw.region_idx.values[d2.argmin(1)]
    res["region_assignment_min_idx"] = int(G.region_idx.min())
    # observed levels
    hi_ = P.B.hier_indices(sites)
    lv = {c: {x: k + 1 for k, x in enumerate(hi_[f"{c}_levels"])} for c in ["wma", "mancat", "wb"]}
    G["wma_idx"] = G.wma.map(lv["wma"]).astype(int)
    G["mancat_idx"] = G.management_catchment.map(lv["mancat"]).fillna(0).astype(int)     # 0 = new level (marginalised)
    G["wb_idx"] = G.water_body_id.map(lv["wb"]).fillna(0).astype(int)
    res["share_len_observed_mancat"] = float(G.len_england_m[G.mancat_idx > 0].sum() / G.len_england_m.sum())
    res["share_len_observed_wb"] = float(G.len_england_m[G.wb_idx > 0].sum() / G.len_england_m.sum())
    # ---------------- network (p3c_01 rules)
    act = Lk[(Lk.form != "canal") | Lk.canal_carries_flow].reset_index(drop=True)
    pos = {l: k for k, l in enumerate(act.id.values)}
    out_of = defaultdict(list)
    for k, uu in enumerate(act.u.values):
        out_of[uu].append(k)
    nA = len(act); nxt = -np.ones(nA, int)
    for k, vv in enumerate(act.v.values):
        o = out_of.get(vv, [])
        if o:
            nxt[k] = o[0]
    alen = act.length.values / 1000.0
    afv_top = act.up_chan_km_top.values; chan = act.chan_km.values
    res["gs_links_not_active"] = int((~G.id.isin(pos)).sum())
    # sites on links
    sk = np.array([pos[l] for l in rn.link_id], int); sf = rn.f_up.values; s_afv = rn.afv_km.values
    on = defaultdict(list)
    for s, (k, f) in enumerate(zip(sk, sf)):
        on[k].append((f, s))
    for k in on:
        on[k].sort()
    # first site downstream of the TOP of each active link: (site, dist km); iterative memo
    fs_site = np.full(nA, -2, int); fs_d = np.zeros(nA)
    for start in range(nA):
        if fs_site[start] != -2:
            continue
        stack = []; k = start; seen = set()
        while k >= 0 and fs_site[k] == -2 and k not in seen:
            seen.add(k)
            if k in on:
                f, s = on[k][0]; fs_site[k] = s; fs_d[k] = f * alen[k]; break
            stack.append(k); k = nxt[k]
        base_s, base_d = (fs_site[k], fs_d[k]) if (k >= 0 and fs_site[k] != -2) else (-1, 0.0)
        if k >= 0 and k in seen and fs_site[k] == -2:   # cycle guard
            base_s, base_d = -1, 0.0
        for kk in reversed(stack):
            if base_s >= 0:
                fs_site[kk] = base_s; fs_d[kk] = alen[kk] + base_d; base_d = fs_d[kk]
            else:
                fs_site[kk] = -1; fs_d[kk] = 0.0
    # (i) verification: next-downstream site and h_next for every site
    sid = {s: k for k, s in enumerate(sites.site)}
    nd_re, h_re = -np.ones(len(sites), int), np.full(len(sites), np.nan)
    for s, (k, f) in enumerate(zip(sk, sf)):
        later = [(f2, s2) for f2, s2 in on[k] if (f2 > f) or (f2 == f and sites.site[s2] > sites.site[s])]
        if later:
            nd_re[s] = later[0][1]; h_re[s] = (later[0][0] - f) * alen[k]
        elif nxt[k] >= 0 and fs_site[nxt[k]] >= 0:
            nd_re[s] = fs_site[nxt[k]]; h_re[s] = (1 - f) * alen[k] + fs_d[nxt[k]]
    nd_ref = np.array([sid[x] if isinstance(x, str) else -1 for x in rn.next_down_site], int)
    agree = nd_re == nd_ref
    res["verify_next_down_site_agreement"] = float(agree.mean())
    both = agree & (nd_ref >= 0)
    hd = np.abs(h_re[both] - rn.h_next_km.values[both])
    res["verify_h_next_share_within_1m"] = float((hd <= 0.001).mean()); res["verify_h_next_max_abs_km"] = float(hd.max())
    # ---------------- Markov blankets of G_S link midpoints
    gk = np.array([pos.get(l, -1) for l in G.id], int)
    g_afv = afv_top[np.maximum(gk, 0)] + 0.5 * chan[np.maximum(gk, 0)]
    child = -np.ones(len(G), int); h_ld = np.zeros(len(G))
    for g_, k in enumerate(gk):
        if k < 0:
            continue
        dn = [(f, s) for f, s in on.get(k, []) if f > 0.5]
        if dn:
            child[g_] = dn[0][1]; h_ld[g_] = (dn[0][0] - 0.5) * alen[k]
        elif nxt[k] >= 0 and fs_site[nxt[k]] >= 0:
            child[g_] = fs_site[nxt[k]]; h_ld[g_] = 0.5 * alen[k] + fs_d[nxt[k]]
    # parents: walk every site downstream until its next site; every G_S midpoint passed gets the site as a parent
    g_of = defaultdict(list)
    for g_, k in enumerate(gk):
        if k >= 0:
            g_of[k].append(g_)
    par = defaultdict(list)          # g -> list of (site, h_km)
    for s, (k, f) in enumerate(zip(sk, sf)):
        d_next = h_re[s] if nd_re[s] >= 0 else np.inf
        # own link: midpoint downstream of the site
        if f < 0.5 and k in g_of:
            h = (0.5 - f) * alen[k]
            if h < d_next:
                for g_ in g_of[k]:
                    par[g_].append((s, h))
        dist_top = (1 - f) * alen[k]; j = nxt[k]; seen = {k}
        while j >= 0 and j not in seen and dist_top < d_next:
            seen.add(j)
            if j in g_of:
                h = dist_top + 0.5 * alen[j]
                if h < d_next:
                    for g_ in g_of[j]:
                        par[g_].append((s, h))
            dist_top += alen[j]; j = nxt[j]
    # model TU parents of every site (for co-parents and the normalisation n_d)
    rs = P.C.rn_structures(sites)
    mpar = defaultdict(list)
    for c_, p_, h_, w_ in zip(rs["tu_child"], rs["tu_par"], rs["tu_h"], rs["tu_pw"]):
        mpar[int(c_)].append((int(p_), float(h_), float(w_)))
    nrm = np.ones(len(sites))
    for c_, lst in mpar.items():
        raw = [min(1.0, s_afv[p_] / s_afv[c_]) for p_, _, _ in lst]
        nrm[c_] = max(1.0, sum(raw))
    # consistency: each parent's next site equals the link's child
    incons = 0
    rows_p, rows_q = [], []
    anom_l = np.zeros(len(G), bool); anom_d = np.zeros(len(G), bool); rld_arr = np.zeros(len(G))
    for g_ in range(len(G)):
        Pl = par.get(g_, [])
        d = child[g_]
        for s, h in Pl:
            if nd_re[s] != d:
                incons += 1
        rr = np.array([min(1.0, s_afv[s] / g_afv[g_]) for s, _ in Pl]) if Pl else np.zeros(0)
        nl = max(1.0, rr.sum()) if len(rr) else 1.0
        anom_l[g_] = rr.sum() > 1 + 1e-9
        for (s, h), r in zip(Pl, rr):
            rows_p.append((g_, s, h, r / nl))
        if d >= 0:
            pset = {s for s, _ in Pl}
            Q = [(q, hq, wq) for q, hq, wq in mpar.get(d, []) if q not in pset]
            raw_ld = min(1.0, g_afv[g_] / s_afv[d]); raw_q = [min(1.0, s_afv[q] / s_afv[d]) for q, _, _ in Q]
            tot = raw_ld + sum(raw_q)
            if tot <= 1 + 1e-9 and nrm[d] <= 1 + 1e-12:      # consistent: model weights (= raw ratios here)
                rld_arr[g_] = raw_ld
                for q, hq, wq in Q:
                    rows_q.append((g_, q, hq, wq))
            else:                                             # AFV non-additive (bifurcation / parallel channel): joint normalisation at d
                anom_d[g_] = True; n_ = max(1.0, tot)
                rld_arr[g_] = raw_ld / n_
                for (q, hq, wq), r in zip(Q, raw_q):
                    rows_q.append((g_, q, hq, r / n_))
    res["parent_child_inconsistencies"] = incons
    Pp = np.array(rows_p) if rows_p else np.zeros((0, 4)); Qq = np.array(rows_q) if rows_q else np.zeros((0, 4))
    G["afv_km"] = g_afv; G["tu_child"] = child; G["tu_h_child_km"] = h_ld; G["tu_r_child"] = rld_arr
    G["tu_afv_anomaly"] = anom_l | anom_d
    res["afv_anomaly_links"] = int((anom_l | anom_d).sum()); res["afv_anomaly_share_length"] = float(G.len_england_m[anom_l | anom_d].sum() / G.len_england_m.sum())
    G["n_tu_parents"] = np.bincount(Pp[:, 0].astype(int), minlength=len(G)) if len(Pp) else 0
    G["flow_connected_site"] = (G.tu_child >= 0) | (G.n_tu_parents > 0)
    # support cells without RSN sites (WMA x Phase 2 size class of the link)
    S2 = pd.read_parquet(P2 / "links_size.parquet", columns=["id", "size_hat"])
    G = G.merge(S2, on="id", how="left")
    rsn = sites[sites.network == "RSN"]
    cells = set(zip(rsn.wma, rsn.size_cls))
    G["rsn_supported_cell"] = [(w, str(s)) in cells or (w, s) in cells for w, s in zip(G.wma, G.size_hat)]
    res["share_len_flow_connected"] = float(G.len_england_m[G.flow_connected_site].sum() / G.len_england_m.sum())
    res["km_without_rsn_support"] = float(G.len_england_m[~G.rsn_supported_cell].sum() / 1e3)
    res["size_class_levels_sites"] = sorted(map(str, set(sites.size_cls))); res["size_class_levels_links"] = sorted(map(str, set(G.size_hat.astype(str))))
    sc = P.scratch()
    G.drop(columns=["u", "v", "start_node", "end_node"], errors="ignore").to_parquet(sc / "gs_links.parquet")
    np.savez(sc / "gs_tu_blanket.npz", parents=Pp, coparents=Qq)
    res["n_parent_edges"] = len(Pp); res["n_coparent_edges"] = len(Qq); res["seconds"] = time.time() - t0
    P.save_json(res, P.P4 / "diagnostics/A3_frame_build.json")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
