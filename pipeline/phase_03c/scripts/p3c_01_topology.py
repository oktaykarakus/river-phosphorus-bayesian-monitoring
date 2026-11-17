"""Phase 3C Stage 3a: river-network topology for the 1,832 model sites (read-only use of the frozen Phase 2 network).

Re-traces every site's downstream path on the Phase 2 directed link graph with exactly the Phase 2B rule (P2B_02:
active links = non-canal or flow-carrying canal; first out-link at bifurcations), but now with
  * the site's position WITHIN its link (Phase 2 `frac`, oriented to flow),
  * exact along-river distances h (site to site, not link-start to link-start),
  * the reduced tree (sites + confluence junctions) needed for tail-down and flow-unconnected distances,
  * per-pair path-quality flags (repaired link, anomalous node, bifurcation, cycle guard),
  * the additive function value (AFV) = upstream channel length at the site, for tail-up weights.
Outputs (phase_03c/data_processed/): rn_sites.csv, rn_tree_nodes.csv, rn_pairs_connected.csv, rn_pairs_unconnected.parquet
and diagnostics/rn_topology.json.
Run from phase_03c/:  ../phase_03a/environment/venv/bin/python -I -W ignore scripts/p3c_01_topology.py
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TA = ROOT.parent
P2 = TA / "phase_02"
OUT = ROOT / "data_processed"; OUT.mkdir(exist_ok=True)
DIAG = ROOT / "diagnostics"; DIAG.mkdir(exist_ok=True)


def is_bad(c, ii, oo):          # identical to phase_02/scripts/02_build_frame.py::_is_bad
    return (c == "none" and not (ii == 1 and oo == 1)) or (c == "junction" and (ii == 0 or oo == 0)) \
        or (c == "source" and ii > 0) or (c == "outlet" and oo > 0)


def main():
    import pyogrio, shapely
    L = pd.read_parquet(P2 / "data_processed/links_gb_attributes.parquet",
                        columns=["id", "u", "v", "start_node", "end_node", "length", "form", "canal_carries_flow", "flag_direction_repaired",
                                 "component", "chan_km", "up_chan_km_top", "up_chan_km_bot", "flag_bifurcation_child", "flag_in_cycle",
                                 "flag_crossborder_basin"])
    sites = pd.read_parquet(TA / "phase_03a/data_processed/sites.parquet")
    SM = pd.read_csv(P2 / "data_processed/sites_matched.csv", usecols=["site", "link_id", "frac", "cls", "dist"])
    S = sites[["site_idx", "site", "network", "region", "mancat", "wb", "x_km", "y_km"]].merge(SM[["site", "link_id", "frac"]], on="site", how="left")
    assert len(S) == len(sites) and S.link_id.notna().all(), "every model site must be matched"
    res = {"n_sites": len(S)}

    # ---- orientation of `frac` (shapely line_locate_point along the OS geometry, start_node -> end_node).
    # Check the convention: for consecutive links L1 -> L2 (v1 = u2) the shared node must be the flow-downstream end of the
    # L1 geometry and the flow-upstream end of the L2 geometry.
    act = L[(L.form != "canal") | L.canal_carries_flow].reset_index(drop=True)
    G = pyogrio.read_dataframe(P2 / "data_processed/frame_links_england.gpkg", columns=["id"])
    geo = dict(zip(G.id, G.geometry.values))
    pos = {l: i for i, l in enumerate(act.id.values)}
    rows = act[act.id.isin(geo.keys())].sample(4000, random_state=1)
    out_of = defaultdict(list)
    for idx, uu in enumerate(act.u.values):
        out_of[uu].append(idx)
    mism = []; checked = 0
    for _, r in rows.iterrows():
        nx_ = [k for k in out_of.get(r.v, []) if act.id.values[k] in geo]
        if not nx_:
            continue
        r2 = act.iloc[nx_[0]]
        g1, g2 = geo[r.id], geo[r2.id]
        c1 = np.asarray(g1.coords); c2 = np.asarray(g2.coords)
        end1 = c1[-1] if r.u == r.start_node else c1[0]          # downstream end of L1 under the convention
        top2 = c2[0] if r2.u == r2.start_node else c2[-1]        # upstream end of L2 under the convention
        mism.append(float(np.hypot(*(end1 - top2)))); checked += 1
    mism = np.array(mism)
    res["orientation_check"] = {"pairs_checked": checked, "share_shared_node_within_1m": float((mism < 1).mean()),
                                "max_gap_m": float(mism.max())}
    print("orientation check", res["orientation_check"])
    a_link = act.set_index("id")
    lk = a_link.loc[S.link_id.values]
    S["f_up"] = np.where(lk.u.values == lk.start_node.values, S.frac.values, 1 - S.frac.values)   # fraction from the upstream end
    S["link_len_m"] = lk.length.values
    S["afv_km"] = lk.up_chan_km_top.values + S.f_up.values * lk.chan_km.values                     # upstream channel length at site
    S["component"] = lk.component.values
    S["flag_crossborder"] = lk.flag_crossborder_basin.values.astype(bool)

    # ---- anomalous nodes (P2B rule, OS hydro_node categories from the cached byte-identical copy)
    sys.path.insert(0, str(P2 / "scripts"))
    from common import OS_GPKG
    Nid = pyogrio.read_dataframe(OS_GPKG, layer="hydro_node", read_geometry=False)
    cat = dict(zip(Nid.id, Nid.hydro_node_category.fillna("none")))
    o_ = Counter(L.u.values); i_ = Counter(L.v.values)
    bad = {n for n in set(L.u.values) | set(L.v.values) if is_bad(cat.get(n, "none"), i_.get(n, 0), o_.get(n, 0))}
    res["remaining_anomalous_nodes"] = len(bad)

    # ---- trace downstream paths (P2B rule: first out-link at bifurcations)
    act_id = act.id.values; act_v = act.v.values; act_len = act.length.values; act_rep = act.flag_direction_repaired.values.astype(bool)
    nxt_of = {}
    paths = {}
    stats = Counter()
    for k, (s_, l_, f_) in enumerate(zip(S.site, S.link_id, S.f_up)):
        i = pos[l_]; seq = []; seen = set()
        d_top = -f_ * act_len[i]                 # distance from site DOWN to the top of the current link (negative on own link)
        cum = dict(rep=0, anom=0, bif=0)
        while True:
            seq.append((act_id[i], d_top, cum["rep"], cum["anom"], cum["bif"]))
            seen.add(i)
            cum["rep"] += int(act_rep[i])
            nx_ = out_of.get(act_v[i], [])
            if act_v[i] in bad:
                cum["anom"] += 1
            if len(nx_) > 1:
                cum["bif"] += 1
            if not nx_:
                break
            j = nx_[0]
            if j in seen:
                stats["cycle_stops"] += 1; break
            nxt_of[act_id[i]] = act_id[j]
            d_top += act_len[i]; i = j
        paths[s_] = seq
    res["path_links_median"] = float(np.median([len(p) for p in paths.values()]))
    res["cycle_stops"] = int(stats["cycle_stops"])

    # ---- reduced tree: sites + junctions (top of a link where >= 2 distinct incoming path links meet)
    incoming = defaultdict(set); on_link = defaultdict(list)
    for s_, seq in paths.items():
        for (l1, *_), (l2, *_) in zip(seq[:-1], seq[1:]):
            incoming[l2].add(l1)
    for s_, l_, f_ in zip(S.site, S.link_id, S.f_up):
        on_link[l_].append((f_, s_))
    junction_links = {l for l, inc in incoming.items() if len(inc) >= 2}
    res["junctions"] = len(junction_links)
    # nodes on each link ordered by position: junction at f = 0 (top), then sites by f_up (ties by name)
    nodes_on = {}
    for l in set(on_link) | junction_links:
        lst = ([(-1.0, f"J:{l}")] if l in junction_links else []) + sorted(on_link.get(l, []))
        nodes_on[l] = [(max(f, 0.0), n) for f, n in lst]
    llen = dict(zip(act_id, act_len))
    # walk each site's path to build node chains (with distances from the site) -> parent of every node
    parent, pdist, erep, ebif = {}, {}, {}, {}

    def chain_from(start_site):
        seq = paths[start_site]
        f0 = S.f_up.values[S.site.values == start_site][0]
        out = []
        for (l, d_top, rep, anom, bif) in seq:
            for f, n in nodes_on.get(l, []):
                d = d_top + f * llen[l]
                if l == seq[0][0]:
                    if n == start_site or f < f0 or (f == f0 and n < start_site and not n.startswith("J:")):
                        continue
                    if n.startswith("J:"):
                        continue
                out.append((n, d, rep, anom, bif))
        return out

    chains = {s_: chain_from(s_) for s_ in S.site}
    # junction chains: from the junction at the top of link l, downstream
    for s_, ch in chains.items():
        prev, prevd, prep, pbif = s_, 0.0, 0, 0
        for n, d, rep, anom, bif in ch:
            if prev not in parent:
                parent[prev] = n; pdist[prev] = d - prevd; erep[prev] = rep - prep; ebif[prev] = bif - pbif
            prev, prevd, prep, pbif = n, d, rep, bif
        if prev not in parent:
            parent[prev] = None; pdist[prev] = 0.0; erep[prev] = 0; ebif[prev] = 0
    nodes = sorted(parent, key=lambda n: (n.startswith("J:"), n))
    # tree roots and tree id
    root_of = {}
    for n in nodes:
        r = n; guard = 0
        while parent[r] is not None and guard < 10000:
            r = parent[r]; guard += 1
        root_of[n] = r
    T = pd.DataFrame({"node": nodes, "is_site": [not n.startswith("J:") for n in nodes], "parent": [parent[n] for n in nodes],
                      "dist_to_parent_km": [pdist[n] / 1000 for n in nodes], "tree_root": [root_of[n] for n in nodes],
                      "edge_repaired_links": [erep[n] for n in nodes], "edge_bifurcations": [ebif[n] for n in nodes]})
    T["tree"] = pd.factorize(T.tree_root)[0] + 1
    T.to_csv(OUT / "rn_tree_nodes.csv", index=False)
    res["tree_nodes"] = len(T); res["trees"] = int(T.tree.nunique())
    tsize = T[T.is_site].groupby("tree").size()
    res["sites_per_tree"] = {"singletons": int((tsize == 1).sum()), "max": int(tsize.max()), "median": float(tsize.median()),
                             "sites_in_trees_ge2": int(tsize[tsize >= 2].sum())}
    res["junction_parent_negative_dist"] = int((T.dist_to_parent_km < -1e-9).sum())

    # ---- connected pairs (exact h) and next-downstream site
    site_set = set(S.site)
    conn = []
    nxt_site, h_next = {}, {}
    for s_, ch in chains.items():
        first = True
        for n, d, rep, anom, bif in ch:
            if n in site_set:
                conn.append((s_, n, d / 1000, rep, anom, bif))
                if first:
                    nxt_site[s_] = n; h_next[s_] = d / 1000; first = False
    C = pd.DataFrame(conn, columns=["up_site", "down_site", "h_km", "n_repaired_links", "n_anomalous_nodes", "n_bifurcations"])
    af = dict(zip(S.site, S.afv_km)); xy = S.set_index("site")[["x_km", "y_km"]]
    C["afv_ratio"] = C.up_site.map(af) / C.down_site.map(af)
    C["euclid_km"] = np.hypot(xy.loc[C.up_site].x_km.values - xy.loc[C.down_site].x_km.values, xy.loc[C.up_site].y_km.values - xy.loc[C.down_site].y_km.values)
    C.to_csv(OUT / "rn_pairs_connected.csv", index=False)
    res["connected_pairs"] = len(C)
    res["connected_pairs_h_le_25km"] = int((C.h_km <= 25).sum())
    res["connected_pairs_through_repaired"] = int((C.n_repaired_links > 0).sum())
    res["connected_pairs_through_anomalous_node"] = int((C.n_anomalous_nodes > 0).sum())
    res["connected_pairs_through_bifurcation"] = int((C.n_bifurcations > 0).sum())
    res["connected_pairs_h_zero"] = int((C.h_km <= 1e-6).sum())
    res["afv_ratio_gt1"] = int((C.afv_ratio > 1 + 1e-9).sum())
    res["h_minus_euclid_negative_gt_0p5km"] = int((C.h_km < C.euclid_km - 0.5).sum())
    # comparison with the Phase 2B pair list (link-start distances)
    P = pd.read_csv(TA / "phase_02b/diagnostics/flow_connected_pairs.csv")
    k_old = set(zip(P.up_site, P.down_site)); k_new = set(zip(C.up_site, C.down_site))
    und_old = {tuple(sorted(k)) for k in k_old}; und_new = {tuple(sorted(k)) for k in k_new}
    res["phase2b_pairs"] = len(P); res["phase2b_unordered"] = len(und_old); res["new_unordered"] = len(und_new)
    res["unordered_only_in_phase2b"] = len(und_old - und_new); res["unordered_only_new"] = len(und_new - und_old)
    res["phase2b_pairs_both_directions"] = int(sum((b, a) in k_old for a, b in k_old) // 2)

    # ---- tail-up parents and the additivity condition sum_p afv(p)/afv(v) <= 1
    S["next_down_site"] = S.site.map(nxt_site); S["h_next_km"] = S.site.map(h_next)
    par = defaultdict(list)
    for s_, n in nxt_site.items():
        par[n].append(s_)
    ssum = {v: sum(af[p] for p in ps) / af[v] for v, ps in par.items()}
    S["n_tu_parents"] = S.site.map(lambda s_: len(par.get(s_, [])))
    S["tu_parent_afv_share"] = S.site.map(lambda s_: ssum.get(s_, 0.0))
    res["tu_nodes_with_parents"] = int((S.n_tu_parents > 0).sum())
    res["tu_max_parents"] = int(S.n_tu_parents.max())
    res["tu_additivity_violations"] = int((S.tu_parent_afv_share > 1 + 1e-9).sum())
    res["tu_parent_share_quantiles"] = S.loc[S.n_tu_parents > 0, "tu_parent_afv_share"].quantile([.5, .9, .99, 1]).round(4).tolist()
    tsite = T[T.is_site].set_index("node").tree
    S["tree"] = S.site.map(tsite)
    S["has_flow_neighbour"] = S.site.isin(set(C.up_site) | set(C.down_site))
    res["sites_with_flow_neighbour"] = int(S.has_flow_neighbour.sum())
    res["sites_with_flow_neighbour_25km"] = int(len(set(C[C.h_km <= 25].up_site) | set(C[C.h_km <= 25].down_site)))
    S.drop(columns=["frac"]).to_csv(OUT / "rn_sites.csv", index=False)

    # ---- flow-unconnected pairs that share a downstream junction (same tree, neither downstream of the other)
    anc = {}
    for n in T.node:
        lst = []; r = n; d = 0.0
        while r is not None:
            lst.append((r, d)); d += pdist[r] / 1000 if parent[r] is not None else 0; r = parent[r]
        anc[n] = lst
    rows = []
    sites_by_tree = S.groupby("tree").site.apply(list)
    for t, lst in sites_by_tree.items():
        if len(lst) < 2:
            continue
        ad = {s_: dict(anc[s_]) for s_ in lst}
        for ii in range(len(lst)):
            a = lst[ii]
            for jj in range(ii + 1, len(lst)):
                b = lst[jj]
                if b in ad[a] or a in ad[b]:
                    continue
                db = ad[b]
                for n, da in anc[a]:
                    if n in db:
                        rows.append((a, b, da, db[n], n)); break
    U = pd.DataFrame(rows, columns=["site_a", "site_b", "a_km", "b_km", "junction"])
    U["euclid_km"] = np.hypot(xy.loc[U.site_a].x_km.values - xy.loc[U.site_b].x_km.values, xy.loc[U.site_a].y_km.values - xy.loc[U.site_b].y_km.values)
    U.to_parquet(OUT / "rn_pairs_unconnected.parquet", index=False)
    res["unconnected_same_tree_pairs"] = len(U)
    res["unconnected_pairs_a_plus_b_le_25km"] = int(((U.a_km + U.b_km) <= 25).sum())
    res["unconnected_pairs_euclid_lt_25km"] = int((U.euclid_km < 25).sum())
    res["component_consistency_violations"] = int((S.groupby("tree").component.nunique() > 1).sum())
    with open(DIAG / "rn_topology.json", "w") as f:
        json.dump(res, f, indent=2, default=float)
    for k, v in res.items():
        print(k, ":", v)


if __name__ == "__main__":
    main()
