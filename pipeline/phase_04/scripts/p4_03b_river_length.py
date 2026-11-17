"""Phase 4 A3 step b: river-length-weighted RSN-level exceedance on G_S (secondary estimand E3; P2B §7, P4-V0 §3 A3).

For posterior draw s, link l (midpoint), year t:
  L = alpha + x_l beta + gamma_t + log SF_{r(l),t} + vA_a + vC_c (observed level, else 0) + vB_b (observed, else 0) + w(s_l)
      + E[v_tu(l) | site v_tu, rho_tu]                                 (exact Gaussian-DAG Markov-blanket conditional)
  V = sigma_u^2 + sigma_om^2 + 1{new catchment} sigma_C^2 + 1{new water body} sigma_B^2 + sigma_tu^2 Var_cond(l)
  pi_R = E_gh Phi((L - psi + s_gh^2/2 - log c) / sqrt(V)),  s_gh = sigma exp(lambda z_l + tau_sig x_gh)
  B_t(c) = sum_l len_l pi_R / sum_l len_l ;  Bbar = (B_2023 + B_2024) / 2
Verification (ii): the DAG conditional is compared with dense closed-form tail-up kriging (independent code path,
phase_03c p3c_lib.tu_cov_closed for sites + walked link-site distances) for a random sample of links.
Outputs: tables/A3_river_length.csv, A3_river_length_strata.csv, A3_river_length_psi.csv; diagnostics/A3_river_length.json;
scratch gs_link_mean_pi.parquet (per-link posterior mean exceedance, for maps)
"""
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.special import ndtr

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import p4_lib as P  # noqa: E402

L = P.redirect()
B, C = L.B, L.C
A = B.A


def tu_conditional(G, Pp, Qq, v, rho):
    """Mean (S, nG) and variance factor (S, nG) of the standardised TU conditional at link midpoints."""
    S, nG = v.shape[0], len(G)
    mean = np.zeros((S, nG)); s_l = np.ones((S, nG))
    if len(Pp):
        g = Pp[:, 0].astype(int); s = Pp[:, 1].astype(int); h = Pp[:, 2]; r = Pp[:, 3]
        M = sp.csr_matrix((np.ones(len(g)), (np.arange(len(g)), g)), shape=(len(g), nG))
        a = np.exp(-h[None, :] / rho[:, None]) * np.sqrt(r)[None, :]
        mean = np.asarray((a * v[:, s]) @ M); s_l = 1.0 - np.asarray((a ** 2) @ M)
    s_l = np.maximum(s_l, 1e-10)
    lam = 1.0 / s_l; num = mean / s_l
    ch = G.tu_child.to_numpy(); has = ch >= 0
    if has.any():
        a_ld = np.where(has[None, :], np.exp(-G.tu_h_child_km.to_numpy()[None, :] / rho[:, None]) * np.sqrt(G.tu_r_child.to_numpy())[None, :], 0.0)
        rd = np.zeros((S, nG)); q2 = np.zeros((S, nG))
        if len(Qq):
            g = Qq[:, 0].astype(int); q = Qq[:, 1].astype(int); h = Qq[:, 2]; w = Qq[:, 3]
            M = sp.csr_matrix((np.ones(len(g)), (np.arange(len(g)), g)), shape=(len(g), nG))
            aq = np.exp(-h[None, :] / rho[:, None]) * np.sqrt(w)[None, :]
            rd = np.asarray((aq * v[:, q]) @ M); q2 = np.asarray((aq ** 2) @ M)
        s_d = np.maximum(1.0 - a_ld ** 2 - q2, 1e-10)
        vd = v[:, np.maximum(ch, 0)]
        lam = lam + np.where(has[None, :], a_ld ** 2 / s_d, 0.0)
        num = num + np.where(has[None, :], a_ld * (vd - rd) / s_d, 0.0)
    return num / lam, 1.0 / lam


def closed_form_check(G, Pp, Qq, sites, n_links=300, seed=20261131):
    """Dense kriging with the closed-form TU correlation (all sites of the river tree) vs the DAG blanket conditional."""
    TA = P.TA
    rn = pd.read_csv(TA / "phase_03c/data_processed/rn_sites.csv").set_index("site").loc[sites.site].reset_index()
    Lk = pd.read_parquet(TA / "phase_02/data_processed/links_gb_attributes.parquet", columns=["id", "u", "v", "length", "form", "canal_carries_flow"])
    act = Lk[(Lk.form != "canal") | Lk.canal_carries_flow].reset_index(drop=True)
    pos = {l: k for k, l in enumerate(act.id.values)}
    out_of = defaultdict(list)
    for k, uu in enumerate(act.u.values):
        out_of[uu].append(k)
    nxt = np.array([out_of[vv][0] if out_of.get(vv) else -1 for vv in act.v.values]); alen = act.length.values / 1000
    rng = np.random.default_rng(seed)
    cand = np.where(G.flow_connected_site.to_numpy())[0]
    pick = rng.choice(cand, size=min(n_links, len(cand)), replace=False)
    gk = {g: pos[G.id.iloc[g]] for g in pick}
    by_k = defaultdict(list)
    for g, k in gk.items():
        by_k[k].append(g)
    sk = np.array([pos[l] for l in rn.link_id]); sf = rn.f_up.values; afv = rn.afv_km.values
    up = defaultdict(list)                                 # g -> all sites upstream (full path walk) with h
    for s, (k, f) in enumerate(zip(sk, sf)):
        if f < 0.5 and k in by_k:
            for g in by_k[k]:
                up[g].append((s, (0.5 - f) * alen[k]))
        d = (1 - f) * alen[k]; j = nxt[k]; seen = {k}
        while j >= 0 and j not in seen:
            seen.add(j)
            if j in by_k:
                for g in by_k[j]:
                    up[g].append((s, d + 0.5 * alen[j]))
            d += alen[j]; j = nxt[j]
    on = defaultdict(list)
    for s, (k, f) in enumerate(zip(sk, sf)):
        on[k].append((f, s))
    rho = 120.0
    tree = rn.tree.to_numpy()
    cpairs = pd.read_csv(TA / "phase_03c/data_processed/rn_pairs_connected.csv"); afvd = dict(zip(rn.site, rn.afv_km))
    out = []
    for g in pick:
        k = gk[g]; dn = [(f - 0.5) * alen[k], ]
        down = [(s, (f - 0.5) * alen[k]) for f, s in on.get(k, []) if f > 0.5]
        d = 0.5 * alen[k]; j = nxt[k]; seen = {k}
        while j >= 0 and j not in seen:
            seen.add(j)
            down += [(s, d + f * alen[j]) for f, s in on.get(j, [])]
            d += alen[j]; j = nxt[j]
        conn = {s: h for s, h in up[g] + down}
        if not conn:
            continue
        tr = tree[list(conn)[0]]
        idx = np.where(tree == tr)[0]
        if len(idx) > 400:
            continue
        sub = sites.iloc[idx].reset_index(drop=True)
        Css = C.tu_cov_closed(C.rn_structures(sub), rho, sub)
        nm = list(sub.site); pz = {n_: k_ for k_, n_ in enumerate(nm)}
        q_ = cpairs[cpairs.up_site.isin(pz) & cpairs.down_site.isin(pz)]
        raw_ = np.array([np.exp(-h_ / rho) * np.sqrt(min(1, afvd[u_] / afvd[d_])) for u_, d_, h_ in zip(q_.up_site, q_.down_site, q_.h_km)])
        mod_ = np.array([Css[pz[u_], pz[d_]] for u_, d_ in zip(q_.up_site, q_.down_site)])
        tree_dev = float(np.abs(raw_ - mod_).max()) if len(q_) else 0.0
        la = G.afv_km.iloc[g]
        c = np.array([np.exp(-conn[s] / rho) * np.sqrt(min(afv[s], la) / max(afv[s], la)) if s in conn else 0.0 for s in idx])
        v = rng.multivariate_normal(np.zeros(len(idx)), Css + 1e-12 * np.eye(len(idx)))
        kk = np.linalg.solve(Css + 1e-12 * np.eye(len(idx)), c)
        m_cf, v_cf = float(kk @ v), float(1 - c @ kk)
        vfull = np.zeros((1, len(sites))); vfull[0, idx] = v
        Gs = G.iloc[[g]].reset_index(drop=True)
        Pg = Pp[Pp[:, 0] == g].copy(); Pg[:, 0] = 0
        Qg = Qq[Qq[:, 0] == g].copy(); Qg[:, 0] = 0
        m_d, v_d = tu_conditional(Gs, Pg, Qg, vfull, np.array([rho]))
        out.append(dict(link=int(g), n_tree_sites=len(idx), tree_raw_vs_model_corr_maxdev=tree_dev, mean_cf=m_cf, mean_dag=float(m_d[0, 0]), var_cf=v_cf, var_dag=float(v_d[0, 0])))
    T = pd.DataFrame(out)
    T["abs_mean_diff"] = (T.mean_cf - T.mean_dag).abs(); T["abs_var_diff"] = (T.var_cf - T.var_dag).abs()
    return T


def main():
    t0 = time.time(); cfg = P.CFG["A3_river_length"]
    sc = P.scratch()
    G = pd.read_parquet(sc / "gs_links.parquet"); bl = np.load(sc / "gs_tu_blanket.npz"); Pp, Qq = bl["parents"], bl["coparents"]
    meas, sites, sy = L.load_tables(); mc = L.R.model_config("R3")
    res = {}
    # ---- (ii) verification first
    CF = closed_form_check(G, Pp, Qq, sites)
    CF.to_csv(P.P4 / "tables/A3_tu_closed_form_check.csv", index=False)
    CF["afv_anomaly"] = G.tu_afv_anomaly.to_numpy()[CF.link.to_numpy()]
    ok = (CF.abs_mean_diff < 1e-6) & (CF.abs_var_diff < 1e-6)
    na = (~CF.afv_anomaly) & (CF.tree_raw_vs_model_corr_maxdev < 1e-9)
    res["verify_tu_closed_form"] = dict(n=len(CF), definition="AFV-consistent = link not flagged and every flow-connected site pair in its tree has model composite weight == raw end-to-end AFV ratio",
                                        n_afv_consistent=int(na.sum()), share_within_1e6_afv_consistent=float(ok[na].mean()),
                                        max_abs_mean_diff_afv_consistent=float(CF.abs_mean_diff[na].max()), max_abs_var_diff_afv_consistent=float(CF.abs_var_diff[na].max()),
                                        n_afv_anomalous=int((~na).sum()), max_abs_mean_diff_anomalous=float(CF.abs_mean_diff[~na].max()) if (~na).any() else None,
                                        max_abs_var_diff_anomalous=float(CF.abs_var_diff[~na].max()) if (~na).any() else None,
                                        share_length_afv_anomalous=float(G.len_england_m[G.tu_afv_anomaly].sum() / G.len_england_m.sum()))
    CF.to_csv(P.P4 / "tables/A3_tu_closed_form_check.csv", index=False)
    print("closed-form check", res["verify_tu_closed_form"], flush=True)
    # ---- posterior draws
    D = P.load_post(cfg["draws"]); S = len(D["lp__"])
    g1 = B.get1
    sr = pd.DataFrame({"region_idx": np.arange(1, 8), "ms": 0, "site": [f"r{k}" for k in range(1, 8)]})
    syr = pd.DataFrame({"site_idx": np.tile(np.arange(1, 8), 2), "year_idx": np.repeat([1, 2], 7)})
    lsf = B.log_sf_siteyear(D, sr, syr, mc)                 # (S, 14): (year-1)*7 + region-1
    gam = g1(D, "gamma"); lam = g1(D, "lambda"); tsg = g1(D, "tau_sig"); sg = g1(D, "sigma")
    rho_tu = g1(D, "rho_tu"); s_tu = g1(D, "sigma_tu"); vtu = np.asarray(D["v_tu"], float)
    alpha = np.asarray(D["alpha"], float); beta = np.asarray(D["beta"], float)
    zA = np.asarray(D["z_A"], float); zC = np.asarray(D["z_C"], float); vBd = np.asarray(D["vB"], float); zw = np.asarray(D["z_w"], float)
    sA, sC, sB = g1(D, "sigma_A"), g1(D, "sigma_C"), g1(D, "sigma_B"); sw, rw = g1(D, "sigma_w"), g1(D, "rho")
    v0 = g1(D, "sigma_u") ** 2 + g1(D, "sigma_om") ** 2
    lenw = G.len_england_m.to_numpy()
    strata = {"all": np.ones(len(G), bool), "afv_consistent": ~G.tu_afv_anomaly.to_numpy(), "afv_anomalous": G.tu_afv_anomaly.to_numpy(), "flow_connected_site": G.flow_connected_site.to_numpy(), "no_flow_connected_site": ~G.flow_connected_site.to_numpy(),
              "observed_water_body": G.wb_idx.to_numpy() > 0, "new_water_body": G.wb_idx.to_numpy() == 0,
              "rsn_supported_cell": G.rsn_supported_cell.to_numpy(), "no_rsn_support": ~G.rsn_supported_cell.to_numpy()}
    for r in range(1, 8):
        strata[f"region_{r}"] = G.region_idx.to_numpy() == r
    for scls in sorted(G.size_hat.astype(str).unique()):
        strata[f"size_{scls}"] = G.size_hat.astype(str).to_numpy() == scls
    psis = sorted(set(P.CFG["A2_selection"]["psi_primary"] + P.CFG["A2_selection"]["psi_extreme"]))
    acc = defaultdict(lambda: np.zeros(S))                  # weighted sums of pi over links, per draw
    link_mean = {f"pi_{c}": np.zeros(len(G)) for c in P.THRESH}
    tu_sd_ratio = np.zeros(len(G))
    gx15, gw15 = A.gh_nodes(cfg["gh_nodes"]); gx7, gw7 = A.gh_nodes(7)
    pg = Pp[:, 0].astype(int) if len(Pp) else np.zeros(0, int); qg = Qq[:, 0].astype(int) if len(Qq) else np.zeros(0, int)
    CH = 2500
    for c0 in range(0, len(G), CH):
        ix = np.arange(c0, min(len(G), c0 + CH)); Gc = G.iloc[ix].reset_index(drop=True)
        sel = (pg >= c0) & (pg < c0 + CH); Pc = Pp[sel].copy(); Pc[:, 0] -= c0
        sel = (qg >= c0) & (qg < c0 + CH); Qc = Qq[sel].copy(); Qc[:, 0] -= c0
        X = A.site_design(Gc)
        mci = Gc.mancat_idx.to_numpy(); wbi = Gc.wb_idx.to_numpy()
        base0 = alpha[:, None] + beta @ X.T + sA[:, None] * zA[:, Gc.wma_idx.to_numpy() - 1]
        base0 += np.where(mci[None, :] > 0, sC[:, None] * zC[:, np.maximum(mci, 1) - 1], 0.0)
        base0 += np.where(wbi[None, :] > 0, vBd[:, np.maximum(wbi, 1) - 1], 0.0)
        PhiE, slE, PhiN, slN = B.hsgp_basis(Gc.x_km, Gc.y_km)
        base0 += B.hsgp_field(PhiE, PhiN, slE, slN, zw, sw, rw)
        tm, tv = tu_conditional(Gc, Pc, Qc, vtu, rho_tu)
        tu_sd_ratio[ix] = np.median(np.sqrt(tv), axis=0)
        vb = v0[:, None] + np.where(mci[None, :] == 0, sC[:, None] ** 2, 0.0) + np.where(wbi[None, :] == 0, sB[:, None] ** 2, 0.0)
        sd_c = np.sqrt(vb + (s_tu ** 2)[:, None] * tv); sd_m = np.sqrt(vb + (s_tu ** 2)[:, None])
        reg = Gc.region_idx.to_numpy() - 1; zt = Gc.z_logarea.to_numpy(); wl = lenw[ix]
        hh15 = [(wg, 0.5 * (sg[:, None] * np.exp(lam[:, None] * zt[None, :] + tsg[:, None] * xg)) ** 2) for xg, wg in zip(gx15, gw15)]
        hh7 = [(wg, 0.5 * (sg[:, None] * np.exp(lam[:, None] * zt[None, :] + tsg[:, None] * xg)) ** 2) for xg, wg in zip(gx7, gw7)]
        for yr in (1, 2):
            Lc = base0 + tm + (gam * (yr == 2))[:, None] + lsf[:, (yr - 1) * 7 + reg]
            Lm = base0 + (gam * (yr == 2))[:, None] + lsf[:, (yr - 1) * 7 + reg]
            for c in P.THRESH:
                lc = np.log(c)
                pc = sum(wg * ndtr((Lc + h - lc) / sd_c) for wg, h in hh15)
                pm = sum(wg * ndtr((Lm + h - lc) / sd_m) for wg, h in hh15)
                acc[("B", c, yr)] += pc @ wl; acc[("Bmarg", c, yr)] += pm @ wl
                link_mean[f"pi_{c}"][ix] += 0.5 * pc.mean(0)
                for lab, msk in strata.items():
                    m_ = msk[ix]
                    if m_.any():
                        acc[("S", lab, c, yr)] += pc[:, m_] @ wl[m_]
                for psi in psis:
                    pp = sum(wg * ndtr((Lc - psi + h - lc) / sd_c) for wg, h in hh7)
                    acc[("psi", psi, c, yr)] += pp @ wl
        print("chunk", c0, round(time.time() - t0), flush=True)
    tot = lenw.sum()
    rows, srows, prow = [], [], []
    for c in P.THRESH:
        B1, B2 = acc[("B", c, 1)] / tot, acc[("B", c, 2)] / tot
        for lab, x in [("B_2023", B1), ("B_2024", B2), ("Bbar", 0.5 * (B1 + B2))]:
            rows.append(dict(threshold=c, quantity=lab, variant="primary (tail-up conditional)", **P.band(x)))
        rows.append(dict(threshold=c, quantity="Bbar", variant="sensitivity: river term marginalised (no tail-up conditioning)",
                         **P.band(0.5 * (acc[("Bmarg", c, 1)] + acc[("Bmarg", c, 2)]) / tot)))
        for lab, msk in strata.items():
            km = lenw[msk].sum()
            if km == 0:
                continue
            srows.append(dict(threshold=c, stratum=lab, km=float(km / 1e3), links=int(msk.sum()),
                              **P.band(0.5 * (acc[("S", lab, c, 1)] + acc[("S", lab, c, 2)]) / km)))
        for psi in psis:
            prow.append(dict(threshold=c, psi=psi, gh_nodes=7, **P.band(0.5 * (acc[("psi", psi, c, 1)] + acc[("psi", psi, c, 2)]) / tot)))
    R_ = pd.DataFrame(rows); R_.to_csv(P.P4 / "tables/A3_river_length.csv", index=False)
    SR = pd.DataFrame(srows); SR.to_csv(P.P4 / "tables/A3_river_length_strata.csv", index=False)
    PS = pd.DataFrame(prow); PS.to_csv(P.P4 / "tables/A3_river_length_psi.csv", index=False)
    LM = G[["id", "x_km", "y_km", "len_england_m", "region_idx", "size_hat", "flow_connected_site", "wb_idx"]].copy()
    for k, v in link_mean.items():
        LM[k] = v
    LM["tu_cond_sd_ratio_median"] = tu_sd_ratio
    LM.to_parquet(sc / "gs_link_mean_pi.parquet")
    res.update(draws=S, seconds=time.time() - t0, gh_check_psi0_vs_15nodes={str(c): float(PS[(PS.threshold == c) & (PS.psi == 0)]["median"].iloc[0]
                                                                                       - R_[(R_.threshold == c) & (R_.quantity == "Bbar") & (R_.variant.str.startswith("primary"))]["median"].iloc[0]) for c in P.THRESH},
               tu_conditional_sd_ratio=dict(median_over_links=float(np.median(tu_sd_ratio)), share_links_below_0_9=float((tu_sd_ratio < 0.9).mean()),
                                            share_length_below_0_9=float(lenw[tu_sd_ratio < 0.9].sum() / tot)))
    P.save_json(res, P.P4 / "diagnostics/A3_river_length.json")
    pd.set_option("display.width", 250)
    print(json.dumps(res, indent=1)); print(R_.round(4).to_string()); print(SR[["threshold", "stratum", "km", "median", "lo95", "hi95"]].round(4).to_string())
    print(PS.round(4).to_string())


if __name__ == "__main__":
    main()
