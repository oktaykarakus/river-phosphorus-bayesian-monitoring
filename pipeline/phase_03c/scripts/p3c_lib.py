"""Shared library for Track A Phase 3C (river-network models, M2-RN).

Imports the frozen Phase 3B library (phase_03b/scripts/p3b_lib.py) READ-ONLY for data tables, the Stan data builder,
posterior reconstruction, Delta_T machinery and residual statistics. Adds:
  * river-network structures: the tail-up (TU) directed acyclic graph over sites and the tail-down (TD) tree over
    sites + confluence junctions (from data_processed/rn_*.csv, built by p3c_01_topology.py);
  * dense reference covariances (closed form and recursion) for verification;
  * a Phase 3C fit runner (writes only to phase_03c and the Phase 3C scratch directory);
  * Delta_T including the river-network location term;
  * held-out prediction / scoring utilities (validation).
Nothing in phase_03b is written: p3b_lib.run_fit / get_model are never called from Phase 3C.
"""
from __future__ import annotations

import json
import os
import resource
import shutil
import sys
sys.dont_write_bytecode = True   # never write __pycache__ into frozen phase trees
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.special import ndtr, log_ndtr

ROOT = Path(__file__).resolve().parents[1]          # phase_03c
TA = ROOT.parent
P3B = TA / "phase_03b"
sys.path.insert(0, str(P3B / "scripts"))
import p3b_lib as B  # noqa: E402  (frozen Phase 3B library; imports phase_03a p3a_lib as B.A)

A = B.A
CFG = yaml.safe_load(open(ROOT / "config" / "phase3c_settings.yaml"))
STAN_FILE = ROOT / "stan" / "track_a_m2rn.stan"
THRESH = B.THRESH
DP = ROOT / "data_processed"
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


def scratch_dir() -> Path:
    p = Path(os.environ.get("TRACK_A_SCRATCH_3C", os.path.expanduser(CFG["scratch_dir"])))
    p.mkdir(parents=True, exist_ok=True)
    return p


def ref_scratch() -> Path:
    return Path(os.path.expanduser(CFG["reference_scratch"]))


def load_tables():
    return B.load_tables()


def save_json(obj, path):
    A.save_json(obj, path)


# ====================================================================== river-network structures
H_FLOOR_KM = 0.01      # 10 m: coincident sites (h = 0) get a minimal separation so the innovation variance stays > 0
AFV_FLOOR_KM = 0.01


def _full_structures(drop_repaired: bool):
    """TU edges and TD tree over ALL 1,832 model sites (phase_03a sites order)."""
    full = pd.read_parquet(TA / "phase_03a/data_processed/sites.parquet").sort_values("site_idx")
    R = pd.read_csv(DP / "rn_sites.csv").set_index("site").loc[full.site]
    Cp = pd.read_csv(DP / "rn_pairs_connected.csv")
    T = pd.read_csv(DP / "rn_tree_nodes.csv")
    I = len(full); sid = {s: k for k, s in enumerate(full.site)}
    afv = np.maximum(R.afv_km.to_numpy(), AFV_FLOOR_KM)
    rep = {(a, b): r for a, b, r in zip(Cp.up_site, Cp.down_site, Cp.n_repaired_links)}
    nxt = -np.ones(I, int); hh = np.zeros(I); rr = np.zeros(I)
    for s, nd, h in zip(R.index, R.next_down_site, R.h_next_km):
        if not isinstance(nd, str):
            continue
        if drop_repaired and rep.get((s, nd), 0) > 0:
            continue
        k = sid[s]; nxt[k] = sid[nd]; hh[k] = max(float(h), H_FLOOR_KM); rr[k] = min(1.0, afv[k] / afv[sid[nd]])
    tot = np.bincount(nxt[nxt >= 0], weights=rr[nxt >= 0], minlength=I)
    pw = np.where(nxt >= 0, rr / np.maximum(1.0, tot[np.maximum(nxt, 0)]), 0.0)
    T = T.copy()
    if drop_repaired:
        T.loc[T.edge_repaired_links > 0, "parent"] = np.nan
    nodes = list(full.site) + [n for n in T.node if n.startswith("J:")]
    nid = {n: k for k, n in enumerate(nodes)}
    Tn = T.set_index("node").loc[nodes]
    td_parent = np.array([nid[p] if isinstance(p, str) else -1 for p in Tn.parent], int)
    td_dist = np.maximum(Tn.dist_to_parent_km.to_numpy(), H_FLOOR_KM); td_dist[td_parent < 0] = 0.0
    return full, R, sid, nxt, hh, pw, nodes, td_parent, td_dist, afv


def rn_structures(sites: pd.DataFrame, drop_repaired: bool = False) -> dict:
    """Tail-up DAG and tail-down tree in the site order of `sites` (any subset of the 1,832 model sites).

    TU (exponential tail-up with additive weights; Ver Hoef & Peterson 2006, 2010): on the full site set every site v
    has as parents the sites p whose NEAREST downstream site is v; edge (p -> v) has river distance h_pv and weight
    pi_pv = min(1, afv_p / afv_v), normalised so that sum_p pi_pv <= 1 at every v (needed only at 5 bifurcation-affected
    nodes; P3C_01 §3). For a SUBSET, edges are composed along the full-set chains (h summed, weights multiplied), which
    is exactly the marginal TU covariance of the subset.
    TD (exponential tail-down): tree over all sites + confluence junctions, rooted at the most-downstream node of every
    river tree; sites outside the subset remain as latent nodes (like junctions). td_site_node maps subset sites to nodes.
    drop_repaired: cut every TU / TD edge whose river path crosses a direction-repaired link (sensitivity)."""
    full, R, sid, nxt, hh, pw, nodes, td_parent, td_dist, afv_full = _full_structures(drop_repaired)
    I = len(sites)
    sub = np.array([sid[s] for s in sites.site], int)
    loc = -np.ones(len(full), int); loc[sub] = np.arange(I)
    ch, pa, H, W = [], [], [], []
    for k_sub, k in enumerate(sub):
        j, hs, lw = k, 0.0, 0.0
        while nxt[j] >= 0:
            hs += hh[j]; lw += np.log(pw[j]) if pw[j] > 0 else -np.inf; j = nxt[j]
            if loc[j] >= 0:
                ch.append(loc[j]); pa.append(k_sub); H.append(hs); W.append(np.exp(lw)); break
    ch = np.array(ch, int); pa = np.array(pa, int); H = np.array(H); W = np.array(W)
    o = np.argsort(ch, kind="stable"); ch, pa, H, W = ch[o], pa[o], H[o], W[o]
    ptr = np.zeros(I + 1, int); np.add.at(ptr, ch + 1, 1); ptr = np.cumsum(ptr)
    remaining = np.bincount(ch, minlength=I).copy(); kids = {}
    for c_, p_ in zip(ch, pa):
        kids.setdefault(p_, []).append(c_)
    order = [i for i in range(I) if remaining[i] == 0]; q = 0
    while q < len(order):
        for c_ in kids.get(order[q], []):
            remaining[c_] -= 1
            if remaining[c_] == 0:
                order.append(c_)
        q += 1
    assert len(order) == I, "TU graph is not acyclic"
    depth = np.zeros(len(nodes), int)
    for j in range(len(nodes)):
        r, dpt = j, 0
        while td_parent[r] >= 0:
            r = td_parent[r]; dpt += 1
        depth[j] = dpt
    td_order = np.argsort(depth, kind="stable")
    tree = R.tree.to_numpy()[sub]
    return dict(I=I, tu_child=ch, tu_par=pa, tu_h=H, tu_pw=W, tu_ptr=ptr, tu_order=np.array(order, int), n_tu_edges=len(ch),
                td_nodes=nodes, n_td=len(nodes), td_parent=td_parent, td_dist=td_dist, td_order=td_order, td_site_node=sub,
                tree=tree, afv=afv_full[sub], drop_repaired=drop_repaired, parent_weight_sum=np.bincount(ch, weights=W, minlength=I))


def tu_stan_data(rn) -> dict:
    return dict(n_tu_edges=int(rn["n_tu_edges"]), tu_order=(rn["tu_order"] + 1).tolist(), tu_ptr=(rn["tu_ptr"] + 1).tolist(),
                tu_par=(rn["tu_par"] + 1).tolist(), tu_h=rn["tu_h"].tolist(), tu_sqrt_pw=np.sqrt(rn["tu_pw"]).tolist())


def td_stan_data(rn) -> dict:
    return dict(n_td=int(rn["n_td"]), td_order=(rn["td_order"] + 1).tolist(), td_parent=(rn["td_parent"] + 1).tolist(),
                td_dist=rn["td_dist"].tolist(), td_site_node=(rn["td_site_node"] + 1).tolist())


# ---------------------------------------------------------------------- reference implementations (verification)
def tu_effect_np(rn, z, sigma, rho):
    """Python mirror of the Stan TU recursion. z: (S, I); sigma, rho: (S,). Returns (S, I)."""
    z = np.atleast_2d(z); S = z.shape[0]
    sigma = np.broadcast_to(np.asarray(sigma, float), (S,)); rho = np.broadcast_to(np.asarray(rho, float), (S,))
    v = np.zeros_like(z, dtype=float)
    ptr, par, h, spw = rn["tu_ptr"], rn["tu_par"], rn["tu_h"], np.sqrt(rn["tu_pw"])
    for i in rn["tu_order"]:
        e = slice(ptr[i], ptr[i + 1])
        a = np.exp(-h[e][None, :] / rho[:, None]) * spw[e][None, :]
        acc = (a * v[:, par[e]]).sum(1)
        s2 = np.maximum(1 - (a ** 2).sum(1), 1e-10)
        v[:, i] = acc + sigma * np.sqrt(s2) * z[:, i]
    return v


def td_effect_np(rn, z, sigma, rho):
    """Python mirror of the Stan TD recursion. z: (S, n_td). Returns site values (S, I)."""
    z = np.atleast_2d(z); S = z.shape[0]
    sigma = np.broadcast_to(np.asarray(sigma, float), (S,)); rho = np.broadcast_to(np.asarray(rho, float), (S,))
    v = np.zeros_like(z, dtype=float)
    for j in rn["td_order"]:
        p = rn["td_parent"][j]
        if p < 0:
            v[:, j] = sigma * z[:, j]
        else:
            a = np.exp(-rn["td_dist"][j] / rho)
            v[:, j] = a * v[:, p] + sigma * np.sqrt(1 - a ** 2) * z[:, j]
    return v[:, rn["td_site_node"]]


def tu_cov_dag(rn, rho):
    """Dense TU correlation implied by the DAG: (I - B)^{-1} D (I - B)^{-T}."""
    I = rn["I"]; Bm = np.zeros((I, I)); d = np.ones(I)
    a = np.exp(-rn["tu_h"] / rho) * np.sqrt(rn["tu_pw"])
    Bm[rn["tu_child"], rn["tu_par"]] = a
    np.subtract.at(d, rn["tu_child"], a ** 2)
    Linv = np.linalg.inv(np.eye(I) - Bm)
    return (Linv * d[None, :]) @ Linv.T


def tu_cov_closed(rn, rho, sites):
    """Closed-form TU correlation exp(-h/rho) * sqrt(prod of edge weights along the flow path), 0 for flow-unconnected
    pairs (independent code path: walks next-downstream chains, uses the exact pair distances of p3c_01)."""
    I = rn["I"]
    par_of = {int(p): (int(c), float(h), float(w)) for c, p, h, w in zip(rn["tu_child"], rn["tu_par"], rn["tu_h"], rn["tu_pw"])}
    Cm = np.eye(I)
    for i in range(I):
        j, hsum, lw = i, 0.0, 0.0
        while j in par_of:
            c, h, w = par_of[j]
            hsum += h; lw += np.log(w)
            Cm[i, c] = Cm[c, i] = np.exp(-hsum / rho + 0.5 * lw)
            j = c
    return Cm


def td_cov_closed(rn, rho, sites):
    """Closed-form tail-down exponential correlation: exp(-h/rho) for flow-connected pairs, exp(-(a+b)/rho) for
    flow-unconnected pairs sharing a junction, 0 across river trees (from the pair tables of p3c_01, exact distances).
    Only valid for drop_repaired=False (pair tables are not re-cut)."""
    I = rn["I"]; sid = {s: k for k, s in enumerate(sites.site)}
    Cp = pd.read_csv(DP / "rn_pairs_connected.csv"); U = pd.read_parquet(DP / "rn_pairs_unconnected.parquet")
    Cm = np.eye(I)
    a = Cp.up_site.map(sid).to_numpy(); b = Cp.down_site.map(sid).to_numpy()
    Cm[a, b] = Cm[b, a] = np.exp(-np.maximum(Cp.h_km.to_numpy(), H_FLOOR_KM) / rho)
    a = U.site_a.map(sid).to_numpy(); b = U.site_b.map(sid).to_numpy()
    Cm[a, b] = Cm[b, a] = np.exp(-(U.a_km.to_numpy() + U.b_km.to_numpy()) / rho)
    return Cm


def td_cov_tree(rn, rho):
    """Dense TD correlation implied by the tree recursion (all nodes), returned at the sites."""
    n = rn["n_td"]; Bm = np.zeros((n, n)); d = np.ones(n)
    for j in range(n):
        p = rn["td_parent"][j]
        if p >= 0:
            a = np.exp(-rn["td_dist"][j] / rho); Bm[j, p] = a; d[j] = 1 - a ** 2
    Linv = np.linalg.inv(np.eye(n) - Bm)
    Cm = (Linv * d[None, :]) @ Linv.T
    return Cm[np.ix_(rn["td_site_node"], rn["td_site_node"])]


def tu_cov_fast(rn, rho, chains=None):
    """Fast dense TU correlation (for REML screening): uses precomputed chains (i, c, hsum, lw)."""
    if chains is None:
        chains = tu_chains(rn)
    I = rn["I"]; Cm = np.eye(I)
    i, c, hs, lw = chains
    v = np.exp(-hs / rho + 0.5 * lw)
    Cm[i, c] = v; Cm[c, i] = v
    return Cm


def tu_chains(rn):
    par_of = {int(p): (int(c), float(h), float(w)) for c, p, h, w in zip(rn["tu_child"], rn["tu_par"], rn["tu_h"], rn["tu_pw"])}
    out = []
    for i in range(rn["I"]):
        j, hsum, lw = i, 0.0, 0.0
        while j in par_of:
            c, h, w = par_of[j]; hsum += h; lw += np.log(w); out.append((i, c, hsum, lw)); j = c
    a = np.array(out)
    return a[:, 0].astype(int), a[:, 1].astype(int), a[:, 2], a[:, 3]


def td_pairs(rn, sites):
    """Pair list (i, j, distance) for TD from the tree (works for drop_repaired too): walk ancestors."""
    n = rn["n_td"]; par = rn["td_parent"]; dist = rn["td_dist"]; I = rn["I"]
    anc = []
    for j in rn["td_site_node"]:
        lst = {}; r = j; d = 0.0
        while r >= 0:
            lst[r] = d; d += dist[r] if par[r] >= 0 else 0; r = par[r]
        anc.append(lst)
    ii, jj, dd = [], [], []
    tree = rn["tree"]
    by_tree = pd.Series(range(I)).groupby(tree).apply(list)
    for lst in by_tree:
        for x in range(len(lst)):
            a = lst[x]; aa = anc[a]
            for y in range(x + 1, len(lst)):
                b = lst[y]; ab = anc[b]
                # nearest common ancestor: walk a's ancestors in order
                best = None
                r = rn["td_site_node"][a]
                while r >= 0:
                    if r in ab:
                        best = aa[r] + ab[r]; break
                    r = par[r]
                if best is not None:
                    ii.append(a); jj.append(b); dd.append(best)
    return np.array(ii, int), np.array(jj, int), np.array(dd)


def td_cov_fast(rn, rho, pairs):
    I = rn["I"]; Cm = np.eye(I)
    i, j, d = pairs
    v = np.exp(-d / rho); Cm[i, j] = v; Cm[j, i] = v
    return Cm


# ====================================================================== Stan data
def model_config(name="S5c_regional_shock_centred", tu=0, td=0, **kw):
    mc = B.model_config(name, hier=1, field=1)
    mc.update(tu=int(tu), td=int(td), drop_repaired=0)
    mc.update(kw)
    return mc


def build_data(meas, sites, sy, mc: dict, J_chk=50):
    d = B.build_data(meas, sites, sy, mc, J_chk=J_chk)
    rn = rn_structures(sites, drop_repaired=bool(mc.get("drop_repaired", 0)))
    d.update(use_tu=int(mc.get("tu", 0)), use_td=int(mc.get("td", 0)))
    d.update(tu_stan_data(rn)); d.update(td_stan_data(rn))
    r = CFG["river_network"]
    d.update(sigma_tu_sd=r["sigma_sd"], rho_tu_a=r["rho_a"], rho_tu_b=r["rho_b"], sigma_td_sd=r["sigma_sd"], rho_td_a=r["rho_a"], rho_td_b=r["rho_b"])
    return d


# ====================================================================== fitting (Phase 3C outputs only)
TOP_CANDIDATES = B.TOP_CANDIDATES + ["sigma_tu", "rho_tu", "sigma_td", "rho_td"]


def get_model(stan_file=STAN_FILE):
    import cmdstanpy
    stan_file = Path(stan_file)
    exe = stan_file.with_suffix("")
    if exe.exists() and exe.stat().st_mtime < stan_file.stat().st_mtime:
        exe.unlink()
    return cmdstanpy.CmdStanModel(stan_file=str(stan_file))


def run_fit(tag, data, mc, iter_warmup=None, iter_sampling=None, chains=None, adapt_delta=None, max_treedepth=None, inits=None, seed=None,
            keep_csv=False, extra_diag=None, metric=None, step_size=None, parallel_chains=None):
    import cmdstanpy, arviz as az
    s = CFG["sampling"]
    chains = chains or s["chains"]
    iw = iter_warmup if iter_warmup is not None else s["iter_warmup"]; isamp = iter_sampling or s["iter_sampling"]
    ad = adapt_delta or s["adapt_delta"]; mt = max_treedepth or s["max_treedepth"]
    outdir = scratch_dir() / f"fit_{tag}"
    if outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True)
    model = get_model()
    r0 = resource.getrusage(resource.RUSAGE_CHILDREN); t0 = time.time()
    kw = {}
    if metric is not None:
        kw["metric"] = metric
    if step_size is not None:
        kw["step_size"] = step_size
    fit = model.sample(data=data, chains=chains, parallel_chains=parallel_chains or chains, iter_warmup=iw, iter_sampling=isamp, adapt_delta=ad,
                       max_treedepth=mt, seed=seed or CFG["seeds"]["stan"], output_dir=str(outdir), show_progress=False, inits=inits, sig_figs=9, **kw)
    wall = time.time() - t0; r1 = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu = (r1.ru_utime - r0.ru_utime) + (r1.ru_stime - r0.ru_stime)
    mv = fit.method_variables()
    idata = az.from_cmdstanpy(posterior=fit)
    post = idata.posterior
    top = [v for v in TOP_CANDIDATES if v in post and post[v].size > 0]
    summ = az.summary(idata, var_names=top, round_to=6)
    lat_vars = [v for v in ["theta", "z_om", "z_sig", "vB", "z_C", "z_A", "w_site", "z_e", "z_tu", "z_td", "v_tu", "v_td"] if v in post and post[v].size > 0]
    lat = az.summary(idata, var_names=lat_vars, kind="diagnostics")
    E = np.asarray(mv["energy__"])  # (draws, chains)
    res = dict(tag=tag, model_config=mc, wall_s=wall, cpu_s=cpu, cpu_h=cpu / 3600, peak_child_rss_mb=r1.ru_maxrss / 1e6, chains=chains,
               iter_warmup=iw, iter_sampling=isamp, adapt_delta=ad, max_treedepth=mt, seed=seed or CFG["seeds"]["stan"],
               I=data["I"], J=data["J"], N=data["N"],
               divergences=int(mv["divergent__"].sum()), divergences_by_chain=np.asarray(mv["divergent__"]).sum(axis=0).astype(int).tolist(),
               treedepth_hits=int((mv["treedepth__"] >= mt).sum()), mean_treedepth=float(mv["treedepth__"].mean()),
               mean_leapfrog=float(mv["n_leapfrog__"].mean()), stepsize_by_chain=np.atleast_2d(mv["stepsize__"])[0].round(5).tolist(),
               ebfmi=[float(x) for x in az.bfmi(idata)],
               top_max_rhat=float(summ["r_hat"].max()), top_argmax_rhat=str(summ["r_hat"].idxmax()),
               top_min_ess_bulk=float(summ["ess_bulk"].min()), top_argmin_ess_bulk=str(summ["ess_bulk"].idxmin()),
               top_min_ess_tail=float(summ["ess_tail"].min()), top_argmin_ess_tail=str(summ["ess_tail"].idxmin()),
               latent_max_rhat=float(lat["r_hat"].max()), latent_argmax_rhat=str(lat["r_hat"].idxmax()),
               latent_min_ess_bulk=float(lat["ess_bulk"].min()), latent_min_ess_tail=float(lat["ess_tail"].min()),
               gradient_evals_sampling=int(mv["n_leapfrog__"].sum()))
    res["ms_per_gradient_sampling_approx"] = None
    if extra_diag:
        res.update(extra_diag)
    cw = {}
    for v in ["alpha", "delta_bar", "sigma_u", "sigma_om", "sigma", "sigma_w", "rho", "sigma_B", "sigma_C", "sigma_A", "sigma_tu", "rho_tu", "sigma_td", "rho_td"]:
        if v in post and post[v].size > 0:
            cw[v] = post[v].values.reshape(post.sizes["chain"], post.sizes["draw"], -1)[..., 0].mean(axis=1).round(4).tolist()
    res["chain_means"] = cw
    summ.to_csv(ROOT / "tables" / f"{tag}_posterior_summary.csv")
    draws = {}
    for v in list(post.data_vars):
        arr = post[v].values
        draws[v] = arr.reshape(arr.shape[0] * arr.shape[1], *arr.shape[2:])
    for v in ["divergent__", "treedepth__", "n_leapfrog__", "lp__", "energy__", "stepsize__"]:
        draws[v] = np.asarray(mv[v]).T.reshape(-1)
    np.savez(scratch_dir() / f"{tag}_draws.npz", **draws)
    # adapted metric / step size for warm starts (validation folds)
    try:
        res["inv_metric_file"] = str(scratch_dir() / f"{tag}_metric.json")
        json.dump({"inv_metric": np.asarray(fit.metric).mean(axis=0).tolist()}, open(res["inv_metric_file"], "w"))
        res["stepsize_mean"] = float(np.mean(fit.step_size))
    except Exception as ex:  # noqa: BLE001
        res["inv_metric_file"] = f"unavailable: {ex}"
    save_json(res, ROOT / "diagnostics" / f"{tag}_sampling_diagnostics.json")
    if not keep_csv:
        shutil.rmtree(outdir, ignore_errors=True)
    return res, draws, idata


def load_draws(tag, ref=False):
    return dict(np.load((ref_scratch() if ref else scratch_dir()) / f"{tag}_draws.npz"))


# ====================================================================== posterior reconstruction
def site_components(D, sites, mc):
    """Phase 3B components plus the river-network terms: rn = v_tu + v_td (location terms). m includes rn."""
    out = B.site_components(D, sites, mc)
    S = B.ndraws(D); I = len(sites)
    rn = np.zeros((S, I))
    if mc.get("tu", 0):
        rn = rn + np.asarray(D["v_tu"], float)
    if mc.get("td", 0):
        rn = rn + np.asarray(D["v_td"], float)
    out["rn"] = rn
    out["m"] = out["m"] + rn
    out["u"] = np.asarray(D["theta"], float) - out["m"]
    return out


def delta_T(D, sites, sy, mc, n_gh=15, include_location=True, return_pi=False, include_rn=True):
    """Phase 3B delta_T (P3B_06 §1; frozen definition) with the river-network term treated as a location term, like the
    field w(s_i): a new site at the unit's location shares v_tu/v_td there; new u and omega marginalised exactly.
    include_rn=False (sensitivity): the river term is instead treated as a new-site effect and marginalised (variance added)."""
    S = B.ndraws(D)
    comp = site_components(D, sites, mc)
    Tm = sy.in_target_T.to_numpy() == 1
    sidx = sy.site_idx.to_numpy()[Tm] - 1; yrt = (sy.year_idx.to_numpy()[Tm] == 2).astype(float)
    w = sy.target_weight.to_numpy()[Tm]; w = w / w.sum()
    loc = comp["vA"] + comp["vC"] + comp["vB"] + comp["w"] + (comp["rn"] if include_rn else 0)
    if not include_location:
        loc = 0 * comp["fixed"]
    base = comp["fixed"][:, sidx] + loc[:, sidx] + B.get1(D, "gamma")[:, None] * yrt[None, :]
    dT = comp["delta_r"][:, sites.region_idx.to_numpy()[sidx] - 1]
    lsf = B.log_sf_siteyear(D, sites, sy, mc)[:, Tm]
    var_re = B.get1(D, "sigma_u") ** 2 + B.get1(D, "sigma_om") ** 2
    if not include_rn and mc.get("tu", 0):
        var_re = var_re + B.get1(D, "sigma_tu") ** 2      # sensitivity: river term treated as a NEW-site effect (marginalised)
    sd_re = np.sqrt(var_re)[:, None]
    sg = B.get1(D, "sigma")
    ghx, ghw = A.gh_nodes(n_gh); lam = B.get1(D, "lambda"); tsg = B.get1(D, "tau_sig"); zt = sites.z_logarea.to_numpy()[sidx]
    res = {}
    for c in THRESH:
        pR = np.zeros_like(base); pM = np.zeros_like(base)
        for xg, wg in zip(ghx, ghw):
            sn = sg[:, None] * np.exp(lam[:, None] * zt[None, :] + tsg[:, None] * xg)
            h = 0.5 * sn ** 2
            pR += wg * ndtr((base + lsf + h - np.log(c)) / sd_re)
            pM += wg * ndtr((base + lsf + dT + h - np.log(c)) / sd_re)
        PR, PM = pR @ w, pM @ w
        res[c] = dict(PR=PR, PM=PM, D=PM - PR, RR=PM / PR)
        if return_pi:
            res[c].update(pR=pR, pM=pM)
    return res


summarise_delta = B.summarise_delta
band = B.band


# ====================================================================== partial CmdStan CSV reader (budget-capped fits)
def load_partial_csv(outdir, warmup_saved=False):
    """Read CmdStan sampling CSVs that may be incomplete (a fit stopped at its CPU budget). Returns (D, n_per_chain):
    D maps each Stan variable to an array (S, *shape) with chains stacked (only complete rows), plus sampler columns."""
    import glob
    files = sorted(glob.glob(str(Path(outdir) / "*.csv")))
    frames, nper = [], []
    for f in files:
        df = pd.read_csv(f, comment="#")
        df = df.dropna()
        if "Elapsed Time" not in open(f).read()[-3000:]:
            df = df.iloc[:-1]                      # last row of an interrupted file may be truncated
        frames.append(df); nper.append(len(df))
    df = pd.concat(frames, ignore_index=True)
    D = {}
    groups = {}
    for col in df.columns:
        parts = col.split(".")
        groups.setdefault(parts[0], []).append((tuple(int(p) for p in parts[1:]), col))
    for name, lst in groups.items():
        if len(lst) == 1 and lst[0][0] == ():
            D[name] = df[lst[0][1]].to_numpy()
            continue
        shape = tuple(max(ix[k] for ix, _ in lst) for k in range(len(lst[0][0])))
        arr = np.empty((len(df),) + shape)
        for ix, col in lst:
            arr[(slice(None),) + tuple(i - 1 for i in ix)] = df[col].to_numpy()
        D[name] = arr
    return D, nper
