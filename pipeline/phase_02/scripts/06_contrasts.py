"""O5: reproduce the Pilot 01 baseline and reassess the MS - RSN exceedance contrast after accounting for river
size, geography, catchment structure, storm-overflow pressure, year and sampling frequency.

All contrasts are ABSOLUTE differences in exceedance probability, MS minus RSN, for annual-mean reactive
orthophosphate (mean_hi, as Pilot 01) above c in {0.05, 0.10, 0.20} mg P/L, primary eligibility rule.
Every standardised contrast states its TARGET distribution:
  T_RSN    covariate distribution of RSN primary site-years
  T_MS     covariate distribution of MS primary site-years
  T_POOL   pooled RSN + MS site-years
  T_FRAME  river-length distribution of the OS Open Rivers inland-river frame (F1), restricted to the size
           range with observed support in BOTH networks (model-based; years weighted equally)
Uncertainty: spatial block bootstrap (25 km BNG cells; all site-years of both networks in a block are
resampled together, so repeated years of a site stay together), B = 1000 (main) / 400 (sensitivity).
Run from phase_02/:  python -I scripts/06_contrasts.py
"""
import os, sys, time
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")   # tiny repeated solves: multithreaded BLAS is ~100x slower here
sys.path.insert(0, __import__("os").path.dirname(__file__))
from common import *  # noqa

import pyogrio
import shapely

B_MAIN, B_SENS = 1000, 400
rng = np.random.default_rng(SEED)
WMAS = sorted(set(AREA_TO_WMA.values()))


# ------------------------------------------------------------------ logistic regression (IRLS, light ridge)
def logit_fit(X, y, w=None, lam=1e-4, iters=60):
    n, p = X.shape
    w = np.ones(n) if w is None else w
    beta = np.zeros(p)
    R = lam * np.eye(p); R[0, 0] = 0
    for _ in range(iters):
        eta = np.clip(X @ beta, -30, 30); mu = 1 / (1 + np.exp(-eta)); W = w * mu * (1 - mu)
        H = X.T @ (X * W[:, None]) + R
        g = X.T @ (w * (y - mu)) - R @ beta
        step = np.linalg.solve(H, g); beta += step
        if np.max(np.abs(step)) < 1e-8:
            break
    return beta


def expit(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


class Design:
    """Fixed-level design matrix so bootstrap replicates share columns."""

    def __init__(self, spec, ref):
        self.spec = spec
        self.lev = {}
        for f in spec.get("factors", []):
            self.lev[f] = sorted(ref[f].dropna().astype(str).unique())
        self.cont = spec.get("cont", [])
        self.inter = spec.get("ms_x", [])            # continuous terms interacted with ms
        self.inter_f = spec.get("ms_xf", [])         # factors interacted with ms

    def X(self, d, ms=None):
        n = len(d)
        msv = d["ms"].values if ms is None else np.full(n, ms)
        cols = [np.ones(n), msv]
        for c in self.cont:
            cols.append(d[c].values.astype(float))
        for f, levs in self.lev.items():
            v = d[f].astype(str).values
            for l in levs[1:]:
                cols.append((v == l).astype(float))
        for c in self.inter:
            cols.append(msv * d[c].values.astype(float))
        for f in self.inter_f:
            v = d[f].astype(str).values
            for l in self.lev[f][1:]:
                cols.append(msv * (v == l))
        return np.column_stack(cols)


def gcomp(d, y, design, targets):
    """Fit on d; return dict target -> (p_ms, p_rsn, diff) averaged over target rows (with optional weights)."""
    X = design.X(d)
    beta = logit_fit(X, y)
    res = {}
    for name, (T, wt) in targets.items():
        p1 = expit(design.X(T, 1) @ beta); p0 = expit(design.X(T, 0) @ beta)
        wt = np.ones(len(T)) if wt is None else wt
        a, b = np.average(p1, weights=wt), np.average(p0, weights=wt)
        res[name] = (a, b, a - b)
    return res


# ------------------------------------------------------------------ direct standardisation
def std_contrast(d, y, cell, target_w):
    """Direct standardisation over `cell`; target_w: Series cell -> weight. Uses cells with both networks.
    Returns (p_ms, p_rsn, diff, retained target weight share)."""
    t = pd.DataFrame({"cell": d[cell].astype(str).values, "ms": d["ms"].values, "y": y})
    m = t.groupby(["cell", "ms"]).y.mean().unstack()
    m = m.reindex(columns=[0, 1])
    both = m.dropna().index
    w = target_w.reindex(both).fillna(0)
    if w.sum() == 0:
        return (np.nan, np.nan, np.nan, 0.0)
    a = float((m.loc[both, 1] * w).sum() / w.sum()); b = float((m.loc[both, 0] * w).sum() / w.sum())
    return (a, b, a - b, float(w.sum() / target_w.sum()))


def pilot_std_frac(d, c, w):
    """Exact Pilot 01 estimator (03b_adjusted_differences.std_frac) for reproduction."""
    if w is None:
        return float((d["mean_hi"] > c).mean())
    pa = d[d.area.isin(w.index)].groupby("area")["mean_hi"].apply(lambda s: (s > c).mean())
    pa = pa.reindex(w.index); ok = pa.notna()
    return float((pa[ok] * w[ok]).sum() / w[ok].sum())


# ------------------------------------------------------------------ bootstrap engine
def block_index(d, col):
    g = {k: np.flatnonzero(d[col].values == k) for k in pd.unique(d[col])}
    return list(g.keys()), g


def boot(d, fn, B, col="block"):
    keys, g = block_index(d, col)
    out = []
    for _ in range(B):
        ks = rng.choice(len(keys), len(keys), replace=True)
        rows = np.concatenate([g[keys[k]] for k in ks])
        out.append(fn(d.iloc[rows].reset_index(drop=True)))
    return out


def summarise(point, reps):
    reps = np.asarray(reps, float)
    return dict(est=point, lo=float(np.nanquantile(reps, .025)), hi=float(np.nanquantile(reps, .975)), se=float(np.nanstd(reps)))


# ------------------------------------------------------------------ frame target
def frame_target(d):
    """Length-weighted sample of frame F1 links restricted to the jointly supported size range."""
    A = pd.read_parquet(PROC / "links_gb_attributes.parquet", columns=["id", "in_england", "F1_inland_river", "wma", "typology_geology", "len_england_m"])
    Z = pd.read_parquet(PROC / "links_size.parquet", columns=["id", "la_hat"])
    G = pyogrio.read_dataframe(PROC / "frame_links_england.gpkg", columns=["id"])
    F = A[A.F1_inland_river & A.wma.notna()].merge(Z, on="id").merge(G, on="id")
    lo = max(d[d.ms == 1].log_area.quantile(.01), d[d.ms == 0].log_area.quantile(.01))
    hi = min(d[d.ms == 1].log_area.quantile(.99), d[d.ms == 0].log_area.quantile(.99))
    share_in = float(F.loc[F.la_hat.between(lo, hi), "len_england_m"].sum() / F.len_england_m.sum())
    Fs = F[F.la_hat.between(lo, hi)].reset_index(drop=True)
    r = np.random.default_rng(SEED + 1)
    k = r.choice(len(Fs), 20000, p=Fs.len_england_m.values / Fs.len_england_m.sum())
    T = Fs.iloc[k].reset_index(drop=True)
    mid = shapely.line_interpolate_point(gpdgeom(T), 0.5, normalized=True)
    edm = pd.read_csv(PILOT / "data/edm_outlets_2024.csv").dropna(subset=["easting", "northing"])
    tree = shapely.STRtree(shapely.points(edm.easting.values, edm.northing.values))
    i, _ = tree.query(mid, predicate="dwithin", distance=5000)
    cnt = np.bincount(i, minlength=len(T))
    T = pd.DataFrame({"wma": T.wma.values, "log_area": T.la_hat.values, "log_area2": T.la_hat.values ** 2, "geology": T.typology_geology.fillna("Unknown").replace({"Salt": "Unknown"}).values,
                      "log_edm5": np.log1p(cnt), "year": r.choice([2023, 2024], len(T)), "log_n": np.nan})
    return T, dict(log_area_support=(float(np.exp(lo)), float(np.exp(hi))), frame_length_share_in_support=share_in)


def gpdgeom(T):
    return T.geometry.values


def main():
    ensure_dirs()
    t0 = time.time()
    P = pd.read_csv(PROC / "site_years_p2.csv")
    P["block"] = blocks(P.easting, P.northing, 25_000)
    P["block10"] = blocks(P.easting, P.northing, 10_000)
    P["block50"] = blocks(P.easting, P.northing, 50_000)
    P["mancat_c"] = P.mancat.fillna("NA_" + P.block)
    P["log_n"] = np.log(P.n)
    P["log_area2"] = P.log_area ** 2
    P["strahler_band"] = pd.cut(P.link_strahler, [0, 2, 3, 4, 99], labels=["1-2", "3", "4", "5+"]).astype(str)
    P["typo_size"] = P.size_typology.fillna("none").astype(str)
    P["cell_ws"] = P.wma + "|" + P.size_cls
    out_rows, info = [], {}

    # ================= 1. baseline reproduction (all primary RSN/MS site-years, pilot estimator)
    rep = []
    for c in THRESH:
        a = P[P.ms == 0]; b = P[P.ms == 1]
        pr, pm = pilot_std_frac(a, c, None), pilot_std_frac(b, c, None)
        er, em, ed = PILOT_BASELINE[c]
        rep.append(dict(threshold=c, n_rsn=len(a), n_ms=len(b), p_rsn=pr, p_ms=pm, diff=pm - pr, pilot_rsn=er, pilot_ms=em, pilot_diff=ed,
                        abs_err_rsn=abs(round(pr, 3) - er), abs_err_ms=abs(round(pm, 3) - em), abs_err_diff=abs(round(pm - pr, 3) - ed)))
    # Pilot D2 (RSN composition over pilot areas with >=10 sites in both networks)
    rsnP, msP = P[P.ms == 0], P[P.ms == 1]
    areas = sorted(set(rsnP.groupby("area").site.nunique().loc[lambda s: s >= 10].index) & set(msP.groupby("area").site.nunique().loc[lambda s: s >= 10].index))
    w_rsn_area = rsnP[rsnP.area.isin(areas)].groupby("area").size(); w_rsn_area = w_rsn_area / w_rsn_area.sum()
    pilot_d2 = pd.read_csv(PILOT / "results/p5_adjusted_differences.csv")
    for c in THRESH:
        dd = pilot_std_frac(msP, c, w_rsn_area) - pilot_std_frac(rsnP, c, w_rsn_area)
        ref = float(pilot_d2[(pilot_d2.estimator == "D2 primary, std RSN composition") & np.isclose(pilot_d2.threshold, c)]["diff"].iloc[0])
        rep.append(dict(threshold=c, estimator="Pilot D2 std RSN area composition", diff=dd, pilot_diff=ref, abs_err_diff=abs(dd - ref)))
    rep = pd.DataFrame(rep); rep.to_csv(TAB / "p2_04_baseline_reproduction.csv", index=False)
    info["baseline_reproduced_within_0.0005"] = bool((rep.abs_err_diff.fillna(0) <= 5e-4 + 1e-12).all() and (rep.abs_err_rsn.fillna(0) <= 5e-4).all() and (rep.abs_err_ms.fillna(0) <= 5e-4).all())
    log(rep.round(4).to_string())

    # ================= analysis sample
    D = P[(P.cls != "U") & P.log_area.notna() & P.log_edm5.notna()].copy()
    info["n_siteyears_dropped_missing_size_or_covariates"] = int(((P.cls != "U") & (P.log_area.isna() | P.log_edm5.isna())).sum())
    D["geology"] = D.geology.fillna("Unknown")
    info["n_siteyears_analysis"] = {"RSN": int((D.ms == 0).sum()), "MS": int((D.ms == 1).sum())}
    info["n_siteyears_dropped_unmatched"] = int((P.cls == "U").sum())
    T_frame, finfo = frame_target(D)
    info["frame_target"] = finfo

    specs = {
        "M1 size": dict(cont=["log_area", "log_area2"], ms_x=["log_area", "log_area2"], factors=["year"]),
        "M2 size+WMA+year": dict(cont=["log_area", "log_area2"], ms_x=["log_area", "log_area2"], factors=["wma", "year"]),
        "M3 size+WMA+year+EDM+geology": dict(cont=["log_area", "log_area2", "log_edm5"], ms_x=["log_area", "log_area2"], factors=["wma", "year", "geology"]),
        "M4 M3+log n (frequency)": dict(cont=["log_area", "log_area2", "log_edm5", "log_n"], ms_x=["log_area", "log_area2"], factors=["wma", "year", "geology"]),
        "M5 M3 no network x size interaction": dict(cont=["log_area", "log_area2", "log_edm5"], ms_x=[], factors=["wma", "year", "geology"]),
    }
    designs = {k: Design(v, D) for k, v in specs.items()}

    def all_estimates(d, T_frame_local=T_frame, which="main", size_var="size_cls"):
        r = {}
        rsn = d[d.ms == 0]
        for c in THRESH:
            y = (d.mean_hi > c).astype(float).values
            r[("U unadjusted", "observed", c)] = y[d.ms.values == 1].mean() - y[d.ms.values == 0].mean()
            wW = rsn.groupby("wma").size().astype(float)
            r[("G WMA-standardised", "T_RSN", c)] = std_contrast(d, y, "wma", wW)[2]
            wS = rsn.groupby(size_var).size().astype(float)
            r[("S size-standardised", "T_RSN", c)] = std_contrast(d, y, size_var, wS)[2]
            d_cell = d.wma + "|" + d[size_var].astype(str)
            dd = d.assign(cell=d_cell)
            wC = dd[dd.ms == 0].groupby("cell").size().astype(float)
            sc = std_contrast(dd, y, "cell", wC)
            r[("GS WMA x size-standardised", "T_RSN", c)] = sc[2]
            r[("GS retained target share", "T_RSN", c)] = sc[3]
            wM = d[d.ms == 0].groupby("mancat_c").size().astype(float)
            sm_ = std_contrast(d, y, "mancat_c", wM)
            r[("C within-management-catchment", "T_RSN", c)] = sm_[2]
            r[("C retained target share", "T_RSN", c)] = sm_[3]
            if which == "main":
                wF = None
            tg = {"T_RSN": (d[d.ms == 0], None), "T_MS": (d[d.ms == 1], None), "T_POOL": (d, None)}
            for mname, des in designs.items():
                if d.log_area.isna().any():
                    break
                if which != "main" and mname not in ("M3 size+WMA+year+EDM+geology",):
                    continue
                tgt = dict(tg)
                if mname in ("M2 size+WMA+year", "M3 size+WMA+year+EDM+geology", "M5 M3 no network x size interaction") and T_frame_local is not None:
                    tgt["T_FRAME"] = (T_frame_local, None)
                g = gcomp(d, y, des, tgt)
                for tn, v in g.items():
                    r[(mname, tn, c)] = v[2]
                    r[(mname + " |p_MS", tn, c)] = v[0]; r[(mname + " |p_RSN", tn, c)] = v[1]
        return r

    # ================= 2. main estimates with block bootstrap
    log(f"main estimates ... ({time.time()-t0:.0f}s)")
    pt = all_estimates(D)
    reps = boot(D, lambda x: all_estimates(x), B_MAIN)
    rows = []
    for k, v in pt.items():
        s = summarise(v, [r_.get(k, np.nan) for r_ in reps])
        rows.append(dict(estimator=k[0], target=k[1], threshold=k[2], **s))
    main = pd.DataFrame(rows); main.to_csv(TAB / "p2_04_contrasts_main.csv", index=False)
    log(main[~main.estimator.str.contains(r"\|p_|retained")].round(3).to_string())
    log(f"main done ({time.time()-t0:.0f}s)")

    # ================= 3. heterogeneity: size class, region (raw and M3 conditional on class)
    het = []
    for var in ["size_cls", "region", "typo_size"]:
        for lev, g in D.groupby(var):
            if (g.ms == 0).sum() < 15 or (g.ms == 1).sum() < 15:
                continue
            for c in THRESH:
                f = lambda x, c=c: float((x[x.ms == 1].mean_hi > c).mean() - (x[x.ms == 0].mean_hi > c).mean())
                bs = boot(g.reset_index(drop=True), f, B_SENS)
                het.append(dict(by=var, level=lev, threshold=c, n_rsn=int((g.ms == 0).sum()), n_ms=int((g.ms == 1).sum()),
                                p_rsn=float((g[g.ms == 0].mean_hi > c).mean()), p_ms=float((g[g.ms == 1].mean_hi > c).mean()), **summarise(f(g), bs)))
    # model-based size-specific contrasts (M3, target = RSN site-years within each size class)
    des = designs["M3 size+WMA+year+EDM+geology"]
    def size_specific(d):
        r = {}
        for c in THRESH:
            y = (d.mean_hi > c).astype(float).values
            beta = logit_fit(des.X(d), y)
            for lev in SIZE_LABELS:
                T = d[(d.ms == 0) & (d.size_cls == lev)]
                if len(T) < 5:
                    continue
                r[(lev, c)] = float(np.mean(expit(des.X(T, 1) @ beta)) - np.mean(expit(des.X(T, 0) @ beta)))
        return r
    pts = size_specific(D); bs = boot(D, size_specific, B_SENS)
    for k, v in pts.items():
        het.append(dict(by="size_cls (M3 g-comp, T_RSN within class)", level=k[0], threshold=k[1], **summarise(v, [b.get(k, np.nan) for b in bs])))
    pd.DataFrame(het).to_csv(TAB / "p2_04_heterogeneity.csv", index=False)
    log(f"heterogeneity done ({time.time()-t0:.0f}s)")

    # interaction test (cluster-robust Wald, statsmodels; clusters = 25 km blocks)
    import statsmodels.formula.api as smf, statsmodels.api as sm
    itest = []
    for c in THRESH:
        D["y"] = (D.mean_hi > c).astype(int)
        f1 = "y ~ ms*(log_area + log_area2) + log_edm5 + C(wma) + C(year) + C(geology)"
        m1 = smf.glm(f1, data=D, family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(D.block)[0]})
        wt = m1.wald_test("ms:log_area = 0, ms:log_area2 = 0", scalar=True)
        itest.append(dict(threshold=c, wald_chi2=float(wt.statistic), df=2, p=float(wt.pvalue),
                          coef_ms=float(m1.params["ms"]), coef_ms_log_area=float(m1.params["ms:log_area"])))
    pd.DataFrame(itest).to_csv(TAB / "p2_04_interaction_test.csv", index=False)

    # ================= 4. sensitivity analyses (M3 / GS / S at T_RSN; B_SENS)
    sens = []
    def run(label, d, size_var="size_cls", col="block", frame=None):
        d = d.reset_index(drop=True)
        if size_var != "size_cls":
            d = d.copy(); d["size_alt"] = d[size_var].astype(str)
        sv = "size_cls" if size_var == "size_cls" else "size_alt"
        p_ = all_estimates(d, T_frame_local=frame, which="sens", size_var=sv)
        b_ = boot(d, lambda x: all_estimates(x, T_frame_local=frame, which="sens", size_var=sv), B_SENS, col=col)
        for k, v in p_.items():
            if k[0].startswith(("U ", "S ", "GS WMA", "C within", "M3")) and k[1] in ("observed", "T_RSN", "T_POOL") and "|p_" not in k[0]:
                sens.append(dict(analysis=label, estimator=k[0], target=k[1], threshold=k[2], n=len(d), **summarise(v, [r_.get(k, np.nan) for r_ in b_])))
    run("Base (25 km blocks)", D)
    run("Blocks 10 km", D, col="block10")
    run("Blocks 50 km", D, col="block50")
    run("Clusters = management catchment", D, col="mancat_c")
    run("Size = WFD typology class (official, water-body level)", D[D.typo_size != "none"], size_var="typo_size")
    run("Size = Strahler band (OS topology)", D, size_var="strahler_band")
    run("Matching class A only", D[D.cls == "A"])
    run("Matching class A+B", D[D.cls.isin(["A", "B"])])
    run("Exclude non-inlandRiver links (lake/tidal/canal)", D[D.link_form == "inlandRiver"])
    run("Exclude catchment-inconsistent matches", D[D.catchment_consistent == True])
    run("Eligibility: strict (n>=10, 4 quarters)", D[(D.n >= 10) & (D.n_quarters == 4)])
    run("Common support: PS in [0.05, 0.95]", D[D.ps.between(.05, .95)])
    run("Common support: PS in [0.10, 0.90]", D[D.ps.between(.10, .90)])
    run("Size support: log-area within both networks' 1-99%", D[D.log_area.between(max(D[D.ms == 1].log_area.quantile(.01), D[D.ms == 0].log_area.quantile(.01)),
                                                                              min(D[D.ms == 1].log_area.quantile(.99), D[D.ms == 0].log_area.quantile(.99)))])
    run("Year 2023 only", D[D.year == 2023]); run("Year 2024 only", D[D.year == 2024])
    # off-OS-network sites (no network size) kept as their own size class; model-based estimators not defined
    Poff = P.copy(); Poff["geology"] = Poff.geology.fillna("Unknown")
    Poff["size_off"] = np.where(Poff.cls == "U", "off-OS-network", Poff.size_cls.astype(str))
    Poff = Poff[Poff.wma.notna()]
    run("All primary site-years, off-OS-network as own size class", Poff, size_var="size_off")
    # size class recomputed from the PUBLISHED OS flow direction (no topology repair), same calibration
    import json as _j
    cp = _j.load(open(PROC / "area_calibration_model.json"))
    la_os = cp["params"]["Intercept"] + cp["params"]["lu"] * np.log(np.maximum(D.link_up_chan_km_mid_osdir.astype(float), 0.05))
    for lev in cp["geology_levels"]:
        la_os = la_os + cp["params"].get(f"C(geology)[T.{lev}]", 0.0) * (D.geology == lev)
    run("Size from published OS flow direction (no topology repair)", D.assign(size_os=size_class(np.exp(la_os.values)).astype(str).values), size_var="size_os")
    sens = pd.DataFrame(sens); sens.to_csv(TAB / "p2_04_sensitivity.csv", index=False)
    log(f"sensitivity done ({time.time()-t0:.0f}s)")

    # ================= 5. sampling-frequency diagnostics
    fq = []
    for net, g in D.groupby("network"):
        fq.append(dict(network=net, spearman_n_vs_mean=float(g[["n", "mean_hi"]].corr("spearman").iloc[0, 1]),
                       spearman_n_vs_logarea=float(g[["n", "log_area"]].corr("spearman").iloc[0, 1]),
                       median_n=float(g.n.median()), share_monthly=float((g.n >= 11).mean())))
    pd.DataFrame(fq).to_csv(TAB / "p2_04_frequency_diagnostics.csv", index=False)
    info["runtime_s"] = time.time() - t0
    dump_json(info, TAB / "p2_04_info.json")
    log(info)


if __name__ == "__main__":
    main()
