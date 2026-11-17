"""Correction stage, amendment AM2 (VC_00 §5): rescoring with the FROZEN Phase 4B scorer (scripts/vf_02_score.py, imported, not
modified), changing only n_draws. Writers are redirected to validation_final/correction/. A CPU watchdog stops the process at the
per-fold cap (vc_settings.yaml rescore_cap_cpu_h).
  vc_04_rescore.py score FOLD N_DRAWS [--r3 original|refit]  -> correction/tables/folds/{FOLD}_{site,pointwise,siteyear}.csv, {FOLD}_rb.json
  vc_04_rescore.py aggregate                                -> corrected four-fold aggregation (C1, C2 copied unchanged from Phase 4B)"""
import argparse
import json
import os
import shutil
import sys
import threading
import time
from pathlib import Path

import pandas as pd

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import vc_lib as K  # noqa: E402

V = K.V
K.to_correction()
sys.path.insert(0, str(K.VFDIR / "scripts"))
import vf_02_score as S  # noqa: E402  (frozen scorer; its import-time redirect now points at correction/)

FO = K.VC / "tables" / "folds"; FO.mkdir(parents=True, exist_ok=True)
STATE = {"r3": "original"}
_orig_fit_diag = S.fit_diag


def load_draws(tag):
    if tag == "vf_W1_R3" and STATE["r3"] == "refit":
        return K.new_draws("vc_W1_R3")
    return K.original_draws(tag)


def fit_diag(fold, model):
    """Frozen fit_diag with the diagnostics path resolved to the Phase 4B record, or to the refit for W1 when scoring the refit."""
    V.VF = K.VFDIR
    try:
        d = _orig_fit_diag(fold, model)
    finally:
        V.VF = K.VC
    if model == "m2rn" and fold == "W1" and STATE["r3"] == "refit":
        p = K.VC / "diagnostics" / "vc_W1_R3_sampling_diagnostics.json"
        r = json.load(open(p))
        lr = r.get("log_scale_diagnostics") or {}
        d.update(source=str(p.relative_to(K.V.TA)), cpu_h=r["cpu_h"], divergences=r["divergences"], divergence_share=r["divergence_share"],
                 min_ebfmi=float(min(r["ebfmi"])), hyper_max_rhat=r["hyper_max_rhat"], hyper_argmax_rhat=r["hyper_argmax_rhat"],
                 worst_log_scale_rhat=max([v["r_hat"] for v in lr.values()] or [1.0]), hyper_min_ess_bulk=r["hyper_min_ess_bulk"],
                 hyper_argmin_ess_bulk=r["hyper_argmin_ess_bulk"], hyper_min_ess_tail=r["hyper_min_ess_tail"], latent_max_rhat=r["latent_max_rhat"],
                 latent_min_ess_bulk=r["latent_min_ess_bulk"], latent_min_ess_tail=r["latent_min_ess_tail"], treedepth_hits=r["treedepth_hits"],
                 N_train=r["N_train"], held_out_sites=r["held_out_sites"])
    return d


S.V.load_draws = load_draws
S.fit_diag = fit_diag


def watchdog(cap_h, job):
    def run():
        while True:
            if K.cpu_used_self() >= cap_h:
                K.ledger(job + " STOPPED AT CPU CAP", note=f"cap {cap_h} CPU-h")
                os._exit(3)
            time.sleep(20)
    threading.Thread(target=run, daemon=True).start()


def score(fold, n, r3):
    STATE["r3"] = r3
    job = f"rescore {fold} ({'refit' if r3 == 'refit' else 'Phase 4B'} posterior) at {n} draws"
    watchdog(K.VCFG["rescore_cap_cpu_h"], job)
    meas, sites, sy = S.L.load_tables()
    o = S.score_fold(fold, sites, meas, sy, n)
    sub = FO if n == 1600 else (K.VC / "tables" / f"folds_{n}draws"); sub.mkdir(exist_ok=True)
    for j, k in enumerate(("site", "pointwise", "siteyear")):
        pd.concat([o["m1s2"][j], o["m2rn"][j]]).to_csv(sub / f"{fold}_{k}.csv", index=False)
    rb = dict(o.get("rb_check") or {}, n_draws=n, note=f"RB statistic at the {n} scoring draws (descriptive; the gates are VC_00 §4)")
    json.dump(rb, open(sub / f"{fold}_rb.json", "w"), indent=1)
    meta = dict(fold=fold, n_draws=n, r3_posterior=("vc_W1_R3" if (fold == "W1" and r3 == "refit") else f"vf_{fold}_R3"))
    json.dump(meta, open(sub / f"{fold}_meta.json", "w"), indent=1)
    K.ledger(job)


def aggregate():
    """Corrected four-fold aggregation with the frozen main(): W1/W2 from the rescored files, C1/C2 copied unchanged (400 draws)."""
    for f in ["C1", "C2"]:
        for k in ("site", "pointwise", "siteyear"):
            shutil.copy2(K.VFDIR / "tables/folds" / f"{f}_{k}.csv", FO / f"{f}_{k}.csv")
        shutil.copy2(K.VFDIR / "tables/folds" / f"{f}_rb.json", FO / f"{f}_rb.json")
    for f in ["W1", "W2"]:
        assert (FO / f"{f}_site.csv").exists(), f"{f} not rescored"
    STATE["r3"] = json.load(open(FO / "W1_meta.json"))["r3_posterior"] == "vc_W1_R3" and "refit" or "original"
    S.main(["W1", "W2", "C1", "C2"], 1600)
    src = K.VC / "diagnostics" / "VF_validation.json"; dst = K.VC / "diagnostics" / "VC_validation_corrected.json"
    d = json.load(open(src)); d["n_draws_by_fold"] = {"W1": 1600, "W2": 1600, "C1": 400, "C2": 400}
    d["note"] = "Correction stage (VC_00): W1/W2 rescored at 1,600 draws (AM2); C1/C2 Phase 4B scores unchanged (400 draws)."
    d["W1_posterior"] = json.load(open(FO / "W1_meta.json"))["r3_posterior"]
    K.save_json(d, dst); src.unlink()
    for nm in ["VF_scores_site", "VF_scores_pointwise", "VF_scores_siteyear", "VF_validation_fits", "VF_validation_folds", "VF_validation_strata"]:
        p = K.VC / "tables" / f"{nm}.csv"
        if p.exists():
            p.rename(K.VC / "tables" / f"{nm.replace('VF_', 'VC_')}_corrected.csv")
    K.ledger("aggregate corrected validation")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["score", "aggregate"]); ap.add_argument("fold", nargs="?")
    ap.add_argument("n", nargs="?", type=int, default=1600); ap.add_argument("--r3", default="original", choices=["original", "refit"])
    a = ap.parse_args()
    score(a.fold, a.n, a.r3) if a.mode == "score" else aggregate()
