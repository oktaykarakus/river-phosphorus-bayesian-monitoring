"""VR_07 (stage S2): the W1 refit, VC_00 §3 specification verbatim (settings read from correction/config/vc_settings.yaml and
checked against repair/config/vr_w1_final.yaml). Same executable, data, held-out sites, inits, pilot metric, initial step and seed
20261115 as the terminated run vc_W1_R3. New tag vr_W1_R3; writes only to repair/ and ~/track_a_scratch/phase_04_vr.
After the fit: the frozen sampler rules (gate) and an informational determinism check (first 440 / 518 draws of chains 1-2 vs
the preserved partial CSVs). Exit code 0 = sampler rules pass, 3 = fail. Writes repair/diagnostics/VR_gate_S2.json."""
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import vr_final_lib as K

V = K.V
vc = yaml.safe_load(open(K.VFDIR / "correction/config/vc_settings.yaml"))["refit"]
rf = K.CFG["refit"]
for k in ["fold", "model", "seed", "chains", "parallel_chains", "iter_warmup", "iter_sampling", "adapt_delta", "max_treedepth"]:
    assert vc[k] == rf[k], f"setting {k} differs from VC_00: {vc[k]} vs {rf[k]}"


def first_rows(path, n):
    rows = []
    with open(path) as f:
        hdr = None
        for ln in f:
            if ln.startswith("#"):
                continue
            if hdr is None:
                hdr = ln; continue
            rows.append(ln.rstrip("\n"))
            if len(rows) == n:
                break
    return rows


def main():
    K.to_repair()
    V.CFG["fits"][rf["tag"]] = dict(kind="fold", fold=rf["fold"], model=rf["model"])
    V.CFG["r3_sampling"] = dict(chains=rf["chains"], parallel_chains=rf["parallel_chains"], iter_warmup=rf["iter_warmup"],
                                iter_sampling=rf["iter_sampling"], adapt_delta=rf["adapt_delta"], max_treedepth=rf["max_treedepth"])
    orig = json.load(open(K.VFDIR / "diagnostics/vf_W1_R3_sampling_diagnostics.json"))
    res, draws = V.run_fit(rf["tag"], seed=rf["seed"])
    assert sorted(res["held_out_sites"]) == sorted(orig["held_out_sites"])
    ad = K.CFG["adequacy_fold"]
    rhat = max([res["hyper_max_rhat"]] + [v["r_hat"] for v in (res.get("log_scale_diagnostics") or {}).values()])
    checks = dict(divergences=res["divergence_share"] <= ad["max_divergence_share"], ebfmi=min(res["ebfmi"]) >= ad["min_ebfmi"],
                  rhat=rhat <= ad["max_rhat_hyper"])
    # determinism (informational): a fresh run with the same seed should regenerate the preserved partial draws
    part = K.VFDIR / "correction/partial_W1_refit"
    new = sorted((K.SCR / f"fit_{rf['tag']}").glob("*_[1-4].csv"))
    det = {}
    for k, n in [(1, 440), (2, 518)]:
        old = first_rows(part / f"track_a_m2rn_r3-20261116075950_{k}.csv", n)
        nw = first_rows([p for p in new if p.stem.endswith(f"_{k}")][0], n)
        det[f"chain_{k}"] = dict(rows=n, identical=bool(old == nw), n_identical=int(sum(a == b for a, b in zip(old, nw))))
    verdict = dict(stage="S2", tag=rf["tag"], checks=checks, pass_=bool(all(checks.values())), divergences=res["divergences"],
                   divergence_share=res["divergence_share"], divergences_by_chain=res.get("divergences_by_chain"), min_ebfmi=min(res["ebfmi"]),
                   max_rhat_incl_log=rhat, hyper_argmax_rhat=res.get("hyper_argmax_rhat"), hyper_min_ess_bulk=res["hyper_min_ess_bulk"],
                   hyper_min_ess_tail=res["hyper_min_ess_tail"], stepsize_by_chain=res.get("stepsize_by_chain"),
                   mean_treedepth=res.get("mean_treedepth"), treedepth_hits=res.get("treedepth_hits"), cpu_h=res["cpu_h"],
                   determinism_vs_terminated_run=det, same_seed=bool(orig["seed"] == res["seed"]),
                   same_executable=bool(orig["executable_sha256"] == res["executable_sha256"]))
    verdict["pass"] = verdict.pop("pass_")
    K.save_json(verdict, K.DIAG / "VR_gate_S2.json")
    K.ledger("S2 W1 refit vr_W1_R3", cpu_h=res["cpu_h"], note="CmdStan chains (cmdstanpy children)")
    print(json.dumps(verdict, indent=1, default=str), flush=True)
    sys.exit(0 if verdict["pass"] else 3)


if __name__ == "__main__":
    main()
