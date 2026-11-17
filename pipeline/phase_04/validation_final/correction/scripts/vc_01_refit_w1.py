"""Correction stage: the single approved W1 refit (VC_00 §3). Same model, data, held-out sites, inits, pilot metric and seed as
vf_W1_R3; adapt_delta 0.99, warmup 600. Writes correction/diagnostics/vc_W1_R3_sampling_diagnostics.json,
correction/tables/vc_W1_R3_posterior_summary.csv, scratch phase_04_vc/vc_W1_R3_draws.npz and the adequacy verdict."""
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import vc_lib as K  # noqa: E402

V = K.V
rf = K.VCFG["refit"]
K.to_correction()
V.CFG["fits"][rf["tag"]] = dict(kind="fold", fold=rf["fold"], model=rf["model"])
V.CFG["r3_sampling"] = dict(chains=rf["chains"], parallel_chains=rf["parallel_chains"], iter_warmup=rf["iter_warmup"],
                            iter_sampling=rf["iter_sampling"], adapt_delta=rf["adapt_delta"], max_treedepth=rf["max_treedepth"])
orig = json.load(open(K.VFDIR / "diagnostics" / "vf_W1_R3_sampling_diagnostics.json"))
res, draws = V.run_fit(rf["tag"], seed=rf["seed"])
assert sorted(res["held_out_sites"]) == sorted(orig["held_out_sites"]), "held-out set differs from the original W1 fit"
ad = K.VCFG["adequacy_fold"]
rhat = max([res["hyper_max_rhat"]] + [v["r_hat"] for v in (res.get("log_scale_diagnostics") or {}).values()])
checks = dict(divergences=res["divergence_share"] <= ad["max_divergence_share"], ebfmi=min(res["ebfmi"]) >= ad["min_ebfmi"],
              rhat=rhat <= ad["max_rhat_hyper"])
verdict = dict(tag=rf["tag"], checks=checks, sampler_pass=bool(all(checks.values())), divergences=res["divergences"],
               divergence_share=res["divergence_share"], divergences_by_chain=res.get("divergences_by_chain"), min_ebfmi=min(res["ebfmi"]),
               max_rhat_incl_log=rhat, hyper_min_ess_bulk=res["hyper_min_ess_bulk"], hyper_argmin_ess_bulk=res["hyper_argmin_ess_bulk"],
               hyper_min_ess_tail=res["hyper_min_ess_tail"], cpu_h=res["cpu_h"], same_inputs_as=orig["tag"],
               same_seed=bool(orig["seed"] == res["seed"]), same_executable=bool(orig["executable_sha256"] == res["executable_sha256"]))
K.save_json(verdict, K.VC / "diagnostics" / "VC_W1_refit_adequacy.json")
K.ledger("W1 refit (vc_W1_R3)", cpu_h=res["cpu_h"], note="CPU of the CmdStan chains (cmdstanpy children)")
print(json.dumps(verdict, indent=1, default=str), flush=True)
print(json.dumps({k: res.get(k) for k in ["stepsize_by_chain", "mean_leapfrog", "mean_treedepth", "treedepth_hits", "latent_max_rhat",
                                          "latent_min_ess_bulk", "latent_min_ess_tail", "log_scale_diagnostics"]}, indent=1, default=str), flush=True)
