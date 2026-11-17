# Where We Monitor Matters: Monitoring-Network Composition and Bayesian River-Network Inference for Phosphorus Exceedance in English Rivers

**Authors:** Oktay Karakuş¹ and Can Eyüpoğlu¹ ²

¹ Department of Computer Science and Informatics, Cardiff University, Cardiff, UK
² Department of Computer Engineering, Turkish Air Force Academy, National Defence University, Istanbul, Türkiye

This repository holds the research code, model specifications, configurations and numerical result summaries for the paper above. The paper is prepared for submission to *Ecological Indicators*. The manuscript itself is not included.

## Abstract

Routine water-quality monitoring sites are selected purposively, whereas probability-based surveillance networks sample river reaches by design. We compared England's probability-based River Surveillance Network (RSN; 679 sites) with routine statutory monitoring (1,153 sites) for reactive phosphorus in 2023–2024. A Bayesian measurement-level model accounted for observation-specific reporting limits, defined exceedance using the latent calendar-year arithmetic mean, and represented geographical structure through management units, a smooth spatial field and a tail-up river-network covariance. Routine site-years exceeded 0.05, 0.10 and 0.20 mg P L⁻¹ 16–22 percentage points more often than RSN site-years. After standardisation to a common set of 2,256 site-years, the differences narrowed to 2–3 percentage points (0.025, 95% CrI 0.004–0.050, at 0.10 mg P L⁻¹), corresponding to a routine-site offset of approximately 8% in concentration. Accounting for river-network dependence halved the contrast remaining after hierarchical and spatial adjustment, because routine sites concentrate on a few phosphorus-enriched river systems. The residual contrast would disappear if unmonitored reaches were approximately 10% more enriched than the RSN sites representing them. In held-out comparisons that did not satisfy all pre-specified validation criteria, the river-network term provided no material predictive advantage in unmonitored catchments, while an apparent improvement on monitored rivers remained unconfirmed. Across 68,822 km of mapped rivers, the model-based probability that the reference-level annual mean exceeded 0.10 mg P L⁻¹ was 0.35 (95% CrI 0.33–0.37), compared with 0.46 across monitored site-years; plausible residual selection shifted this estimate by approximately ±0.06. Network disagreement mainly reflects where networks sample, so assessments should define their target population.

**Keywords:** river water quality; phosphorus; monitoring network design; preferential sampling; spatial stream-network models; Bayesian hierarchical models; selection sensitivity

## Citation

The paper has not yet been published, and no DOI exists yet. Until it is published, cite it as:

> Karakuş, O., Eyüpoğlu, C. Where We Monitor Matters: Monitoring-Network Composition and Bayesian River-Network Inference for Phosphorus Exceedance in English Rivers. Manuscript in preparation for *Ecological Indicators*.

Code repository: <https://github.com/oktaykarakus/river-phosphorus-bayesian-monitoring>. Machine-readable metadata is in [`CITATION.cff`](CITATION.cff). Publication details and a persistent identifier will be added once they exist.

## Scientific overview

The study compares two monitoring networks in England:
- the probability-based River Surveillance Network (RSN);
- routine statutory monitoring.

It uses reactive phosphorus (Environment Agency determinand 0180) for 2023–2024.

The model works at the level of individual measurements. It has five main parts:
1. **Censoring:** lognormal measurements, each censored at its own reporting limit.
2. **Annual mean:** a latent calendar-year arithmetic mean, on which exceedance of 0.05, 0.10 and 0.20 mg P L⁻¹ is defined.
3. **Site-level structure:** a hierarchy of management units, a Hilbert-space Gaussian-process spatial field, and a tail-up river-network covariance in exact directed-acyclic-graph form.
4. **Site-level variance:** the independent-site, water-body and river-network effects are integrated out analytically.
5. **Network offset:** a routine-site offset δ.

The inferential targets are:
- a **standardised network contrast** Δ_T over 2,256 common site-years;
- **sensitivity to unobserved selection**, through the parameters ψ and κ;
- a **river-length-weighted exceedance** B̄ over a 68,822 km mapped river frame.

A held-out comparison checks the predictive contribution of the river-network term.

## Repository structure

```
README.md            this file
LICENSE              MIT License
CITATION.cff         citation metadata
requirements.txt     Python dependencies (versions of the analysis environment)
MANIFEST.md          file list with SHA-256 checksums
data/README.md       data sources, licences and how to obtain them
results/             numerical summaries behind the paper's tables and figures (see results/README.md)
pipeline/            analysis code, in the original stage layout (scripts locate each other by relative path)
  pilot_01/          data acquisition and site-year dataset
  phase_02/          river frame, river size, site matching, coverage, raw contrasts
  phase_02b/         modelling specification: priors, settings, standardisation constants
  phase_03a/         shared data tables, priors, annual-mean helpers (library)
  phase_03b/         comparison model without the river-network term (Stan) and library
  phase_03c/         river-network topology, tail-up covariance, held-out folds and scores (library)
  phase_03c_r/       reparameterised model library
  phase_03c_f/       final model (R3), main posterior fit and its continuation (addendum_T1)
  phase_04/          estimands, selection sensitivity, river-length inference, checks, figures;
                     validation_final/: held-out fold fits, scoring, corrections and the final validation attempt
```

The **authoritative model** is `pipeline/phase_03c_f/stan/track_a_m2rn_r3.stan`. The comparison model without the river-network term is `pipeline/phase_03b/stan/track_a_m1.stan`.

Three earlier Stan programs are still included because the shared libraries refer to them:
- `track_a_measurement.stan`
- `track_a_m2rn.stan`
- `track_a_m2rn_r.stan`

They are not used for the reported results.

## Installation

1. Use Python 3.12 and install the dependencies with `pip install -r requirements.txt`. The geospatial stack (geopandas, pyogrio, shapely) is easiest to install through conda-forge.
2. Install **CmdStan 2.40** through CmdStanPy (`python -c "import cmdstanpy; cmdstanpy.install_cmdstan(version='2.40.0')"`).

The original analysis ran on macOS. The final W1 refit was also run on an x86 Linux cluster; its job scripts are not included because they are specific to that cluster.

## Reproducibility workflow

Run each script from its stage directory, as the scripts use relative paths. Large intermediate files, including CmdStan CSVs, draw archives and the river frame, are written to `~/track_a_scratch/<stage>/`, as set in each stage's `config/*.yaml`. Only some scripts create their output folders. Before running a stage, create its `data/`, `raw/`, `metadata/`, `data_raw/`, `data_processed/`, `tables/`, `diagnostics/`, `figures/` and `provenance/` folders where they are used. The table below lists the steps in order; one step's outputs are the next step's inputs.

| Step | Scripts (in `pipeline/`) | Purpose | Inputs |
|---|---|---|---|
| 1 Data acquisition | `pilot_01/scripts/01_download_wqa.py`, `02a_extract_rsn_sites.py`, `02_build_dataset.py`, `02b_crosscheck_ard.py`, `04_covariates.py` | Download measurements; build the official RSN site list, the site-year dataset and the storm-overflow covariate | Public (see `data/README.md`) |
| 2 River network and river size | `phase_02/scripts/01_provenance.py` … `06_contrasts.py` (`common.py`, `netmatch.py`) | Build the river frame and repair flow direction; calibrate catchment area; match sites to links; coverage; raw contrasts | Public, **plus NRFA station records, which you must obtain yourself** |
| 3 Model specification | `phase_02b/scripts/p2b_08_constants.py`; `phase_02b/specifications/*` | Frozen standardisation constants, priors and settings | Step 2 outputs |
| 4 Model inputs | `phase_03a/scripts/p3a_01_build_dataset.py`; `phase_03c/scripts/p3c_01_topology.py`; `phase_03c_f/scripts/p3f_02_prepare_inputs.py` | Data tables; flow-connected pairs, tail-up DAG and blocks; Stan `data.json` and initial values | Steps 1–3 |
| 5 Posterior inference | `phase_03c_f/scripts/p3f_run_r3.py`; `phase_03c_f/addendum_T1/scripts/t1_02_run.py`, `t1_03_combine.py`, `t1_04_gates.py`, `t1_05_detail.py` | Main fit (4 chains), exact continuation to 4 × 5,000 draws, convergence gates | Step 4. **Computationally expensive.** |
| 6 Diagnostics | `phase_03c_f/scripts/p3f_04_main_diagnostics.py`, `p3f_05_prediction_settings.py`, `p3f_06_identifiability.py`, `p3f_07_upper_tail.py` | Posterior diagnostics, prediction settings, identifiability, upper tail | Step 5 |
| 7 Estimands and sensitivity | `phase_04/scripts/p4_01_validation_review.py` … `p4_05_identifiability_uppertail.py` | Selection sensitivity (ψ, κ); river-length B̄ with the tail-up conditional; posterior predictive checks; prior sensitivity | Step 5 draws |
| 8 Held-out validation | `phase_04/validation_final/scripts/vf_01_fit.py`, `vf_02_score.py`, `vf_02b_sampler_sensitivity.py`, `vf_02c_rb_check.py`, `vf_03_sensitivity.py`, `vf_09_consolidate.py`; `correction/scripts/vc_01`–`vc_05`; `repair/scripts/vr_01`–`vr_08`, `vr_mcse.py` | Fold fits (W1, W2, C1, C2) of both models; scoring; correlation-aware Monte Carlo error; the pre-specified final W1 attempt (stopped at the 5th divergence, as reported in the paper) | Step 4. **Computationally expensive.** |
| 9 Figures and tables | `phase_04/scripts/p4_07_figures.py`; `phase_04/validation_final/scripts/vf_05_figures.py`, `vf_06_tables.py`; `correction/scripts/vc_06_manuscript_assets.py`; `repair/editorial/ve_01_fig6_final_mcse.py` | Figures 1–8 and the table values | Steps 5–8 |

**What can and cannot be rerun:**
- **Step 1** runs from public sources only.
- **Steps 2 onwards** depend on the river-size calibration, which needs NRFA station records. These cannot be redistributed, so you must obtain them from the NRFA yourself. Without them the river frame, the size covariate and every later step cannot be rebuilt exactly.
- **Posterior draws and per-draw scores are not included**, because of their size. Steps 6–9 therefore need Steps 5 and 8 to be rerun.
- **Fold-fit costs:** the fold fits took 0.9–3.4 CPU-hours each. The main fit is the most expensive step.
- **Output tables:** `results/` holds the final numerical summaries, so you can compare your outputs with the paper without rerunning anything.

The pipeline has **not** been rerun independently from this repository. The code is copied unchanged from the analysis project. The configurations record the seeds, sampler settings and frozen protocol values used.

## Data sources and access conditions

See [`data/README.md`](data/README.md). In short:
- the Environment Agency datasets and OS Open Rivers are available under the Open Government Licence v3.0;
- the NRFA gauged catchment areas are licensed under the NRFA click-through licence and are **not** redistributed here.

## Licensing and citation guidance

- **Code licence:** the code, configurations and documentation in this repository are released under the [MIT License](LICENSE).
- **Third-party data:** these keep their own licences (see `data/README.md`). Data obtained under the Open Government Licence must be attributed as specified there.
- **Citing this work:** if you use the code or results, please cite the paper (see Citation above).
