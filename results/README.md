# Numerical result summaries

These are outputs of the analysis pipeline, copied unchanged. They are posterior and validation summaries only: they contain no raw measurements and no NRFA data. Where a value here and the paper disagree in rounding, the paper is authoritative.

| File | Produced by | Content | Paper |
|---|---|---|---|
| `T1_combined_delta_T.csv` | `phase_03c_f/addendum_T1/scripts/t1_04_gates.py` | Standardised contrast Δ_T at 0.05, 0.10 and 0.20 mg P L⁻¹ (median, 95% interval, P(Δ_T > 0)) for the primary model, with comparison columns | Table 2, Fig. 4 |
| `A2_selection_grid.csv` | `phase_04/scripts/p4_02_selection_sensitivity.py` | Siting contrast Δ^S_T over the ψ × κ grid | §5.6, Fig. 7a |
| `A2_PR_by_psi.csv` | same | Reference-level exceedance over the target site-years as a function of ψ | §5.6, Fig. 7b |
| `A3_river_length.csv` | `phase_04/scripts/p4_03b_river_length.py` | River-length-weighted exceedance B̄ (and per year), primary and sensitivity variants | Table 4, §5.5 |
| `A3_river_length_psi.csv` | same | B̄ as a function of ψ | Table 4, Fig. 7b |
| `A3_river_length_strata.csv` | same | B̄ by stratum (river size, monitoring context, region) | §5.5, Fig. 8b |
| `A4_ppc_cells.csv`, `A4_lowtail.csv` | `phase_04/scripts/p4_04_ppc_summary.py` | Posterior predictive checks of sample-mean exceedance; lower-tail check that does not depend on reporting limits | §5.7 |
| `VF_validation_fits.csv` | `phase_04/validation_final/scripts/vf_02_score.py` / `vf_06_tables.py` | Sampling diagnostics of the held-out fold fits | Table B.1 |
| `VR_fold_results.csv`, `VR_pooled_results.csv` | `phase_04/validation_final/repair/scripts/vr_01_corrected_validation.py` | Fold and pooled score differences, clustered standard errors, and Monte Carlo errors | Table 3, Table B.2, Fig. 6 |

## How to read the Monte Carlo error columns
The two files above hold three kinds of Monte Carlo error column:
- `*_frozen`: the pre-specified sum of site-level variances.
- `*_corrected`: the correlation-aware estimator reported in the paper.
- `ratio_corrected_split_only`: a superseded preliminary implementation. It is kept for transparency and is not used in the paper.

Setting A did not meet the pre-specified validation criteria. Setting B's conclusion rests on the correlation-aware analysis that was introduced after the fold fits. See the paper's §5.4 and Appendix B.
