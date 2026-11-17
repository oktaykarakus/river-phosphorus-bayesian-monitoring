// Track A — Phase 3B measurement-level censored lognormal model, M0-H ... M1-S2.
// Phase 2B specification (MS-§2-§6) + Addendum A1 (2026-11-12): site-specific measurement SD (C1), harmonic prior 0.3 (C2),
// centred site levels / non-centred site-year effects (C4).
//
// eta_ij = theta_i + gamma*1{t=2024} + omega_it + H(tau_ij) . c_i
//   theta_i ~ N(m_i, sigma_u)                                            (centred site level, C4)
//   m_i     = alpha + X_i beta + delta_{r(i)} MS_i [+ vA_a(i) + vC_c(i) + vB_b(i)] [+ w(s_i)]
//   c_i     = harm + [regional 1st-harmonic deviation] + Zs_i * Cs      (site harmonic coefficients a1,b1,a2,b2)
//   [+ e_{r(i),t,month(tau)} = sigma_e z_e: regional month-by-year common effects (use_shock)]
//   omega   = sigma_om * z_om                                            (non-centred)
//   w(s)    = HSGP approximation of GP(0, sigma_w^2 k_SE(rho)) in 2-D, separable basis (Riutort-Mayol et al. 2023)
// log Y_ij ~ N(eta_ij, sigma_i);  left-censored: Phi((log L_ij - eta_ij)/sigma_i)
// log sigma_i = log sigma + lambda z_logarea_i + tau_sig zeta_i            (C1; hetero_sigma = 1)
functions {
  // HSGP field at the sites: w = rows_dot_product(PhiE * (sqrt(S) .* Z), PhiN), with the 2-D squared-exponential
  // spectral density S(om) = sigma_w^2 * 2 pi rho^2 * exp(-rho^2 |om|^2 / 2)  =>  sqrt(S) = sigma_w sqrt(2 pi) rho exp(-rho^2 |om|^2 / 4)
  vector hsgp_field(matrix PhiE, matrix PhiN, matrix LAM2, matrix Z, real sigma_w, real rho) {
    matrix[rows(LAM2), cols(LAM2)] sqrt_spd = sigma_w * sqrt(2 * pi()) * rho * exp(-0.25 * square(rho) * LAM2);
    return rows_dot_product(PhiE * (sqrt_spd .* Z), PhiN);
  }
}
data {
  int<lower=1> N;
  int<lower=0> N_unc;
  int<lower=0> N_cens;
  array[N_unc] int<lower=1, upper=N> unc_idx;
  array[N_cens] int<lower=1, upper=N> cens_idx;
  vector[N_unc] log_y;
  vector[N_cens] log_L;
  int<lower=1> I;
  int<lower=1> J;
  array[N] int<lower=1, upper=J> sy;
  array[J] int<lower=1, upper=I> sy_site;
  array[J] int<lower=1, upper=2> sy_year;
  vector<lower=0, upper=1>[N] tau;
  int<lower=0> K;
  matrix[I, K] X;                           // z_logarea, z_logarea^2, z_edm, geology (Sil, Org, Unk)
  vector<lower=0, upper=1>[I] ms;
  array[I] int<lower=1, upper=7> region;
  int<lower=0, upper=2> network_mode;       // 0 none, 1 global delta, 2 delta_bar + regional deviations (M2)
  // seasonality
  int<lower=0, upper=1> regional_season;    // regional first-harmonic deviations (MS-§2)
  int<lower=0> P_s;                         // number of seasonal modifiers (site-level covariates)
  matrix[I, P_s] Zs;
  int<lower=0, upper=4> n_hmod;             // harmonic columns modulated by Zs (2 = first harmonic, 4 = both)
  real<lower=0> seas_mod_sd;
  int<lower=0, upper=1> use_shock;          // regional month-by-year common effects e_{r,t,m} (Stage 3 candidate S5)
  real<lower=0> sigma_e_sd;
  int<lower=0, upper=1> shock_centre;       // 1: shocks centred within each region x year (removes the ridge with gamma)
  // measurement SD
  int<lower=0, upper=1> hetero_sigma;
  // hierarchy (S1/S2)
  int<lower=0, upper=1> use_hier;
  int<lower=1> n_A; int<lower=1> n_C; int<lower=1> n_B;
  array[I] int<lower=1, upper=n_A> wma;
  array[I] int<lower=1, upper=n_C> mancat;
  array[I] int<lower=1, upper=n_B> wb;
  int<lower=0, upper=1> wb_marginal;        // 1: water-body effects integrated out analytically (exact; P3B_04 §2)
  // HSGP field (S2)
  int<lower=0, upper=1> use_field;
  int<lower=1> m1; int<lower=1> m2;
  matrix[I, m1] PhiE;
  matrix[I, m2] PhiN;
  vector<lower=0>[m1] sqrt_lamE;
  vector<lower=0>[m2] sqrt_lamN;
  // priors
  real alpha_mu; real<lower=0> alpha_sd;
  vector<lower=0>[K] beta_sd;
  real<lower=0> gamma_sd; real<lower=0> harm_sd; real<lower=0> tau_s_sd;
  real<lower=0> delta_sd; real<lower=0> tau_delta_sd;
  real<lower=0> sigma_u_sd; real<lower=0> sigma_om_sd; real<lower=0> sigma_sd;
  real<lower=0> lambda_sd; real<lower=0> tau_sig_sd;
  real<lower=0> sigma_A_sd; real<lower=0> sigma_C_sd; real<lower=0> sigma_B_sd;
  real<lower=0> sigma_w_sd; real<lower=0> rho_a; real<lower=0> rho_b;
  // generated-quantity check
  int<lower=0, upper=J> J_chk;
  int<lower=1> Q;
}
transformed data {
  matrix[N, 4] H;
  for (n in 1:N) {
    H[n, 1] = cos(2 * pi() * tau[n]);  H[n, 2] = sin(2 * pi() * tau[n]);
    H[n, 3] = cos(4 * pi() * tau[n]);  H[n, 4] = sin(4 * pi() * tau[n]);
  }
  matrix[Q, 4] Hq;
  for (q in 1:Q) {
    real tq = (q - 0.5) / Q;
    Hq[q, 1] = cos(2 * pi() * tq); Hq[q, 2] = sin(2 * pi() * tq);
    Hq[q, 3] = cos(4 * pi() * tq); Hq[q, 4] = sin(4 * pi() * tq);
  }
  array[N] int obs_site;
  for (n in 1:N) obs_site[n] = sy_site[sy[n]];
  vector[J] yr2024;
  for (j in 1:J) yr2024[j] = sy_year[j] == 2;
  matrix[m1, m2] LAM2;                       // |omega|^2 for each 2-D basis function
  for (j in 1:m1) for (k in 1:m2) LAM2[j, k] = square(sqrt_lamE[j]) + square(sqrt_lamN[k]);
  int n_delta = network_mode > 0 ? 1 : 0;
  int n_dreg = network_mode == 2 ? 7 : 0;
  int n_seas = regional_season == 1 ? 7 : 0;
  int n_het = hetero_sigma == 1 ? 1 : 0;
  int n_h = use_hier == 1 ? 1 : 0;
  int n_f = use_field == 1 ? 1 : 0;
  // region x year x month block (month block = floor(12 tau) + 1, consistent between data and quadrature nodes)
  int n_rm = 7 * 24;
  array[N] int obs_rm;
  for (n in 1:N) {
    int mth = min(12, 1 + to_int(floor(12 * tau[n])));
    obs_rm[n] = (region[obs_site[n]] - 1) * 24 + (sy_year[sy[n]] - 1) * 12 + mth;
  }
  vector[n_B] n_wb = rep_vector(0, n_B);
  for (i in 1:I) n_wb[wb[i]] += 1;
  array[Q] int q_month;
  for (q in 1:Q) q_month[q] = min(12, 1 + to_int(floor(12 * (q - 0.5) / Q)));
}
parameters {
  real alpha;
  vector[K] beta;
  real gamma;
  vector[4] harm;                            // a1, b1, a2, b2 (national)
  vector[n_delta] delta_bar;
  vector[n_dreg] z_delta;
  vector<lower=0>[n_dreg > 0 ? 1 : 0] tau_delta;
  matrix[n_seas, 2] z_seas;
  vector<lower=0>[n_seas > 0 ? 1 : 0] tau_s;
  matrix[P_s, n_hmod] Cs;                    // seasonal modifiers (fixed effects)
  real<lower=0> sigma_u;
  vector[I] theta;                           // centred site levels (C4)
  real<lower=0> sigma_om;
  vector[J] z_om;                            // non-centred site-year effects (C4)
  real<lower=0> sigma;
  vector[n_het] lambda;
  vector<lower=0>[n_het] tau_sig;
  vector[hetero_sigma ? I : 0] z_sig;
  vector<lower=0>[n_h] sigma_A;
  vector<lower=0>[n_h] sigma_C;
  vector<lower=0>[n_h] sigma_B;
  vector[use_hier ? n_A : 0] z_A;
  vector[use_hier ? n_C : 0] z_C;
  vector[(use_hier && !wb_marginal) ? n_B : 0] z_B;
  vector<lower=0>[n_f] sigma_w;
  vector<lower=0>[n_f] rho;                  // km
  matrix[use_field ? m1 : 0, use_field ? m2 : 0] z_w;
  vector<lower=0>[use_shock] sigma_e;
  vector[use_shock ? n_rm : 0] z_e;
}
transformed parameters {
  vector[use_shock ? n_rm : 0] e_rm;        // regional month-by-year effects
  if (use_shock) {
    e_rm = sigma_e[1] * z_e;
    if (shock_centre) for (b in 1:14) e_rm[((b - 1) * 12 + 1):(b * 12)] -= mean(e_rm[((b - 1) * 12 + 1):(b * 12)]);
  }
}
model {
  // ---- priors
  alpha ~ normal(alpha_mu, alpha_sd);
  beta ~ normal(0, beta_sd);
  gamma ~ normal(0, gamma_sd);
  harm ~ normal(0, harm_sd);
  delta_bar ~ normal(0, delta_sd);
  z_delta ~ std_normal();
  tau_delta ~ normal(0, tau_delta_sd);
  to_vector(z_seas) ~ std_normal();
  tau_s ~ normal(0, tau_s_sd);
  to_vector(Cs) ~ normal(0, seas_mod_sd);
  sigma_u ~ normal(0, sigma_u_sd);
  sigma_om ~ normal(0, sigma_om_sd);
  z_om ~ std_normal();
  sigma ~ normal(0, sigma_sd);
  lambda ~ normal(0, lambda_sd);
  tau_sig ~ normal(0, tau_sig_sd);
  z_sig ~ std_normal();
  sigma_A ~ normal(0, sigma_A_sd);
  sigma_C ~ normal(0, sigma_C_sd);
  sigma_B ~ normal(0, sigma_B_sd);
  z_A ~ std_normal(); z_C ~ std_normal(); z_B ~ std_normal();
  sigma_w ~ normal(0, sigma_w_sd);
  rho ~ inv_gamma(rho_a, rho_b);
  to_vector(z_w) ~ std_normal();
  sigma_e ~ normal(0, sigma_e_sd);
  z_e ~ std_normal();
  // ---- site-level mean and centred site levels
  {
    vector[7] delta_r = rep_vector(0, 7);
    if (network_mode == 1) delta_r = rep_vector(delta_bar[1], 7);
    if (network_mode == 2) delta_r = delta_bar[1] + tau_delta[1] * z_delta;
    vector[I] m = alpha + X * beta + ms .* delta_r[region];
    if (use_hier) m += sigma_A[1] * z_A[wma] + sigma_C[1] * z_C[mancat];
    if (use_hier && !wb_marginal) m += sigma_B[1] * z_B[wb];
    if (use_field) m += hsgp_field(PhiE, PhiN, LAM2, z_w, sigma_w[1], rho[1]);
    if (use_hier && wb_marginal) {
      // theta_b ~ MVN(m_b, sigma_u^2 I + sigma_B^2 J) within each water body b (v^B_b integrated out exactly):
      // quadratic form (sum e^2 - sB2 (sum_b e)^2 / (su2 + n_b sB2)) / su2;  log det = (n_b - 1) log su2 + log(su2 + n_b sB2)
      vector[I] e = theta - m;
      vector[n_B] se = rep_vector(0, n_B);
      for (i in 1:I) se[wb[i]] += e[i];
      real su2 = square(sigma_u);
      real sb2 = square(sigma_B[1]);
      vector[n_B] den = su2 + n_wb * sb2;
      target += -0.5 * (dot_self(e) - sb2 * sum(square(se) ./ den)) / su2 - 0.5 * ((I - n_B) * log(su2) + sum(log(den)));
    } else {
      theta ~ normal(m, sigma_u);
    }
  }
  // ---- likelihood
  {
    // site harmonic coefficients c_i (I x 4)
    matrix[I, 4] C = rep_matrix(harm', I);
    if (regional_season == 1) {
      matrix[7, 2] dev = tau_s[1] * z_seas;
      C[:, 1] += dev[region, 1]; C[:, 2] += dev[region, 2];
    }
    if (P_s > 0) C[:, 1:n_hmod] += Zs * Cs;
    vector[J] sy_lp = theta[sy_site] + gamma * yr2024 + sigma_om * z_om;
    vector[N] eta = sy_lp[sy] + rows_dot_product(H, C[obs_site]);
    if (use_shock) eta += e_rm[obs_rm];
    vector[I] sig_site = hetero_sigma ? sigma * exp(lambda[1] * X[:, 1] + tau_sig[1] * z_sig) : rep_vector(sigma, I);
    vector[N] sd_obs = sig_site[obs_site];
    target += normal_lpdf(log_y | eta[unc_idx], sd_obs[unc_idx]);
    target += normal_lcdf(log_L | eta[cens_idx], sd_obs[cens_idx]);
  }
}
generated quantities {
  vector[(use_hier && wb_marginal) ? n_B : 0] vB;      // posterior draws of water-body effects (conditional on all other parameters)
  if (use_hier && wb_marginal) {
    vector[7] delta_r = rep_vector(0, 7);
    if (network_mode == 1) delta_r = rep_vector(delta_bar[1], 7);
    if (network_mode == 2) delta_r = delta_bar[1] + tau_delta[1] * z_delta;
    vector[I] m = alpha + X * beta + ms .* delta_r[region] + sigma_A[1] * z_A[wma] + sigma_C[1] * z_C[mancat];
    if (use_field) m += hsgp_field(PhiE, PhiN, LAM2, z_w, sigma_w[1], rho[1]);
    vector[n_B] se = rep_vector(0, n_B);
    for (i in 1:I) se[wb[i]] += theta[i] - m[i];
    real su2 = square(sigma_u); real sb2 = square(sigma_B[1]);
    for (b in 1:n_B) vB[b] = normal_rng(sb2 * se[b] / (su2 + n_wb[b] * sb2), sqrt(sb2 * su2 / (su2 + n_wb[b] * sb2)));
  }
  // HSGP field at the sites (Stan reference for the Python reconstruction) and calendar-year arithmetic means
  // of the first J_chk site-years (cross-check of the Python annual-mean code).
  vector[use_field ? I : 0] w_site;
  if (use_field) w_site = hsgp_field(PhiE, PhiN, LAM2, z_w, sigma_w[1], rho[1]);
  vector[J_chk] mu_chk;
  {
    matrix[I, 4] C = rep_matrix(harm', I);
    if (regional_season == 1) {
      matrix[7, 2] dev = tau_s[1] * z_seas;
      C[:, 1] += dev[region, 1]; C[:, 2] += dev[region, 2];
    }
    if (P_s > 0) C[:, 1:n_hmod] += Zs * Cs;
    vector[I] sig_site = hetero_sigma ? sigma * exp(lambda[1] * X[:, 1] + tau_sig[1] * z_sig) : rep_vector(sigma, I);
    for (j in 1:J_chk) {
      int i = sy_site[j];
      real base = theta[i] + gamma * yr2024[j] + sigma_om * z_om[j];
      vector[Q] sh = rep_vector(0, Q);
      if (use_shock) for (q in 1:Q) sh[q] = e_rm[(region[i] - 1) * 24 + (sy_year[j] - 1) * 12 + q_month[q]];
      mu_chk[j] = mean(exp(base + Hq * C[i]' + sh + 0.5 * square(sig_site[i])));
    }
  }
}
