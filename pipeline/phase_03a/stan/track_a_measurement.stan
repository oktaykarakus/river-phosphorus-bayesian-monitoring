// Track A — measurement-level censored lognormal model (Phase 2B specification, MS-§2-§6).
// Phase 3A fits the non-spatial M0 family only. Hierarchical catchment/water-body effects and the HSGP field
// (M1, S2) are NOT part of this file; they are added in Phase 3B after approval.
//
// eta_ij = alpha + X_i beta + gamma*1{t=2024} + sum_h [a_h cos 2 pi h tau + b_h sin 2 pi h tau]
//          (+ regional first-harmonic deviations) + u_i + omega_it + delta_{r(i)} * MS_i
// log Y_ij ~ Normal(eta_ij, sigma);   left-censored: Phi((log L_ij - eta_ij)/sigma)
// mu_it = (1/Q) sum_q exp(eta_it(tau_q) + sigma^2/2)   (calendar-year arithmetic mean; GQ check subset only)
data {
  int<lower=1> N;                         // measurements
  int<lower=0> N_unc;
  int<lower=0> N_cens;
  array[N_unc] int<lower=1, upper=N> unc_idx;
  array[N_cens] int<lower=1, upper=N> cens_idx;
  vector[N_unc] log_y;                     // log concentration (mg P/L), uncensored
  vector[N_cens] log_L;                    // log observation-specific limit, censored
  int<lower=0, upper=1> use_interval;      // R1: rounding-interval likelihood for uncensored values
  vector[N_unc] log_lo;                    // log(y - h)  (only used if use_interval)
  vector[N_unc] log_hi;                    // log(y + h)
  int<lower=1> I;                          // sites
  int<lower=1> J;                          // site-years
  array[N] int<lower=1, upper=J> sy;       // site-year of each measurement
  array[J] int<lower=1, upper=I> sy_site;
  array[J] int<lower=1, upper=2> sy_year;  // 1 = 2023, 2 = 2024
  vector<lower=0, upper=1>[N] tau;         // time of year
  int<lower=0> K;
  matrix[I, K] X;                          // z_logarea, z_logarea^2, z_edm, geology dummies (Sil, Org, Unk)
  vector<lower=0, upper=1>[I] ms;          // 1 = MS (routine), 0 = RSN
  array[I] int<lower=1, upper=7> region;
  int<lower=0, upper=2> network_mode;      // 0: none; 1: global delta; 2: delta_bar + regional deviations
  int<lower=0, upper=1> regional_season;   // 1: regional first-harmonic deviations
  int<lower=0, upper=1> prior_only;
  int<lower=0, upper=1> centred_site;
  int<lower=0, upper=1> centred_siteyear; // 1: omega_it ~ N(0, sigma_om) sampled directly
  int<lower=0, upper=1> hetero_sigma;     // 1 (M0-H, proposed correction): log sigma_i = log sigma + lambda z_logarea_i + tau_sig zeta_i     // 1: theta_i ~ N(alpha + X_i beta + delta MS_i, sigma_u) (hierarchical centring)
  // priors (from phase_02b/specifications/priors_and_settings.yaml)
  real alpha_mu; real<lower=0> alpha_sd;
  vector<lower=0>[K] beta_sd;
  real<lower=0> gamma_sd; real<lower=0> harm_sd; real<lower=0> tau_s_sd;
  real<lower=0> delta_sd; real<lower=0> tau_delta_sd;
  real<lower=0> sigma_u_sd; real<lower=0> sigma_om_sd; real<lower=0> sigma_sd;
  // generated-quantity check: annual means of the first J_chk site-years
  int<lower=0, upper=J> J_chk;
  int<lower=1> Q;
  real<lower=0> lambda_sd; real<lower=0> tau_sig_sd;
}
transformed data {
  matrix[N, 4] H;
  array[N] int obs_region;
  for (n in 1:N) {
    H[n, 1] = cos(2 * pi() * tau[n]);  H[n, 2] = sin(2 * pi() * tau[n]);
    H[n, 3] = cos(4 * pi() * tau[n]);  H[n, 4] = sin(4 * pi() * tau[n]);
    obs_region[n] = region[sy_site[sy[n]]];
  }
  matrix[Q, 4] Hq;
  for (q in 1:Q) {
    real tq = (q - 0.5) / Q;
    Hq[q, 1] = cos(2 * pi() * tq); Hq[q, 2] = sin(2 * pi() * tq);
    Hq[q, 3] = cos(4 * pi() * tq); Hq[q, 4] = sin(4 * pi() * tq);
  }
  int n_delta = network_mode > 0 ? 1 : 0;
  int n_dreg = network_mode == 2 ? 7 : 0;
  int n_seas = regional_season == 1 ? 7 : 0;
  int n_het = hetero_sigma == 1 ? 1 : 0;
  array[N] int obs_site;
  for (n in 1:N) obs_site[n] = sy_site[sy[n]];
  vector[J] yr2024;
  for (j in 1:J) yr2024[j] = sy_year[j] == 2;
}
parameters {
  real alpha;
  vector[K] beta;
  real gamma;
  vector[4] harm;                          // a1, b1, a2, b2
  vector[n_delta] delta_bar;
  vector[n_dreg] z_delta;
  vector<lower=0>[n_dreg > 0 ? 1 : 0] tau_delta;
  matrix[n_seas, 2] z_seas;
  vector<lower=0>[n_seas > 0 ? 1 : 0] tau_s;
  real<lower=0> sigma_u;
  vector[centred_site ? 0 : I] z_u;        // non-centred site effects
  vector[centred_site ? I : 0] theta;      // centred site levels
  real<lower=0> sigma_om;
  vector[centred_siteyear ? 0 : J] z_om;
  vector[centred_siteyear ? J : 0] om_c;
  real<lower=0> sigma;                     // (baseline) measurement-scale SD
  vector[n_het] lambda;
  vector<lower=0>[n_het] tau_sig;
  vector[hetero_sigma ? I : 0] z_sig;
}
transformed parameters {
  vector[7] delta_r = rep_vector(0, 7);
  if (network_mode == 1) delta_r = rep_vector(delta_bar[1], 7);
  if (network_mode == 2) delta_r = delta_bar[1] + tau_delta[1] * z_delta;
  vector[I] site_mean = alpha + X * beta + ms .* delta_r[region];   // site level without site effect
  vector[J] omega = centred_siteyear ? om_c : sigma_om * z_om;      // site-year effects
  vector[I] sig_site = rep_vector(sigma, I);                         // site-specific measurement SD
  if (hetero_sigma == 1) sig_site = sigma * exp(lambda[1] * X[:, 1] + tau_sig[1] * z_sig);
}
model {
  // priors
  alpha ~ normal(alpha_mu, alpha_sd);
  beta ~ normal(0, beta_sd);
  gamma ~ normal(0, gamma_sd);
  harm ~ normal(0, harm_sd);
  delta_bar ~ normal(0, delta_sd);
  z_delta ~ std_normal();
  tau_delta ~ normal(0, tau_delta_sd);
  to_vector(z_seas) ~ std_normal();
  tau_s ~ normal(0, tau_s_sd);
  sigma_u ~ normal(0, sigma_u_sd);
  sigma_om ~ normal(0, sigma_om_sd);
  sigma ~ normal(0, sigma_sd);
  lambda ~ normal(0, lambda_sd);
  tau_sig ~ normal(0, tau_sig_sd);
  z_sig ~ std_normal();
  z_u ~ std_normal();
  if (centred_site) theta ~ normal(site_mean, sigma_u);
  z_om ~ std_normal();
  if (centred_siteyear) om_c ~ normal(0, sigma_om);
  if (!prior_only) {
    vector[I] site_lp = centred_site ? theta : site_mean + sigma_u * z_u;
    vector[J] sy_lp = site_lp[sy_site] + gamma * yr2024 + omega;
    vector[N] eta = sy_lp[sy] + H * harm;
    if (regional_season == 1) {
      matrix[7, 2] dev = tau_s[1] * z_seas;
      eta += H[:, 1] .* dev[obs_region, 1] + H[:, 2] .* dev[obs_region, 2];
    }
    vector[N] sd_obs = sig_site[obs_site];
    if (use_interval == 0) {
      target += normal_lpdf(log_y | eta[unc_idx], sd_obs[unc_idx]);
    } else {
      // vectorised interval (rounding) likelihood log[Phi(z_hi) - Phi(z_lo)], evaluated in the numerically stable
      // tail: lower-tail CDFs when the interval lies below the mean, upper-tail (survival) probabilities when above.
      // (A naive Phi(z_hi) - Phi(z_lo) cancels catastrophically in the upper tail; a scalar log_diff_exp loop was ~10x slower.)
      {
        vector[N_unc] zl = (log_lo - eta[unc_idx]) ./ sd_obs[unc_idx];
        vector[N_unc] zh = (log_hi - eta[unc_idx]) ./ sd_obs[unc_idx];
        vector[N_unc] up;                                  // 1 if interval entirely above the mean
        for (k in 1:N_unc) up[k] = zl[k] > 0;
        vector[N_unc] pr = up .* (Phi(-zl) - Phi(-zh)) + (1 - up) .* (Phi(zh) - Phi(zl));
        target += sum(log(pr));
      }
    }
    target += normal_lcdf(log_L | eta[cens_idx], sd_obs[cens_idx]);
  }
}
generated quantities {
  // site effects on the common scale u_i = site level - site_mean (identical meaning in both parameterisations)
  vector[I] u_site = (centred_site ? theta : site_mean + sigma_u * z_u) - site_mean;
  // Annual arithmetic mean for the first J_chk site-years (cross-check of the Python implementation)
  vector[J_chk] mu_chk;
  {
    vector[I] site_lp = centred_site ? theta : site_mean + sigma_u * z_u;
    for (j in 1:J_chk) {
      real base = site_lp[sy_site[j]] + gamma * yr2024[j] + omega[j];
      vector[Q] seas = Hq * harm;
      if (regional_season == 1) {
        int r = region[sy_site[j]];
        seas += Hq[:, 1] * (tau_s[1] * z_seas[r, 1]) + Hq[:, 2] * (tau_s[1] * z_seas[r, 2]);
      }
      mu_chk[j] = mean(exp(base + seas + 0.5 * square(sig_site[sy_site[j]])));
    }
  }
}
