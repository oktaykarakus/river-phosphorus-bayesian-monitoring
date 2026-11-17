// Track A — Phase 3C-R: corrected parameterisations of the river-network model M2-RN.
// Superset of phase_03c/stan/track_a_m2rn.stan (frozen). Two new data switches:
//   site_nc = 0, use_wb = 1 : R0 = the frozen Phase 3C program (centred site levels, marginalised water-body effects).
//                             Verified identical log density to the Phase 3C executable (tests/test_phase3cr.py U1).
//   site_nc = 1, use_wb = 1 : R1 = NON-CENTRED site levels, SAME MODEL. Within water body b (n_b sites),
//                             theta_b = m_b + A_b z_b,  z_b ~ N(0, I),  A_b = sigma_u (I - P_b) + s_b P_b,
//                             P_b = J/n_b (projector on the water-body mean), s_b = sqrt(sigma_u^2 + n_b sigma_B^2).
//                             A_b is the symmetric square root of sigma_u^2 I + sigma_B^2 J, so theta_b ~ MVN(m_b, sigma_u^2 I + sigma_B^2 J)
//                             exactly as the frozen marginal water-body model: a pure reparameterisation (U2, U5).
//                             theta_i = m_i + sigma_u z_i + (s_b - sigma_u) zbar_b.
//   site_nc = 1, use_wb = 0 : R2 = water-body effect REMOVED (substantive change): theta = m + sigma_u z.
//   site_nc = 0, use_wb = 0 : centred version of R2 (diagnostic only).
// The site-level mean m (fixed effects, network offset, WMA, catchment, HSGP field, tail-up) is unchanged and is computed
// by site_mean(); theta is a transformed parameter in every mode so that all downstream (Phase 3B/3C) code reads 'theta'.
functions {
  vector hsgp_field(matrix PhiE, matrix PhiN, matrix LAM2, matrix Z, real sigma_w, real rho) {
    matrix[rows(LAM2), cols(LAM2)] sqrt_spd = sigma_w * sqrt(2 * pi()) * rho * exp(-0.25 * square(rho) * LAM2);
    return rows_dot_product(PhiE * (sqrt_spd .* Z), PhiN);
  }
  vector tu_effect(array[] int order, array[] int ptr, array[] int par, vector h, vector sqrt_pw, vector z, real sigma, real rho) {
    int I = num_elements(order);
    vector[I] v;
    for (k in 1:I) {
      int i = order[k];
      real acc = 0;
      real s2 = 1;
      for (e in ptr[i]:(ptr[i + 1] - 1)) {
        real a = exp(-h[e] / rho) * sqrt_pw[e];
        acc += a * v[par[e]];
        s2 -= square(a);
      }
      v[i] = acc + sigma * sqrt(fmax(s2, 1e-10)) * z[i];
    }
    return v;
  }
  vector td_effect(array[] int order, array[] int parent, vector d, vector z, real sigma, real rho) {
    int n = num_elements(order);
    vector[n] v;
    for (k in 1:n) {
      int j = order[k];
      if (parent[j] == 0) {
        v[j] = sigma * z[j];
      } else {
        real a = exp(-d[j] / rho);
        v[j] = a * v[parent[j]] + sigma * sqrt(1 - square(a)) * z[j];
      }
    }
    return v;
  }
  vector delta_vec(int network_mode, vector delta_bar, vector tau_delta, vector z_delta) {
    vector[7] delta_r = rep_vector(0, 7);
    if (network_mode == 1) delta_r = rep_vector(delta_bar[1], 7);
    if (network_mode == 2) delta_r = delta_bar[1] + tau_delta[1] * z_delta;
    return delta_r;
  }
  vector site_mean(real alpha, matrix X, vector beta, vector ms, vector delta_r, array[] int region,
                   int use_hier, vector sigma_A, vector z_A, array[] int wma, vector sigma_C, vector z_C, array[] int mancat,
                   int expl_wb, vector sigma_B, vector z_B, array[] int wb,
                   int use_field, matrix PhiE, matrix PhiN, matrix LAM2, matrix z_w, vector sigma_w, vector rho,
                   int use_tu, array[] int tu_order, array[] int tu_ptr, array[] int tu_par, vector tu_h, vector tu_sqrt_pw,
                   vector z_tu, vector sigma_tu, vector rho_tu,
                   int use_td, array[] int td_order, array[] int td_parent, vector td_dist, vector z_td, vector sigma_td, vector rho_td,
                   array[] int td_site_node) {
    vector[rows(X)] m = alpha + X * beta + ms .* delta_r[region];
    if (use_hier) m += sigma_A[1] * z_A[wma] + sigma_C[1] * z_C[mancat];
    if (expl_wb) m += sigma_B[1] * z_B[wb];
    if (use_field) m += hsgp_field(PhiE, PhiN, LAM2, z_w, sigma_w[1], rho[1]);
    if (use_tu) m += tu_effect(tu_order, tu_ptr, tu_par, tu_h, tu_sqrt_pw, z_tu, sigma_tu[1], rho_tu[1]);
    if (use_td) m += td_effect(td_order, td_parent, td_dist, z_td, sigma_td[1], rho_td[1])[td_site_node];
    return m;
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
  matrix[I, K] X;
  vector<lower=0, upper=1>[I] ms;
  array[I] int<lower=1, upper=7> region;
  int<lower=0, upper=2> network_mode;
  int<lower=0, upper=1> regional_season;
  int<lower=0> P_s;
  matrix[I, P_s] Zs;
  int<lower=0, upper=4> n_hmod;
  real<lower=0> seas_mod_sd;
  int<lower=0, upper=1> use_shock;
  real<lower=0> sigma_e_sd;
  int<lower=0, upper=1> shock_centre;
  int<lower=0, upper=1> hetero_sigma;
  int<lower=0, upper=1> use_hier;
  int<lower=1> n_A; int<lower=1> n_C; int<lower=1> n_B;
  array[I] int<lower=1, upper=n_A> wma;
  array[I] int<lower=1, upper=n_C> mancat;
  array[I] int<lower=1, upper=n_B> wb;
  int<lower=0, upper=1> wb_marginal;
  int<lower=0, upper=1> use_field;
  int<lower=1> m1; int<lower=1> m2;
  matrix[I, m1] PhiE;
  matrix[I, m2] PhiN;
  vector<lower=0>[m1] sqrt_lamE;
  vector<lower=0>[m2] sqrt_lamN;
  real alpha_mu; real<lower=0> alpha_sd;
  vector<lower=0>[K] beta_sd;
  real<lower=0> gamma_sd; real<lower=0> harm_sd; real<lower=0> tau_s_sd;
  real<lower=0> delta_sd; real<lower=0> tau_delta_sd;
  real<lower=0> sigma_u_sd; real<lower=0> sigma_om_sd; real<lower=0> sigma_sd;
  real<lower=0> lambda_sd; real<lower=0> tau_sig_sd;
  real<lower=0> sigma_A_sd; real<lower=0> sigma_C_sd; real<lower=0> sigma_B_sd;
  real<lower=0> sigma_w_sd; real<lower=0> rho_a; real<lower=0> rho_b;
  int<lower=0, upper=1> use_tu;
  int<lower=0> n_tu_edges;
  array[I] int<lower=1, upper=I> tu_order;
  array[I + 1] int<lower=1> tu_ptr;
  array[n_tu_edges] int<lower=1, upper=I> tu_par;
  vector<lower=0>[n_tu_edges] tu_h;
  vector<lower=0, upper=1>[n_tu_edges] tu_sqrt_pw;
  int<lower=0, upper=1> use_td;
  int<lower=1> n_td;
  array[n_td] int<lower=1, upper=n_td> td_order;
  array[n_td] int<lower=0, upper=n_td> td_parent;
  vector<lower=0>[n_td] td_dist;
  array[I] int<lower=1, upper=n_td> td_site_node;
  real<lower=0> sigma_tu_sd; real<lower=0> rho_tu_a; real<lower=0> rho_tu_b;
  real<lower=0> sigma_td_sd; real<lower=0> rho_td_a; real<lower=0> rho_td_b;
  int<lower=0, upper=J> J_chk;
  int<lower=1> Q;
  // Phase 3C-R switches
  int<lower=0, upper=1> site_nc;            // 0 centred site levels (frozen C4); 1 non-centred (R1 / R2)
  int<lower=0, upper=1> use_wb;             // 1 water-body effect (frozen); 0 removed (R2)
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
  matrix[m1, m2] LAM2;
  for (j in 1:m1) for (k in 1:m2) LAM2[j, k] = square(sqrt_lamE[j]) + square(sqrt_lamN[k]);
  int n_delta = network_mode > 0 ? 1 : 0;
  int n_dreg = network_mode == 2 ? 7 : 0;
  int n_seas = regional_season == 1 ? 7 : 0;
  int n_het = hetero_sigma == 1 ? 1 : 0;
  int n_h = use_hier == 1 ? 1 : 0;
  int n_f = use_field == 1 ? 1 : 0;
  int has_wb = (use_hier == 1 && use_wb == 1) ? 1 : 0;
  int marg_wb = (has_wb == 1 && wb_marginal == 1) ? 1 : 0;
  int n_rm = 7 * 24;
  array[N] int obs_rm;
  for (n in 1:N) {
    int mth = min(12, 1 + to_int(floor(12 * tau[n])));
    obs_rm[n] = (region[obs_site[n]] - 1) * 24 + (sy_year[sy[n]] - 1) * 12 + mth;
  }
  vector[n_B] n_wb = rep_vector(0, n_B);
  for (i in 1:I) n_wb[wb[i]] += 1;
  vector[n_B] n_wb1 = fmax(n_wb, 1);          // guards levels without sites (never indexed)
  array[Q] int q_month;
  for (q in 1:Q) q_month[q] = min(12, 1 + to_int(floor(12 * (q - 0.5) / Q)));
}
parameters {
  real alpha;
  vector[K] beta;
  real gamma;
  vector[4] harm;
  vector[n_delta] delta_bar;
  vector[n_dreg] z_delta;
  vector<lower=0>[n_dreg > 0 ? 1 : 0] tau_delta;
  matrix[n_seas, 2] z_seas;
  vector<lower=0>[n_seas > 0 ? 1 : 0] tau_s;
  matrix[P_s, n_hmod] Cs;
  real<lower=0> sigma_u;
  vector[I] theta_par;                       // centred: the site levels theta; non-centred: z (standard normal)
  real<lower=0> sigma_om;
  vector[J] z_om;
  real<lower=0> sigma;
  vector[n_het] lambda;
  vector<lower=0>[n_het] tau_sig;
  vector[hetero_sigma ? I : 0] z_sig;
  vector<lower=0>[n_h] sigma_A;
  vector<lower=0>[n_h] sigma_C;
  vector<lower=0>[has_wb] sigma_B;
  vector[use_hier ? n_A : 0] z_A;
  vector[use_hier ? n_C : 0] z_C;
  vector[(has_wb && !wb_marginal) ? n_B : 0] z_B;
  vector<lower=0>[n_f] sigma_w;
  vector<lower=0>[n_f] rho;
  matrix[use_field ? m1 : 0, use_field ? m2 : 0] z_w;
  vector<lower=0>[use_shock] sigma_e;
  vector[use_shock ? n_rm : 0] z_e;
  vector<lower=0>[use_tu] sigma_tu;
  vector<lower=0>[use_tu] rho_tu;
  vector[use_tu ? I : 0] z_tu;
  vector<lower=0>[use_td] sigma_td;
  vector<lower=0>[use_td] rho_td;
  vector[use_td ? n_td : 0] z_td;
}
transformed parameters {
  vector[use_shock ? n_rm : 0] e_rm;
  if (use_shock) {
    e_rm = sigma_e[1] * z_e;
    if (shock_centre) for (b in 1:14) e_rm[((b - 1) * 12 + 1):(b * 12)] -= mean(e_rm[((b - 1) * 12 + 1):(b * 12)]);
  }
  vector[I] theta;                           // site levels (saved in every mode; downstream code reads 'theta')
  if (site_nc == 0) {
    theta = theta_par;
  } else {
    vector[I] m_site = site_mean(alpha, X, beta, ms, delta_vec(network_mode, delta_bar, tau_delta, z_delta), region,
                    use_hier, sigma_A, z_A, wma, sigma_C, z_C, mancat, has_wb && !wb_marginal, sigma_B, z_B, wb,
                    use_field, PhiE, PhiN, LAM2, z_w, sigma_w, rho,
                    use_tu, tu_order, tu_ptr, tu_par, tu_h, tu_sqrt_pw, z_tu, sigma_tu, rho_tu,
                    use_td, td_order, td_parent, td_dist, z_td, sigma_td, rho_td, td_site_node);
    if (marg_wb) {
      vector[n_B] zbar = rep_vector(0, n_B);
      for (i in 1:I) zbar[wb[i]] += theta_par[i];
      zbar = zbar ./ n_wb1;
      vector[n_B] s_b = sqrt(square(sigma_u) + n_wb * square(sigma_B[1]));
      theta = m_site + sigma_u * theta_par + (s_b[wb] - sigma_u) .* zbar[wb];
    } else {
      theta = m_site + sigma_u * theta_par;
    }
  }
}
model {
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
  sigma_tu ~ normal(0, sigma_tu_sd);
  rho_tu ~ inv_gamma(rho_tu_a, rho_tu_b);
  z_tu ~ std_normal();
  sigma_td ~ normal(0, sigma_td_sd);
  rho_td ~ inv_gamma(rho_td_a, rho_td_b);
  z_td ~ std_normal();
  // ---- site levels
  if (site_nc == 1) {
    theta_par ~ std_normal();
  } else if (marg_wb) {
    vector[I] e = theta - site_mean(alpha, X, beta, ms, delta_vec(network_mode, delta_bar, tau_delta, z_delta), region,
                    use_hier, sigma_A, z_A, wma, sigma_C, z_C, mancat, has_wb && !wb_marginal, sigma_B, z_B, wb,
                    use_field, PhiE, PhiN, LAM2, z_w, sigma_w, rho,
                    use_tu, tu_order, tu_ptr, tu_par, tu_h, tu_sqrt_pw, z_tu, sigma_tu, rho_tu,
                    use_td, td_order, td_parent, td_dist, z_td, sigma_td, rho_td, td_site_node);
    vector[n_B] se = rep_vector(0, n_B);
    for (i in 1:I) se[wb[i]] += e[i];
    real su2 = square(sigma_u);
    real sb2 = square(sigma_B[1]);
    vector[n_B] den = su2 + n_wb * sb2;
    target += -0.5 * (dot_self(e) - sb2 * sum(square(se) ./ den)) / su2 - 0.5 * ((I - n_B) * log(su2) + sum(log(den)));
  } else {
    theta ~ normal(site_mean(alpha, X, beta, ms, delta_vec(network_mode, delta_bar, tau_delta, z_delta), region,
                    use_hier, sigma_A, z_A, wma, sigma_C, z_C, mancat, has_wb && !wb_marginal, sigma_B, z_B, wb,
                    use_field, PhiE, PhiN, LAM2, z_w, sigma_w, rho,
                    use_tu, tu_order, tu_ptr, tu_par, tu_h, tu_sqrt_pw, z_tu, sigma_tu, rho_tu,
                    use_td, td_order, td_parent, td_dist, z_td, sigma_td, rho_td, td_site_node), sigma_u);
  }
  // ---- likelihood (unchanged)
  {
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
  vector[use_tu ? I : 0] v_tu;
  vector[use_td ? I : 0] v_td;
  if (use_tu) v_tu = tu_effect(tu_order, tu_ptr, tu_par, tu_h, tu_sqrt_pw, z_tu, sigma_tu[1], rho_tu[1]);
  if (use_td) v_td = td_effect(td_order, td_parent, td_dist, z_td, sigma_td[1], rho_td[1])[td_site_node];
  // exact conditional draws of the (marginalised) water-body effects given theta and m (identical to Phase 3B/3C)
  vector[marg_wb ? n_B : 0] vB;
  if (marg_wb) {
    vector[I] m_site = site_mean(alpha, X, beta, ms, delta_vec(network_mode, delta_bar, tau_delta, z_delta), region,
                    use_hier, sigma_A, z_A, wma, sigma_C, z_C, mancat, has_wb && !wb_marginal, sigma_B, z_B, wb,
                    use_field, PhiE, PhiN, LAM2, z_w, sigma_w, rho,
                    use_tu, tu_order, tu_ptr, tu_par, tu_h, tu_sqrt_pw, z_tu, sigma_tu, rho_tu,
                    use_td, td_order, td_parent, td_dist, z_td, sigma_td, rho_td, td_site_node);
    vector[n_B] se = rep_vector(0, n_B);
    for (i in 1:I) se[wb[i]] += theta[i] - m_site[i];
    real su2 = square(sigma_u); real sb2 = square(sigma_B[1]);
    for (b in 1:n_B) vB[b] = normal_rng(sb2 * se[b] / (su2 + n_wb[b] * sb2), sqrt(sb2 * su2 / (su2 + n_wb[b] * sb2)));
  }
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
