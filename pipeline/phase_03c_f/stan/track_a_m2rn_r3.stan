// Track A — Phase 3C-R candidate R3: M2-RN with the exchangeable site-level components MARGINALISED and the variance
// components in total-SD x allocation-share coordinates. SAME MODEL and SAME PRIORS as R0/R1 (verified, tests U6-U8).
//   theta (centred) | m ~ MVN(m, Sigma),  Sigma = sigma_u^2 I + sigma_B^2 Z_B Z_B' + sigma_tu^2 C_TU(rho_tu)
// i.e. site (u), water-body (vB) and tail-up (v_tu) effects are integrated out exactly. Sigma is block-diagonal over the
// connected components of {same river tree} U {same water body} (233 blocks, largest 257 sites); C_TU is the closed-form
// tail-up correlation exp(-h/rho) sqrt(prod of normalised AFV weights) for flow-connected pairs (= the Phase 3C DAG form, G1).
//   sigma_u = sigma_site sqrt(pi_1), sigma_B = sigma_site sqrt(pi_2), sigma_tu = sigma_site sqrt(pi_3), pi ~ simplex;
// the ORIGINAL half-normal priors on (sigma_u, sigma_B, sigma_tu) are kept by adding log|d(s)/d(sigma_site, pi)| =
// 5 log sigma_site - sum_k log s_k (+ const). No prior is changed.
// Generated quantities: v_tu and vB are drawn EXACTLY from their joint conditional given (theta, m) by Matheron's rule
// (prior draw + Sigma_cross Sigma^{-1} residual), so all Phase 3B/3C post-processing (location terms, Delta_T, scores) is unchanged.
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
  vector site_mean0(real alpha, matrix X, vector beta, vector ms, vector delta_r, array[] int region,
                    vector sigma_A, vector z_A, array[] int wma, vector sigma_C, vector z_C, array[] int mancat,
                    int use_field, matrix PhiE, matrix PhiN, matrix LAM2, matrix z_w, vector sigma_w, vector rho) {
    vector[rows(X)] m = alpha + X * beta + ms .* delta_r[region] + sigma_A[1] * z_A[wma] + sigma_C[1] * z_C[mancat];
    if (use_field) m += hsgp_field(PhiE, PhiN, LAM2, z_w, sigma_w[1], rho[1]);
    return m;
  }
  matrix block_cov(int n, vector wbs_flat, int npr, array[] int pr_i, array[] int pr_j, vector pr_h, vector pr_lw, int p0,
                   real su2, real sb2, real stu2, real rho_tu) {
    matrix[n, n] S = sb2 * to_matrix(wbs_flat, n, n);
    for (i in 1:n) S[i, i] += su2 + stu2;
    for (p in p0:(p0 + npr - 1)) {
      real c = stu2 * exp(-pr_h[p] / rho_tu + 0.5 * pr_lw[p]);
      S[pr_i[p], pr_j[p]] += c;
      S[pr_j[p], pr_i[p]] += c;
    }
    return S;
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
  // R3 block structure (sites permuted block by block)
  int<lower=1> n_blk;
  array[n_blk] int<lower=1> blk_size;
  array[n_blk] int<lower=1> blk_start;                // into blk_sites
  array[I] int<lower=1, upper=I> blk_sites;
  int<lower=0> n_wbs;                                 // length of the flattened same-water-body indicators (sum n^2 over blocks)
  array[n_blk] int<lower=1> wbs_start;
  vector<lower=0, upper=1>[n_wbs] wbs;                // column-major n x n same-water-body indicator per block
  int<lower=0> n_pr;                                  // flow-connected pairs (tail-up)
  array[n_blk] int<lower=1> pr_start;                 // pairs of block k: pr_start[k] : pr_start[k] + pr_n[k] - 1
  array[n_blk] int<lower=0> pr_n;
  array[n_pr] int<lower=1> pr_i;                      // local indices within the block
  array[n_pr] int<lower=1> pr_j;
  vector<lower=0>[n_pr] pr_h;                         // river distance (km)
  vector<upper=0>[n_pr] pr_lw;                        // log product of normalised AFV weights
  int<lower=0> n_single;
  array[n_single] int<lower=1, upper=I> single_sites; // sites forming 1-site blocks
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
  vector[4] harm;
  vector[n_delta] delta_bar;
  vector[n_dreg] z_delta;
  vector<lower=0>[n_dreg > 0 ? 1 : 0] tau_delta;
  matrix[n_seas, 2] z_seas;
  vector<lower=0>[n_seas > 0 ? 1 : 0] tau_s;
  matrix[P_s, n_hmod] Cs;
  real<lower=0> sigma_site;                  // total exchangeable site-level SD: sqrt(sigma_u^2 + sigma_B^2 + sigma_tu^2)
  simplex[3] share;                          // allocation of sigma_site^2 to (site u, water body B, tail-up TU)
  vector[I] theta;                           // centred site levels
  real<lower=0> sigma_om;
  vector[J] z_om;
  real<lower=0> sigma;
  vector[n_het] lambda;
  vector<lower=0>[n_het] tau_sig;
  vector[hetero_sigma ? I : 0] z_sig;
  vector<lower=0>[n_h] sigma_A;
  vector<lower=0>[n_h] sigma_C;
  vector[use_hier ? n_A : 0] z_A;
  vector[use_hier ? n_C : 0] z_C;
  vector<lower=0>[n_f] sigma_w;
  vector<lower=0>[n_f] rho;
  matrix[use_field ? m1 : 0, use_field ? m2 : 0] z_w;
  vector<lower=0>[use_shock] sigma_e;
  vector[use_shock ? n_rm : 0] z_e;
  vector<lower=0>[1] rho_tu;                 // km
}
transformed parameters {
  vector[use_shock ? n_rm : 0] e_rm;
  if (use_shock) {
    e_rm = sigma_e[1] * z_e;
    if (shock_centre) for (b in 1:14) e_rm[((b - 1) * 12 + 1):(b * 12)] -= mean(e_rm[((b - 1) * 12 + 1):(b * 12)]);
  }
  real<lower=0> sigma_u = sigma_site * sqrt(share[1]);
  vector<lower=0>[1] sigma_B = rep_vector(sigma_site * sqrt(share[2]), 1);
  vector<lower=0>[1] sigma_tu = rep_vector(sigma_site * sqrt(share[3]), 1);
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
  // original half-normal priors on the three SDs + Jacobian of (sigma_site, share) -> (sigma_u, sigma_B, sigma_tu)
  target += normal_lpdf(sigma_u | 0, sigma_u_sd) + normal_lpdf(sigma_B[1] | 0, sigma_B_sd) + normal_lpdf(sigma_tu[1] | 0, sigma_tu_sd);
  target += 5 * log(sigma_site) - log(sigma_u) - log(sigma_B[1]) - log(sigma_tu[1]);
  sigma_om ~ normal(0, sigma_om_sd);
  z_om ~ std_normal();
  sigma ~ normal(0, sigma_sd);
  lambda ~ normal(0, lambda_sd);
  tau_sig ~ normal(0, tau_sig_sd);
  z_sig ~ std_normal();
  sigma_A ~ normal(0, sigma_A_sd);
  sigma_C ~ normal(0, sigma_C_sd);
  z_A ~ std_normal(); z_C ~ std_normal();
  sigma_w ~ normal(0, sigma_w_sd);
  rho ~ inv_gamma(rho_a, rho_b);
  to_vector(z_w) ~ std_normal();
  sigma_e ~ normal(0, sigma_e_sd);
  z_e ~ std_normal();
  rho_tu ~ inv_gamma(rho_tu_a, rho_tu_b);
  // ---- site levels: theta ~ MVN(m, Sigma), block by block (exact marginalisation of u, vB, v_tu)
  {
    vector[I] m = site_mean0(alpha, X, beta, ms, delta_vec(network_mode, delta_bar, tau_delta, z_delta), region,
                             sigma_A, z_A, wma, sigma_C, z_C, mancat, use_field, PhiE, PhiN, LAM2, z_w, sigma_w, rho);
    vector[I] e = theta - m;
    target += normal_lpdf(e[single_sites] | 0, sigma_site);
    for (k in 1:n_blk) {
      int n = blk_size[k];
      if (n > 1) {
        array[n] int idx = blk_sites[blk_start[k]:(blk_start[k] + n - 1)];
        matrix[n, n] L = cholesky_decompose(block_cov(n, wbs[wbs_start[k]:(wbs_start[k] + n * n - 1)], pr_n[k], pr_i, pr_j, pr_h, pr_lw, pr_start[k],
                                                      square(sigma_u), square(sigma_B[1]), square(sigma_tu[1]), rho_tu[1]));
        target += multi_normal_cholesky_lpdf(e[idx] | rep_vector(0, n), L);
      }
    }
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
  // exact joint conditional draws of (v_tu, vB) given (theta, m) by Matheron's rule:
  //   (v*, vB*, u*) ~ prior;  r = e - (v* + vB*[wb] + u*);  w = Sigma^{-1} r;  v_tu = v* + sigma_tu^2 C_TU w;  vB_b = vB*_b + sigma_B^2 sum_{i in b} w_i
  vector[I] v_tu;
  vector[n_B] vB;
  {
    vector[I] m = site_mean0(alpha, X, beta, ms, delta_vec(network_mode, delta_bar, tau_delta, z_delta), region,
                             sigma_A, z_A, wma, sigma_C, z_C, mancat, use_field, PhiE, PhiN, LAM2, z_w, sigma_w, rho);
    vector[I] zt;
    for (i in 1:I) zt[i] = normal_rng(0, 1);
    vector[I] vstar = tu_effect(tu_order, tu_ptr, tu_par, tu_h, tu_sqrt_pw, zt, sigma_tu[1], rho_tu[1]);
    vector[n_B] bstar;
    for (b in 1:n_B) bstar[b] = normal_rng(0, sigma_B[1]);
    vector[I] r;
    for (i in 1:I) r[i] = theta[i] - m[i] - vstar[i] - bstar[wb[i]] - normal_rng(0, sigma_u);
    vector[I] w = r / square(sigma_site);                 // 1-site blocks: Sigma = sigma_site^2
    for (k in 1:n_blk) {
      int n = blk_size[k];
      if (n > 1) {
        array[n] int idx = blk_sites[blk_start[k]:(blk_start[k] + n - 1)];
        matrix[n, n] S = block_cov(n, wbs[wbs_start[k]:(wbs_start[k] + n * n - 1)], pr_n[k], pr_i, pr_j, pr_h, pr_lw, pr_start[k],
                                   square(sigma_u), square(sigma_B[1]), square(sigma_tu[1]), rho_tu[1]);
        w[idx] = mdivide_left_spd(S, r[idx]);
      }
    }
    // v_tu = vstar + sigma_tu^2 C_TU w  (C_TU: unit diagonal + flow-connected pairs)
    v_tu = vstar + square(sigma_tu[1]) * w;
    for (k in 1:n_blk) {
      int n = blk_size[k];
      if (n > 1) {
        array[n] int idx = blk_sites[blk_start[k]:(blk_start[k] + n - 1)];
        for (p in pr_start[k]:(pr_start[k] + pr_n[k] - 1)) {
          real c = square(sigma_tu[1]) * exp(-pr_h[p] / rho_tu[1] + 0.5 * pr_lw[p]);
          v_tu[idx[pr_i[p]]] += c * w[idx[pr_j[p]]];
          v_tu[idx[pr_j[p]]] += c * w[idx[pr_i[p]]];
        }
      }
    }
    vector[n_B] sw = rep_vector(0, n_B);
    for (i in 1:I) sw[wb[i]] += w[i];
    vB = bstar + square(sigma_B[1]) * sw;
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
