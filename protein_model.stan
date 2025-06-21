
functions {
  real log_protein_pdf(real p, real alpha, real k, real tau) {
    return log(alpha)
         + log(k)
         - k * log(tau)
         + (k - 1) * log(p)
         - (alpha + 1) * log1p(pow(p / tau, k));
  }
}
data {
  int<lower=1> N;
  vector<lower=1e-8>[N] p;
}
parameters {
  real<lower=1e-6, upper=1e6> alpha;
  real<lower=1e-6, upper=1e6> k;
  real<lower=1e-6, upper=1e6> tau;
}
model {
  for (n in 1:N)
    target += log_protein_pdf(p[n], alpha, k, tau);
}
