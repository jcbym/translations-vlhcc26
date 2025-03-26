data {
  // The number of observations
  int<lower=0> N;

  // Observed correctness
  array[N] int<lower=0, upper=1> correct;
}

parameters {
  // Probability of correctness
  real<lower=0, upper=1> theta;
}

model {
  theta ~ beta(1, 1);

  for (n in 1:N) {
    correct[n] ~ bernoulli(theta);
  }
}
