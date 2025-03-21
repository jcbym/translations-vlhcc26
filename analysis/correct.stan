data {
  // The number of observations
  int<lower=0> N;

  // The number of interfaces
  int<lower=1> I;

  // Assigned interface
  array[N] int<lower=1, upper=I> interface;

  // Observed correctness
  array[N] int<lower=0, upper=1> correct;
}

parameters {
  // Probability of correctness on interface i
  vector<lower=0, upper=1>[I] theta;
}

model {
  for (n in 1:N) {
    correct[n] ~ bernoulli(theta[interface[n]]);
  }
}

generated quantities {
  matrix[I, I] esTheta;
  for (i1 in 1:I) {
    for (i2 in 1:I) {
      esTheta[i1, i2] = 2 * asin(theta[i1]) - 2 * asin(theta[i2]);
    }
  }
}
