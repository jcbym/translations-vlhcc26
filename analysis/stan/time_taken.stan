data {
  // The number of observations
  int<lower=0> N;

  // The number of interfaces
  int<lower=1> I;

  // Assigned interface
  array[N] int<lower=1, upper=I> interface;

  // Observed time taken (minutes)
  vector<lower=0>[N] timeTaken;
}

parameters {
  // Log-normal parameters
  vector<lower=0, upper=log(90)>[I] logmu;
  real<lower=0, upper=log(90)> sigma;
}

model {
  for (n in 1:N) {
    timeTaken[n] ~ lognormal(logmu[interface[n]], sigma);
  }
}

generated quantities {
  vector[I] mu = exp(logmu);

  matrix[I, I] logmuES;
  for (i1 in 1:I) {
    for (i2 in 1:I) {
      // logmuES[i1, i2] = (logmu[i1] - logmu[i2]) / sigma;
      logmuES[i1, i2] = mu[i1] - mu[i2];
    }
  }
}
