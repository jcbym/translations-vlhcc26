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
}
