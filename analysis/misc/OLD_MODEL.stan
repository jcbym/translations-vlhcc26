data {
  // The number of observations
  int<lower=0> N;

  // The number of tasks
  int<lower=1> T;

  // The number of interfaces (levels)
  int<lower=1> I;

  // Observed interface
  array[N] int<lower=1, upper=I> interface;

  // Observed correctness
  array[N] int<lower=0, upper=T> correct;
}

parameters {
  // Probability of correctness on interface i, task t
  array[I] real <lower=0, upper=1> theta;
}

model {
  for (i in 1:I) {
    theta[i] ~ beta(1, 1);
  }
  for (n in 1:N) {
    correct[n] ~ binomial(T, theta[interface[n]]);
  }
}
