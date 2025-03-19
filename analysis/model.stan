data {
  // The number of observations
  int<lower=0> N;

  // The number of tasks
  int<lower=1> T;

  // The number of interfaces
  int<lower=1> I;

  // Observed interface
  array[N] int<lower=1, upper=I> interface;

  // Observed correctness
  array[N, T] int<lower=0, upper=1> correct;
}

parameters {
  // Probability of correctness on interface i, task t
  array[I, T] real<lower=0, upper=1> theta;
}

model {
  for (i in 1:I) {
    theta[i] ~ beta(1, 1);
  }

  for (n in 1:N) {
    correct[n] ~ bernoulli(theta[interface[n]]);
  }
}

generated quantities {
  array[I] real<lower=0, upper=1> phiInterface;
  array[T] real<lower=0, upper=1> phiTask;

  for (t in 1:T) {
    phiTask[t] = 1;
    for (i in 1:I) {
      phiTask[t] *= theta[i, t];
    }
  }

  for (i in 1:I) {
    phiInterface[i] = 1;
    for (t in 1:T) {
      phiInterface[i] *= theta[i, t];
    }
  }
}
