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
  // array[I, T] real<lower=0, upper=1> theta;


  vector[N] ability;
  vector[I] goodness;
  vector[T] difficulty;
  array[I, T] real fitness;
}

model {
  //for (i in 1:I) {
  //  theta[i] ~ beta(1, 1);
  //}
  goodness ~ normal(0, 10);
  difficulty ~ normal(0, 10);
  for (t in 1:T) {
    fitness ~ normal(0, 10);
  }

  for (n in 1:N) {
    vector[T] logits;
    for (t in 1:T) {
      logits[t] =
        ability[n] + goodness[interface[n]] +
          difficulty[t] + fitness[interface[n], t];
    }
    correct[n] ~ bernoulli_logit(logits);
  }
}

// generated quantities {
//   array[I] real<lower=0, upper=1> phiInterface;
//   array[T] real<lower=0, upper=1> phiTask;
// 
//   for (t in 1:T) {
//     phiTask[t] = 1;
//     for (i in 1:I) {
//       phiTask[t] *= theta[i, t];
//     }
//   }
// 
//   for (i in 1:I) {
//     phiInterface[i] = 1;
//     for (t in 1:T) {
//       phiInterface[i] *= theta[i, t];
//     }
//   }
// }
