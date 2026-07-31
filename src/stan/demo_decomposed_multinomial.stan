data {
  int<lower=2> K;                     // Number of categories
  int<lower=0> N;                     // Total number of trials
  array[K] int<lower=0> y;            // Observed counts for each category
  array[K] int<lower=0> U;            // Upper bounds for each category
  vector<lower=0>[K] alpha;           // Dirichlet prior hyperparameters
}

transformed data {
  // Verify that data satisfies the upper bounds and total sum
  for (k in 1:K) {
    if (y[k] > U[k]) {
      reject("Observed count y[", k, "] exceeds upper bound U[", k, "]");
    }
  }
  if (sum(y) != N) {
    reject("Sum of counts y must equal N");
  }
}

parameters {
  simplex[K] p;                       // True category probabilities
}

model {
  // Prior
  p ~ dirichlet(alpha);
  
  // Sequential Binomial Decomposition
  int remaining_trials = N;
  real remaining_p = 1.0;
  
  // Loop from category 1 to K-1
  for (k in 1:(K - 1)) {
    // Conditional probability for the current category
    real cond_p = p[k] / remaining_p;
    
    // Binomial contribution to the likelihood
    y[k] ~ binomial(remaining_trials, cond_p);
    
    // Update remaining trials and probability mass for the next iteration
    remaining_trials -= y[k];
    remaining_p -= p[k];
  }
  // The last category K is deterministic given the previous choices
  if (y[K] > remaining_p){
    target += negative_infinity();
  }
}
