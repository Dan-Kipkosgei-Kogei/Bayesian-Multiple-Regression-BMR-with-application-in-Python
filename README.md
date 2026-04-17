# Bayesian Multiple Regression (BMR) with application in Python
### By Dan Kipkosgei Kogei
## Introduction
- Bayesian linear regression extends classical linear regression by treating model parameters as random variables rather than fixed unknowns.
- Traditional regression models are defined by precise parameters or weights and typically assume no uncertainty in these estimates.
- Bayesian regression was introduced to model the relationship between predictor and target variables probabilistically, introducing uncertainty into the modeling process.
- At the heart of any Bayesian model is:

$p(\theta \mid y) = \frac{p(y \mid \theta)\, p(\theta)}{p(y)}$

  - $y$: observed data
  - $\theta$: parameters
  - $p(\theta)$: prior
  - $p(y \mid \theta)$: likelihood
  - $p(\theta \mid y)$: posterior
