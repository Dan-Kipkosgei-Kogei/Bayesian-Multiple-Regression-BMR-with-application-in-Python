# Bayesian Multiple Regression (BMR) with application in Python
### By Dan Kipkosgei Kogei
## Introduction
- Bayesian linear regression extends classical linear regression by treating model parameters as random variables rather than fixed unknowns.
- Traditional regression models are defined by precise parameters or weights and typically assume no uncertainty in these estimates.
- Bayesian regression was introduced to model the relationship between predictor and target variables probabilistically, introducing uncertainty into the modeling process.

## Bayesian Regression Formulation
- At the heart of any Bayesian model is the Bayes rule defined as:

   $p(\theta \mid y) = \frac{p(y \mid \theta)\, p(\theta)}{p(y)}$

  - $y$: observed data
  - $\theta$: parameters
  - $p(\theta)$: prior
  - $p(y \mid \theta)$: likelihood
  - $p(\theta \mid y)$: posterior
- For a dataset with $n$ samples, the linear relationship is:
  
  $y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_p x_p + \epsilon$

     where $\beta$ are regression coefficients and $\epsilon \sim \mathcal{N}(0, \sigma^2)$
 - The assumptions are;

      - The error terms $\epsilon = \{\epsilon_1, \epsilon_2, \ldots, \epsilon_N\}$
        are independent and identically distributed (i.i.d.).

      - The target variable $Y$ follows a normal distribution with mean 
        $\mu = f(x, \boldsymbol{\beta})$
        and variance $\sigma^2$, i.e., $Y \sim \mathcal{N}(f(x, \boldsymbol{\beta}), \sigma^2)$.
