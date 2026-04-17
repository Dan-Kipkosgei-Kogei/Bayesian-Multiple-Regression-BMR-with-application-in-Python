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
- The probability density function of $Y$ given $X$ is:
  
  $P(y \mid x, \boldsymbol{\beta}, \sigma^2)= \frac{1}{\sqrt{2\pi\sigma^2}}\exp\left[
       -\frac{(y - f(x, \boldsymbol{\beta}))^2}{2\sigma^2}\right]$

- For $N$ observations:

  $L(Y \mid X, \boldsymbol{\beta}, \sigma^2)= \prod_{i=1}^{N} P(y_i \mid x_{i1}, x_{i2}, \ldots, x_{ip})$

- which simplifies to:

  $L(Y \mid X, \boldsymbol{\beta}, \sigma^2)= \prod_{i=1}^{N}\frac{1}{\sqrt{2\pi\sigma^2}}
      \exp\left[-\frac{(y_i - f(x_i, \boldsymbol{\beta}))^2}{2\sigma^2}\right]$

- Taking the logarithm of the likelihood function:

  $ln L(Y \mid X, \boldsymbol{\beta}, \sigma^2)= -\frac{N}{2} ln(2\pi\sigma^2)- \frac{1}{2\sigma^2}
      \sum_{i=1}^{N} (y_i - f(x_i, \boldsymbol{\beta}))^2$

- We define precision $\tau$ as:

  $\tau = \frac{1}{\sigma^2}$

- Substituting into the likelihood function:
  
   $\ln L(Y \mid X, \boldsymbol{\beta}, \sigma^2)=-\frac{N}{2} \ln(2\pi)
    +\frac{N}{2} \ln(\tau)-\frac{\tau}{2}\sum_{i=1}^{N} (y_i - f(x_i, \boldsymbol{\beta}))^2$

- The negative log-likelihood is:

   $-\ln L(Y \mid X, \boldsymbol{\beta}, \sigma^2)
   = \frac{\tau}{2}\sum_{i=1}^{N} (y_i -f(x_i, \boldsymbol{\beta}))^2+ constant$

- Taking the logarithm of the posterior:

   $\ln P(\boldsymbol{\beta} \mid X, \alpha, \tau)= \ln L(Y \mid X, \boldsymbol{\beta}, \tau)
   +\ln P(\boldsymbol{\beta} \mid \alpha)$

- Substituting the expressions:

   $\hat{\boldsymbol{\beta}}= \frac{\tau}{2} \sum_{i=1}^{N} (y_i - f(x_i, \boldsymbol{\beta}))^2
  +\frac{\alpha}{2} \boldsymbol{\beta}^T \boldsymbol{\beta}$

  - Minimizing this expression gives the maximum posterior estimate, which is equivalent to ridge regression.
  - Bayesian regression provides a probabilistic framework for linear regression by incorporating prior knowledge.
  - Instead of estimating a single set of parameters, we obtain a distribution over possible parameters, which enhances           robustness in situations with limited data or multicollinearity.
  ## Modeling Student Performance using Bayesian MLR
  
