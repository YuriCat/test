# 正規分布のベイズ更新によるパラメータ推定
# https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf

import numpy as np

# 母集団の分布
mu_, sigma_ = 10, 5

def sample(n=1):
  return np.random.randn(n) * sigma_ + mu_

# 事前分布
mu_mean0, mu_std0 = 0, 1 # muの事前分布(正規分布)
sigma_mean0, sigma_std0 = 1, 2 # sigmaの事前分布(逆ガンマ分布)

mu_mu0 = mu_mean0
mu_kappa0 = (1 / mu_std0) ** 2
sigma_alpha0 = (sigma_mean0 / sigma_std0) ** 2 + 2
sigma_beta0 = (sigma_alpha0 - 1) * sigma_mean0

print('answer =', mu_mu0, sigma_mean0)

def update(mu_mu, mu_kappa, sigma_alpha, sigma_beta, samples):
  n = len(samples)
  alpha = sigma_alpha + n / 2
  beta = sigma_beta + (n * np.var(samples, ddof=0) + n * mu_kappa / (mu_kappa + n) * ((np.average(samples) - mu_mu) ** 2)) / 2
  mu = (mu_kappa * mu_mu + n * np.average(samples)) / (mu_kappa + n)
  kappa = mu_kappa + n
  print(mu, kappa, alpha, beta)
  return mu, kappa, alpha, beta

mu_mu, mu_kappa, sigma_alpha, sigma_beta = mu_mu0, mu_kappa0, sigma_alpha0, sigma_beta0
for i in range(100000):
  samples = sample(1)
  print(samples)
  mu_mu, mu_kappa, sigma_alpha, sigma_beta = update(mu_mu, mu_kappa, sigma_alpha, sigma_beta, samples)
  mu_mean, sigma_mean = mu_mu, (sigma_beta / (sigma_alpha - 1)) ** 0.5
  print(mu_mean, sigma_mean)

