# http://proceedings.mlr.press/v80/zhou18e/zhou18e.pdf

import numpy as np
import scipy.stats as stats

T = 1000 # ステップ数
K = 3    # 候補数
sigma = 0.1
delta = 0.2
eta = 0.1

pi = np.array([0.7, 0.2, 0.1])
v = np.array([0.6, 0.5, 0.4])


def gumbel():
    r = np.random.random(K)
    e = 1e-12
    return -np.log(-np.log(r + e) + e)

def beta_f(m, delta):
    val = np.log(1 / delta)
    val += 3 * np.log(np.log(1 / delta))
    val += 3 / 2 * np.log(np.log(np.e * m))
    return (val * 2 * eta / m) ** 0.5

def sample(mean):
    return 1 if np.random.rand() < mean else 0

alpha, beta = np.ones(len(pi)), np.ones(len(pi))
likelihood = [1 for _ in pi]

def f(mu, eps):
    ch = pi * np.exp(eps) * (mu == np.max(mu))
    mom = np.prod(stats.beta(1, 1).pdf(mu))
    return ch / mom

n = np.zeros(len(pi))
for t in range(T):
    eps = gumbel()
    m, d = 1, []
    f_sum = np.zeros(K)
    while True:
        mu = np.random.beta(alpha, beta)
        d.append(mu)
        f_sum += f(mu, eps)
        f_mean = f_sum / m
        #print(f_mean)
        i1 = np.argmax(f_mean)
        f1 = f_mean[i1]
        f_mean[i1] = -float('inf')
        i2 = np.argmax(f_mean)
        f2 = f_mean[i2]
        m += 1

        #print(beta_f(m, delta))
        if f1 - f2 > 2 * beta_f(m, delta) - sigma:
            break
    I = i1
    x = sample(v[I])
    if x > 0.5: alpha[I] += 1
    else: beta[I] += 1

    n[I] += 1
    print(n)

