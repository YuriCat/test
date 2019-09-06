# http://proceedings.mlr.press/v80/zhou18e/zhou18e.pdf

import numpy as np

T = 1000 # ステップ数
K = 3    # 候補数
sigma = 0.1
delta = 0.2

pi = np.array([0.7, 0.2, 0.1])
v = np.array([0.6, 0.5, 0.4])


def gumbel():
    r = np.random.random(K)
    e = 1e-12
    return -np.log(-np.log(r + e) + e)

def beta(m, delta):
    val = np.log(1 / delta)
    val += 3 * np.log(np.log(1 / delta))
    val += 3 / 2 * np.log(np.log(np.e * m))
    return val ** 0.5

def sample(mean):
    return 1 if np.random.rand() < mean else 0

X = [[] for _ in range(K)]
B = [p for p in pi]
likelihood = [1 for _ in pi]
Beta = [(5, 5) for _ i pi]

for t in range(T):
    eps = gumbel()
    m, d = 1, []
    f_sum = np.zeros(K)
    # calculate besy arm probability
    p, N = np.zeros(K), N
    for i in range(1000):
        b = np.random.beta()
        p[np.argmax(b)] += 1.0 / N
    while True:
        mu = np.random.choice(np.arange(K), p=B)
        d.append(mu)
        m += 1
        f_sum +=  / B
        i1 = np.argmax(f_sum / m)
        if f1 - f2 > 2 * beta(m, delta) - sigma:
            break
    I = i1
    x = sample(v[I])
    X[I].append(x)
    likelihood[I] *= Beta[I][x] / (Beta[I][0] + Beta[I][1])
    Beta[I][x] += 1


