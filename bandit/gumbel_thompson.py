# http://proceedings.mlr.press/v80/zhou18e/zhou18e.pdf

import numpy as np

T = 1000 # ステップ数
K = 3    # 候補数
sigma = 0.1
delta = 0.2

pi = np.array([0.7, 0.2, 0.1])
v = np.array([0.5, 0.5, 0.4])

def sample(mean):
    return 1 if np.random.rand() < mean else 0

n = np.zeros(K)
beta = np.ones((2, K)) * 2

for t in range(T):
    p, M = np.zeros(K), 10000
    while True:
        a0 = np.random.choice(np.arange(K), p=pi)
        a1 = np.argmax(np.random.beta(beta[0,:], beta[1,:]))
        if a0 == a1:
            a = a0
            break
    r = sample(v[a])
    beta[1 - r][a] += 1

    n[a] += 1
    print(n)


