
# 対数尤度のstatsはどういう値になるか？？

import numpy as np


A = 30
N = 200000
base = np.log(A)

print(base)

likelihood_sum = N * np.log(1 / A)

print(base + likelihood_sum / N)

likelihood_sum = 0
for _ in range(N):
    policy = np.random.dirichlet([1 / A] * A).astype(np.float64)
    likelihood_sum += np.log(policy[0])

print(base + likelihood_sum / N)

likelihood_sum = 0
for _ in range(N):
    policy = np.random.dirichlet([1 / A / A] * A).astype(np.float64)
    likelihood_sum += (policy * np.log(policy)).sum()

print(base + likelihood_sum / N)

likelihood_sum = 0
for _ in range(N):
    policy = np.random.dirichlet([1 / A] * A).astype(np.float64)
    likelihood_sum += (policy * np.log(policy)).sum()

print(base + likelihood_sum / N)

likelihood_sum = 0
for _ in range(N):
    policy = np.random.dirichlet([1] * A).astype(np.float64)
    likelihood_sum += (policy * np.log(policy)).sum()

print(base + likelihood_sum / N)

likelihood_sum = 0
for _ in range(N):
    policy = np.random.dirichlet([A] * A).astype(np.float64)
    likelihood_sum += (policy * np.log(policy)).sum()

print(base + likelihood_sum / N)

likelihood_sum = 0
for _ in range(N):
    policy = np.random.dirichlet([10000] * A).astype(np.float64)
    likelihood_sum += (policy * np.log(policy)).sum()

print(base + likelihood_sum / N)

likelihood_sum = N * A * (1 / A * np.log(1 / A))

print(base + likelihood_sum / N)