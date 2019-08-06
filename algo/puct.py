import numpy as np
v = [0, 1e-2]
n = [0, 0]
p = [0.75, 0.25]
eps = 1e-20
for _ in range(100000):
  pucb = [v[i] + p[i] * ((np.sum(n) + 1) ** 0.5) / (n[i] + 1) + np.random.randn() * eps for i in range(len(p))]
  n[np.argmax(pucb)] += 1
print(n)
