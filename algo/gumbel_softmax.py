# Gumbel Softmax

import numpy as np

pi = np.array([0.7, 0.2, 0.1])
tau = 1

st = np.zeros_like(pi)
for _ in range(10000):
    u = np.random.random(len(pi))
    g = -np.log(-np.log(u))
    z = np.argmax(np.log(pi) / tau + g)
    st[z] += 1

print(st)
