import numpy as np

P = [0.9, 0.1]
p = P
G = [0] * len(P)

for _ in range(10000):
    k = np.random.choice(list(range(len(P))), p=P)
    l = np.random.choice(list(range(len(p))), p=p)
    if k == l:
        G[k] += 1
print(G)
