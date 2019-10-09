# Gumbel Softmax

import numpy as np

pi = np.array([0.7, 0.2, 0.1])
tau = 1

st = np.zeros_like(pi)
for _ in range(10000):
    #  Gumbel分布に従う乱数を生成
    u = np.random.random(len(pi))
    g = -np.log(-np.log(u))
    #g = np.random.gumbel(size=len(pi))
    # argmax(log pi + 乱数) は pi に従う
    z = np.argmax(np.log(pi) / tau + g)
    st[z] += 1

print(st)

st = np.zeros_like(pi)
for _ in range(10000):
    #  Gumbel分布に従う乱数を生成
    u = np.random.random(len(pi))
    g = -np.log(-np.log(u))
    #g = np.random.gumbel(size=len(pi))
    # argmax(log pi + 乱数) は pi に従うcccccckdbulifkhrgnlkikcnhggifrigvuefitnihlgc

    o = (np.log(pi) + g) / tau
    d = np.exp(o) / np.exp(o).sum()
    z = np.random.choice(np.arange(len(pi)), p=d)
    #z = np.argmax(d)
    st[z] += 1

print(st)

# 以下実験

st = np.zeros_like(pi)
for _ in range(10000):
    g = np.random.gumbel(scale=pi *(1 / pi), size=len(pi))
    z = np.argmax(g)
    st[z] += 1
print(st)


