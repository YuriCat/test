
# カーネルによる重み付けがある場合のUCB選択

import random
import math

V0 = [0, 3, 0, 3, 8, 2, 4, 1, 0, 0, 2, 10, 0, 3, 1, 0]
K = [1, 2, 3, 4, 3, 2, 1]

V = [sum([K[j] * (V0[i + j - len(K) // 2] / 2 if 0 <= i + j - len(K) // 2 < len(V0) else 0)
     for j in range(len(K))]) / sum(K) for i in range(len(V0))]

print(V)

'''NQ = [(0, 0) for _ in V]

for m in range(100000):
    index = 0
    max_ucb = -10000
    for i, (n, q) in enumerate(NQ):
        #ucb = q + (math.log2(m + 1) / (n + 1)) ** 0.5
        ucb = q + ((m + 1) ** 0.5) / (n + 1)
        if ucb > max_ucb:
            index = i
            max_ucb = ucb
    v = random.random() * V0[index]
    n, q = NQ[index]
    NQ[index] = (n + 1, (q * n + v) / (n + 1))

print([n for n, _ in NQ])
print([q for _, q in NQ])

NQX = [(0, 0, 0) for _ in V]
RNQ = [(0, 0) for _ in V]

for m in range(100000):
    index = 0
    max_ucb = -10000
    for i, (n, q, x) in enumerate(NQX):
        ucb = x + (math.log2(m + 1) / (n + 1)) ** 0.5
        #ucb = x + ((m + 1) ** 0.5) / (n + 1)
        if ucb > max_ucb:
            index = i
            max_ucb = ucb
    v = random.random() * V0[index]
    n, oq, x = NQX[index]
    nq = (oq * n + v) / (n + 1)
    NQX[index] = (n, nq, x)
    for j in range(len(K)):
        u = index + j - len(K) // 2
        if 0 <= u < len(V0):
            c = K[j] / sum(K)
            n, q, x = NQX[u]
            NQX[u] = (n + c, q, x + c * (nq - oq))

    n, q = RNQ[index]
    RNQ[index] = (n + 1, (q * n + v) / (n + 1))


print([int(n * 10) / 10 for n, _, _ in NQX])
print([q for _, q, _ in NQX])
print([x for _, _, x in NQX])

print([n for n, _ in RNQ])
print([q for _, q in RNQ])


NQX = [(0, 0, 0) for _ in V]
RNQ = [(0, 0) for _ in V]

for m in range(100000):
    index = 0
    max_ucb = -10000
    for i, (n, q, x) in enumerate(NQX):
        ucb = q + (math.log2(m + 1) / (n + 1)) ** 0.5
        #ucb = q + ((m + 1) ** 0.5) / (n + 1)
        if ucb > max_ucb:
            index = i
            max_ucb = ucb
    v = random.random() * V0[index]
    n, q, x = NQX[index]
    NQX[index] = (n + 1, q, x)
    for j in range(len(K)):
        u = index + j - len(K) // 2
        if 0 <= u < len(V0):
            c = K[j] / sum(K)
            n, q, x = NQX[u]
            NQX[u] = (n, (q * x + v * c) / (x + c), x + c)

    n, q = RNQ[index]
    RNQ[index] = (n + 1, (q * n + v) / (n + 1))


print([int(n * 10) / 10 for n, _, _ in NQX])
print([q for _, q, _ in NQX])
print([x for _, _, x in NQX])

print([n for n, _ in RNQ])
print([q for _, q in RNQ])'''

def dot(a, b):
    x = 0
    for i, a_ in enumerate(a):
        x += a_ * b[i]
    return x

KD = [dot([0] * i + K + [0] * (len(K) * 2 - 2 - i), [0] * (len(K) - 1) + K + [0] * (len(K) - 1)) / (sum(K) ** 2) for i in range(len(K) * 2 - 1)]
print(KD)

# this is OK
NQEX = [(0, 0, 0, 0) for _ in V]
RNQ = [(0, 0) for _ in V]

for m in range(100000):
    index = 0
    max_ucb = -10000
    for i, (n, _, _, x) in enumerate(NQEX):
        ucb = x + (math.log2(m + 1) / (n + 1)) ** 0.5
        #ucb = x + ((m + 1) ** 0.5) / (n + 1)
        if ucb > max_ucb:
            index = i
            max_ucb = ucb
    v = random.random() * V0[index]
    n, oq, e, x = NQEX[index]
    nq = (oq * n + v) / (n + 1)
    NQEX[index] = (n + 1, nq, e, x)
    for j in range(len(K)):
        u = index + j - len(K) // 2
        if 0 <= u < len(V0):
            c = K[j] / sum(K)
            n, q, e, x = NQEX[u]
            NQEX[u] = (n, q, e + c * (nq - oq), x)
    for j in range(len(KD)):
        u = index + j - len(KD) // 2
        if 0 <= u < len(V0):
            c = KD[j]
            n, q, e, x = NQEX[u]
            NQEX[u] = (n, q, e, x + c * (nq - oq))

    n, q = RNQ[index]
    RNQ[index] = (n + 1, (q * n + v) / (n + 1))

print([int(n * 10) / 10 for n, _, _, _ in NQEX])
print([q for _, q, _, _ in NQEX])
print([e for _, _, e, _ in NQEX])

print([n for n, _ in RNQ])
print([q for _, q in RNQ])

# hieralchiral

'''HNQEX = [
    [(0, 0, 0, 0) for _ in range(2)],
    [(0, 0, 0, 0) for _ in range(4)],
    [(0, 0, 0, 0) for _ in range(8)],
    [(0, 0, 0, 0) for _ in range(16)]
]
RNQ = [(0, 0) for _ in V]

for m in range(100000):
    index = 0
    for d in range(4):
        best_index = 0
        max_ucb = -10000
        for i in range(index * 2, index * 2 + 2):
            n, _, _, x = HNQEX[d][i]
            ucb = x + (math.log2(m + 1) / (n + 1)) ** 0.5
            #ucb = x + ((m + 1) ** 0.5) / (n + 1)
            if ucb > max_ucb:
                best_index = i
                max_ucb = ucb
        index = best_index

    v = random.random() * V0[index]

    # last layer
    n, oq, e, x = HNQEX[-1][index]
    nq = (oq * n + v) / (n + 1)
    HNQEX[-1][index] = (n + 1, nq, e, x)
    for j in range(len(K)):
        u = index + j - len(K) // 2
        if 0 <= u < len(V0):
            c = K[j] / sum(K)
            n, q, e, x = HNQEX[-1][u]
            HNQEX[-1][u] = (n, q, e + c * (nq - oq), x)
    for j in range(len(KD)):
        u = index + j - len(KD) // 2
        if 0 <= u < len(V0):
            c = KD[j]
            n, q, e, x = HNQEX[-1][u]
            HNQEX[-1][u] = (n, q, e, x + c * (nq - oq))

    # upper layer
    for d in range(2, -1, -1):
        for i in range(len(HNQEX[d])):
            n = HNQEX[d + 1][i * 2][0] + HNQEX[d + 1][i * 2 + 1][0]
            q = (HNQEX[d + 1][i * 2][1] + HNQEX[d + 1][i * 2 + 1][1]) / 2
            e = (HNQEX[d + 1][i * 2][2] + HNQEX[d + 1][i * 2 + 1][2]) / 2
            x = max(HNQEX[d + 1][i * 2][3], HNQEX[d + 1][i * 2 + 1][3])
            HNQEX[d][i] = n, q, e, x

    n, q = RNQ[index]
    RNQ[index] = (n + 1, (q * n + v) / (n + 1))


print([int(n * 10) / 10 for n, _, _, _ in HNQEX[-1]])
print([q for _, q, _, _ in HNQEX[-1]])
print([e for _, _, e, _ in HNQEX[-1]])

print([n for n, _ in RNQ])
print([q for _, q in RNQ])'''