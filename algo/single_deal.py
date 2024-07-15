
N = [3, 2, 1]
C = [1, 1, 1, 1, 1, 1]

W = [
   [1, 1, 1, 1, 1, 1],
   [1, 2, 3, 4, 5, 6],
   [0, 0, 0, 1, 2, 3],
]

# normalization

def normalize(w, n, c):
    for _ in range(10):
        for p in range(len(N)):
            sum = 0
            for i in range(len(c)):
                sum += w[p][i]
            for i in range(len(c)):
                w[p][i] = w[p][i] / (sum / n[p]) if sum > 0 and n[p] > 0 else 0
        for i in range(len(c)):
            sum = 0
            for p in range(len(n)):
                sum += w[p][i]
            for p in range(len(n)):
                w[p][i] = w[p][i] / (sum / c[i]) if sum > 0 and c[i] > 0 else 0

print(N)
print(C)
normalize(W, N, C)
for p, w in enumerate(W):
    print(w)

import random
import copy

# random card order 
MAP = [[0 for _ in C] for __ in N]
ERR = 0

for i in range(100000):
    res = [[] for _ in range(len(N))]
    cards = list(range(len(C)))
    random.shuffle(cards)
    n = copy.deepcopy(N)
    for c in cards:
        wsum = 0
        for p in range(len(n)):
            wsum += W[p][c] * n[p] / N[p]
        r = random.random() * wsum
        for p in range(len(n)):
            r -= W[p][c] * n[p] / N[p]
            if r <= 0:
                break
        n[p] -= 1
        res[p].append(c)

    #print(res)
    for p, lst in enumerate(res):
        if len(lst) != N[p]:
           ERR += 1
        for i in lst:
            MAP[p][i] += 1

for p, m in enumerate(MAP):
    print(m)
print(ERR)

# random player order
MAP = [[0 for _ in C] for __ in N]
ERR = 0

for i in range(10000):
    res = [[] for _ in range(len(N))]
    players = list(range(len(N)))
    #random.shuffle(players)
    w = copy.deepcopy(W)
    n = copy.deepcopy(N)
    c = copy.deepcopy(C)
    for ip, p in enumerate(players):
        if ip > 0:
            normalize(w, n, c)
        wsum = n[p]
        #print(ip, p, wsum, sum(w[p])) 
        while n[p] > 0:
            r = random.random() * wsum
            for i in range(len(C)):
                if c[i] <= 0:
                    continue
                r -= w[p][i]
                if r <= 0:
                    break
            wsum -= w[p][i]
            c[i] -= 1
            n[p] -= 1
            res[p].append(i)

    #print(res)
    for p, lst in enumerate(res):
        if len(lst) != N[p]:
           ERR += 1
        for i in lst:
           MAP[p][i] += 1

for p, m in enumerate(MAP):
    print(m, ':', sum(m))
print(ERR)


# random deal
MAP = [[0 for _ in C] for __ in N]
ERR = 0

for i in range(10000):
    res = [[] for _ in range(len(N))]
    players = list(range(len(N)))
    #random.shuffle(players)
    w = copy.deepcopy(W)
    n = copy.deepcopy(N)
    c = copy.deepcopy(C)
    for ip, p in enumerate(players):
        if ip > 0:
            normalize(w, n, c)
        wsum = n[p]
        cards = [i for i in range(len(c)) if c[i] > 0]
        for i in cards:
            r = random.random()
            if r < w[p][i] * n[p] / wsum:
                c[i] -= 1
                n[p] -= 1
                res[p].append(i)
            wsum -= w[p][i]
            if n[p] <= 0:
                break

    #print(res)
    for p, lst in enumerate(res):
        if len(lst) != N[p]:
            ERR += 1
        for i in lst:
            MAP[p][i] += 1

for p, m in enumerate(MAP):
    print(m, ':', sum(m))
print(ERR)

