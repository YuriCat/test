import copy, math
import numpy as np

N = 6
M = 2

def sample(n):
    s = S = list(range(n))
    np.random.shuffle(s)
    return s[:N//2], s[N//2:]

#def present(p, m):
#    p = np.copy(p)
#    present = sorted([p[1], reverse=True)

def invert(p, m):
    p = copy.deepcopy(p)
    # |m| 枚の札を選ぶ
    #print(p)
    for _ in range(m):
        thrown = p[1][np.random.randint(len(p[1]))]
        p[1].remove(thrown)
        p[0].append(thrown)
    # 献上札を選ぶ
    strongers = []
    p1max = np.max(p[1])
    for x in p[0]:
        if x > p1max:
            strongers.append(x)
    #print(p, strongers)
    num_strongers = len(strongers)
    if num_strongers < m:
        return None, 0
    else:
        for _ in range(m):
            presented = strongers[np.random.randint(len(strongers))]
            p[0].remove(presented)
            strongers.remove(presented)
            p[1].append(presented)
    def combination(n, k):
        return math.factorial(n) / math.factorial(k) / math.factorial(n - k)
    return p, combination(num_strongers, m)

for _ in range(5):
    k = sample(N)
    kk = invert(k, M)
    print(k, kk)

mp = {}
samples = 0

while samples < 100000:
    p = sample(N)
    if 1 not in p[1]:
        continue
    '''if 0 not in p[1]:
        continue'''
    samples += 1
    k, r = invert(p, M)
    if k is not None:
        k = str((sorted(k[0], reverse=True), sorted(k[1], reverse=True)))
    else:
        k = 'None'
    if k not in mp:
        mp[k] = 0
    mp[k] += r
for key in sorted(mp.keys()):
    print(key, mp[key])


