import time, pickle
import numpy as np

N = 10000
K = 512

al = np.random.random((N * K))
l = list(al)
ls = ' '.join(map(str, l))
divl = [al.reshape(N, K)[i] for i in range(N)]
lbin = al.dumps()
lbins = [al.reshape(N, K)[i].dumps() for i in range(N)]
print(len(lbins))

#print(ls)

t = time.time()
print(np.fromstring(ls, dtype=np.float64, sep=' ').dtype)
print(time.time() - t)

t = time.time()
print(np.array(list(map(float, ls.split()))).dtype)
print(time.time() - t)

t = time.time()
print(np.array([float(s) for s in ls.split()]).dtype)
print(time.time() - t)

t = time.time()
print(pickle.loads(lbin).shape)
print(time.time() - t)

t = time.time()
lst = []
for lb in lbins:
    lst.append(pickle.loads(lb))
print(np.array(lst).shape)
print(time.time() - t)

t = time.time()
print(np.array(divl).shape)
print(time.time() - t)
print(np.array(divl).sum())

from functools import lru_cache

@lru_cache(maxsize=10000)
def load_index(i):
    return pickle.loads(lbins[i])

for i in range(4):
    t = time.time()
    lst = []
    for i in range(N):
        lst.append(load_index(i))
    print(np.array(lst).shape)
    print(time.time() - t)

@lru_cache(maxsize=10000)
def load_bin(lb):
    return pickle.loads(lb)

for i in range(4):
    t = time.time()
    lst = []
    for lb in lbins:
        lst.append(load_bin(lb))
    print(np.array(lst).shape)
    print(time.time() - t)
    print(np.array(lst).sum())