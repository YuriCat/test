import time
import numpy as np

L = 512
N = 100000
K = 5

xb = np.random.randn(N, L)
xb /= (xb ** 2).sum(axis=-1, keepdims=True) ** 0.5

xq = np.random.randn(1, L)
xq /= (xq ** 2).sum(axis=-1, keepdims=True) ** 0.5

print(xq)
print((xq ** 2).sum(axis=-1))

def metric(x, y):
    return (x @ y.T).squeeze()

print('sklearn')

import sklearn
from sklearn.neighbors import NearestNeighbors

nn = NearestNeighbors(algorithm='brute', metric='cosine')
nn.fit(xb)
st = time.time()
_, result = nn.kneighbors(xq, n_neighbors=K)
print(time.time() - st)
print(result)
print(metric(xb[result], xq))

nn = NearestNeighbors(algorithm='kd_tree', metric='euclidean')
nn.fit(xb)
st = time.time()
_, result = nn.kneighbors(xq, n_neighbors=K)
print(time.time() - st)
print(result)
print(metric(xb[result], xq))

nn = NearestNeighbors(algorithm='ball_tree', metric='euclidean')
nn.fit(xb)
st = time.time()
_, result = nn.kneighbors(xq, n_neighbors=K)
print(time.time() - st)
print(result)
print(metric(xb[result], xq))

print('annoy')

from annoy import AnnoyIndex

t = AnnoyIndex(xb.shape[1], 'angular')  
for i, v in enumerate(xb):
    t.add_item(i, v)

t.build(100)
result = []
st = time.time()
for i, v in enumerate(xq):
    r = t.get_nns_by_vector(v, K)
    result.append(r)
result = np.array(result)
print(time.time() - st)
print(result)
print(metric(xb[result], xq))