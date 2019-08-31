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
print('score = ', result)
print(metric(xb[result], xq))

nn = NearestNeighbors(algorithm='kd_tree', metric='euclidean')
nn.fit(xb)
st = time.time()
_, result = nn.kneighbors(xq, n_neighbors=K)
print(time.time() - st)
print('score = ', result)
print(metric(xb[result], xq))

nn = NearestNeighbors(algorithm='ball_tree', metric='euclidean')
nn.fit(xb)
st = time.time()
_, result = nn.kneighbors(xq, n_neighbors=K)
print(time.time() - st)
print('score = ', result)
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
print('score = ', result)
print(metric(xb[result], xq))

print('lsh')

class LSHIndex:
    def __init__(self,d,k,L):
        self.d = d
        self.k = k
        self.powers = 2 ** np.arange(k, dtype=np.int64)
        self.hash_keys = None
        self.hash_tables = []
        self.resize(L)

    def resize(self,L):
        """ update the number of hash tables to be used """
        if L < len(self.hash_tables):
            self.hash_tables = self.hash_tables[:L]
        else:
            # initialise a new hash table for each hash function
            hash_keys = np.random.randn(L - len(self.hash_tables), self.k, self.d)
            if self.hash_keys is None:
                self.hash_keys = hash_keys
            else:
                self.hash_keys = np.r_[self.hash_keys, hash_keys]
            self.hash_tables.extend([{} for _ in range(len(self.hash_tables), L)])

    def hash(self, g, p):
        return np.dot(self.powers, np.matmul(g, p.T) > 0)

    def index(self, points):
        """ index the supplied points """
        self.points = points
        keyss = self.hash(self.hash_keys, points)
        for i, keys in enumerate(keyss):
            for ix, key in enumerate(keys):
                key = keys[ix]
                table = self.hash_tables[i]
                if key not in table:
                    table[key] = []
                table[key].append(ix)
        # reset stats
        self.tot_touched = 0
        self.num_queries = 0

    def query(self,q,metric,max_results):
        """ find the max_results closest indexed points to q according to the supplied metric """
        candidates = set()
        keys = self.hash(self.hash_keys, q)
        for i, key in enumerate(keys):
            matches = self.hash_tables[i].get(key[0], [])
            candidates.update(matches)
        candidates = list(candidates)
        # update stats
        self.tot_touched += len(candidates)
        self.num_queries += 1
        # rerank candidates
        scores = [-metric(q, self.points[ix]) for ix in candidates]
        top_indice = np.argpartition(scores, kth=(1, max_results))[:max_results]
        return [candidates[i] for i in top_indice]

    def get_avg_touched(self):
        """ mean number of candidates inspected per query """
        return self.tot_touched/self.num_queries

k, l = 12, 512
lsh = LSHIndex(L, k, l)
lsh.index(xb)

st = time.time()
result = lsh.query(xq, metric, K)
print(time.time() - st)
print('score = ', result)
print(metric(xb[result], xq))
print('touch_rate = ', lsh.get_avg_touched() / len(xb))