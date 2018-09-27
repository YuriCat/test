
# 局所性鋭敏型ハッシュ
import numpy as np
from scipy.stats import pearsonr

N = 4096 # 元データの次元
K = 256 # ハッシュ値のビット数

R = np.random.rand(K, N) * 2 - 1
#A = np.random.randn(N)
#B = np.random.rand() * (2 ** K)

#print(R)

# バケットによるもの
def hash(v):
  m = np.matmul(R, v.reshape(N, 1))
  k = np.heaviside(m.flatten(), 0).astype(np.int32)
  return ''.join(str(val) for val in k)
'''# 安定分布によるもの
def hash(v):
  n = np.mod(np.dot(A, v) + B, 2 ** K)
  print(n)
  return format(int(n), '#0' + str(K + 2) + 'b')[2:]'''

def hamming_distance(k1, k2):
  return (np.array(list(k1)) != np.array(list(k2))).sum()

# 試し打ち
for i in range(16):
  v = np.random.rand(N) * 2 - 1
  key = hash(v)
  print(key)

# ハミング距離とその他の距離の比較
x, y = [], []
for i in range(1000):
  v1, v2 = np.random.rand(N) * 2 - 1, np.random.rand(N) * 2 - 1
  k1, k2 = hash(v1), hash(v2)

  hamming = hamming_distance(k1, k2)
  l1_norm = np.abs(v1 - v2).sum()
  l2_norm = np.linalg.norm(v1 - v2)
  print(hamming, l1_norm, l2_norm)
  x.append(hamming)
  y.append(l1_norm)

r, p = pearsonr(x, y)
print(r, p)

import matplotlib.pyplot as plt
plt.scatter(x, y)
plt.show()