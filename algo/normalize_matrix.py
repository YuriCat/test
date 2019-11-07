# 反復計算による遷移確率行列の正規化

import numpy as np

p = [
  [1, 2, 3],
  [4, 5, 6],
  [7, 8, 9]
]

def normalize(p):
  q = np.copy(p)
  for _ in range(10):
    for i in range(q.shape[0]):
      q[i,:] /= q[i,:].sum()
    for j in range(q.shape[1]):
      q[:,j] /= q[:,j].sum()
    print(q)
  return q

p = normalize(np.array(p, dtype=float))

print(p)
