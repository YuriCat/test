# 相互情報量

import numpy as np

p = [
  [0.4, 0.3],
  [0.2, 0.1]
]

q = [
  [0.25, 0.25],
  [0.25, 0.25]
]

r = [
  [0.97, 0.01],
  [0.01, 0.01]
]

s = [
  [0.6, 0.3],
  [0.1, 0.05]
]

t = [
  [0.4, 0.1],
  [0.1, 0.4]
]

u = [
  [0.4, 0.4],
  [0.1, 0.1]
]


def mutual_information(p):
  p = np.array(p, dtype=float) / np.sum(p)
  sum = 0
  isum, jsum = np.sum(p, axis=1), np.sum(p, axis=0)
  for i in range(p.shape[0]):
    for j in range(p.shape[1]):
      sum += p[i,j] * np.log(p[i,j] / (isum[i] * jsum[j]))
  return sum

for p in [p, q, r, s, t, u]:
  print(np.array(p))
  print(mutual_information(p))
