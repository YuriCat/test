# Multidimensional Elo rating

# プレーヤー数
N = 3
# 勝利数
M = [
  [ 1,  7,  8],
  [ 3,  1,  5],
  [ 2,  5,  1]
]
VG = 100 # 事前分布として与える架空のプレーヤーとの試合数
# ゲーム数と勝利数の各チーム合計
W = [0.0] * N
G = [0.0] * N

for i in range(N):
  for j in range(N):
    if i != j:
      W[i] += M[i][j]
      G[i] += M[i][j]
      G[j] += M[i][j]
print(W)
print(G)

import numpy as np

def omega(k):
  omg = np.zeros((2 * k, 2 * k))
  for i in range(k):
    e0, e1 = np.zeros(2 * k), np.zeros(2 * k)
    e0[2 * i] = 1
    e1[2 * i + 1] = 1
    omg += np.outer(e0, e1.T) - np.outer(e1, e0.T)
  return omg

def sigmoid(x, a):
  return 1 / (1 + np.exp(-x / a))

def wp(r0, r1, c0, c1, k):
  return sigmoid((r0 - r1) + (c0.T @ omega(k) @ c1), 400)

def diff(delta, r0, r1, c0, c1, k):
  r_diff = [16 * delta, -16 * delta]
  c_diff0, c_diff1 = np.empty(2 * k), np.empty(2 * k)
  for i in range(k):
    c_diff0[2 * i]     = c1[2 * i + 1]
    c_diff0[2 * i + 1] = c1[2 * i]
    c_diff1[2 * i]     = c0[2 * i + 1]
    c_diff1[2 * i + 1] = c0[2 * i]
  return r_diff, [delta * c_diff0, delta * c_diff1]

print(omega(1))
print(omega(2))
print(omega(3))

epochs = 200

def learn(k):
  r = np.zeros((N))
  c = np.zeros((N, 2 * k))
  
  for _ in range(epochs):
    i, j = np.random.randint(0, N), np.random.randint(0, N)
    w = wp(r[i], r[j], c[i, :], c[j, :], k)
    dst = M[i][j] / (M[i][j] + M[j][i])
    print(w, dst)

    rd, cd = diff(dst - w, r[i], r[j], c[i, :], c[j, :], k)
    r[i] += rd[0]
    r[j] += rd[1]
    c[i] += cd[0]
    c[j] += cd[1]

    print(r)
    print(c)

#learn(1)