
# プレーヤー数
N = 3
# 勝利数
M = [
  [-1,  7,  8],
  [ 3, -1,  5],
  [ 2,  5, -1]
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

# MM法
pi = [1.0] * N 
for i in range(100):
  print(pi)
  npi = [1.0] * N
  for i in range(N):
    isum = VG / (pi[i] + 1.0) # 架空の試合  
    for j in range(N):
      if i != j:
        isum += (M[i][j] + M[j][i]) / (pi[i] + pi[j])
    npi[i] = (W[i] + VG / 2) / isum
  nsum = 0
  for i in range(N):
    nsum += npi[i]
  for i in range(N):
    pi[i] = npi[i] * N / nsum

