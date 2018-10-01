
# プレーヤー数
N = 3
# ホームでの試合数
G = [
  [ 0, 10, 10],
  [10,  0, 10],
  [10, 10,  0]
]
# ホームでの勝利数
W = [
  [ 0,  7,  8],
  [ 9,  0,  5],
  [ 3,  6,  0]
]
VG = 1 # 事前分布として与える架空のプレーヤーとの試合数(ホーム,アウェーそれぞれ)
# ホームゲーム数とホーム勝利数の各チーム合計
g_home, g_away = [0.0] * N, [0.0] * N
w_home, w_away = [0.0] * N, [0.0] * N
g_sum = 0
w_home_sum = 0

for i in range(N):
  for j in range(N):
    g_home[i] += G[i][j]
    w_home[i] += W[i][j]
    g_away[i] += G[j][i]
    w_away[i] += W[j][i]
    g_sum += G[i][j]
    w_home_sum += W[i][j]

print(g_home, g_away)
print(w_home, w_away)
print(g_sum)
print(w_home_sum)

# MM法
pi = [1.0] * N
home_bias = 1

for i in range(10):
  print(pi)
  print(home_bias)

  # home_bias の更新
  bias_isum = 0
  for i in range(N):
    for j in range(N):
      bias_isum += G[i][j] * pi[i] / (home_bias * pi[i] + pi[j])
    bias_isum += VG * pi[i] / (home_bias * pi[i] + 1.0)
  home_bias = (w_home_sum + VG * N / 2) / bias_isum

  # pi の更新
  npi = [1.0] * N
  for i in range(N):
    isum = 0
    for j in range(N):
      isum += G[i][j] * home_bias / (home_bias * pi[i] + pi[j])
      isum += G[j][i] * 1 / (home_bias * pi[j] + pi[i])
    # 架空の試合
    isum += VG * home_bias / (home_bias * pi[i] + 1.0)
    isum += VG * 1 / (home_bias * 1.0 + pi[i])
    npi[i] = (w_home[i] + w_away[j] + VG) / isum

  nsum = 0
  for i in range(N):
    nsum += npi[i]
  print(nsum)
  for i in range(N):
    pi[i] = npi[i] * N / nsum