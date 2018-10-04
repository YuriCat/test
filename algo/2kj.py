# 2つしかないじゃんけん問題
import math
import numpy as np

np.random.seed(0)

M = 2 # プレーヤー数

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

players = np.random.randn(M)
players -= np.mean(players)
print(players)
players = sigmoid(players)
print(players)

def pi2w(pi0, pi1):
  return 0.5 * (pi0 - pi1 + 1)
def pi2dwdpi(pi0, pi1):
  return 0.5
def pi2dpidt(pi0):
  return pi0 * (1 - pi0)
def pis2w(pi):
  return np.array([[pi2w(pi[i], pi[j]) for j in range(M)] for i in range(M)])
def pis2dwdpi(pi):
  return np.array([[pi2dwdpi(pi[i], pi[j]) for j in range(M)] for i in range(M)])
def pis2dpidt(pi):
  return np.array([pi2dpidt(pi[i]) for i in range(M)])

N, X = np.zeros((M, M)), np.zeros((M, M))

for i in range(100000):
  p0index, p1index = np.random.randint(M), np.random.randint(M)
  p0, p1 = players[p0index], players[p1index]
  w = pi2w(p0, p1)
  result = 0 if np.random.rand() < w else 1
  if result == 0:
    X[p0index][p1index] += 1
  N[p0index][p1index] += 1

print(N)
print(X)

# 逆推定
theta = [0] * M
pi = [0.5] * M
for iteration in range(100):
  print(pi, theta)
  w = pis2w(pi)
  dwdpi = pis2dwdpi(pi)
  dpidt = pis2dpidt(pi)
  grad = np.sum((X / w - (N - X) / (1 - w)) * dwdpi * dpidt, axis=1)
  next_theta = theta + 0.01 / np.sqrt(np.mean(N)) * grad
  # 拘束条件
  mean = np.sum(next_theta) / M
  next_theta -= mean
  next_pi = sigmoid(next_theta)
  # 更新
  pi = next_pi
  theta = next_theta
