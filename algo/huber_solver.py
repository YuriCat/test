# Huber Lossの最小化問題をニュートン・ラプソン法で解く
import numpy as np

# 最適化したい関数全体
epsilon = 1
def huber(x, p = 0.5):
  k = p / (1 - p) if x < 0 else (1 - p) / p
  return (x ** 2) / (2 * epsilon) if np.abs(x) <= epsilon * k else k * (np.abs(x) - epsilon * k / 2)
def huber1(x, p = 0.5):
  k = p / (1 - p) if x < 0 else (1 - p) / p
  return x / epsilon if np.abs(x) <= epsilon * k else k * np.sign(x)
def huber2(x, p = 0.5):
  k = p / (1 - p) if x < 0 else (1 - p) / p
  return 1 / epsilon if np.abs(x) < epsilon * k else 0

def f(x, values, p = 0.5):
  loss = 0
  for val in values:
    loss += huber(x - val, p)
  return loss
def f1(x, values, p = 0.5):
  loss = 0
  for val in values:
    loss += huber1(x - val, p)
  return loss
def f2(x, values, p = 0.5):
  loss = 0
  for val in values:
    loss += huber2(x - val, p)
  return loss

def gd(x0, values, epochs = 20):
  print(x0)
  x = x0
  for e in range(epochs):
    d = f1(x, values)
    x = x - d * 0.5 / (1 + e * 0.1)
    print(x, f(x, values))

def newton(x0, values, p = 0.5, epochs = 10):
  print(x0)
  x = x0
  for _ in range(epochs):
    v0, v1 = f1(x, values, p), f2(x, values, p)
    print(v0, v1)
    if np.abs(v0) < 1e-4:
      break
    x = x - v0 / v1
    print(x)

def local_search(x0, values, epochs = 10):
  x = x0
  for _ in range(epochs):
    within = []
    for val in values:
      if np.abs(x - val) < epsilon:
        within.append(val)
    print('near points = ', within)

N = 100
values = sorted(np.random.randn(N) * 10)

from matplotlib import pyplot as plt
plt.plot([f1(x / 10.0, [-1, 0, 0.5, 0.5, 0.75]) for x in range(-20, 21, 1)])
#plt.plot([f(x / 10.0, [0], 0.8) for x in range(-20, 21, 1)])
plt.show()

print(values)
p = 0.98
mean, quantile = np.mean(values), np.quantile(values, p)
print('mean = ', mean, f(mean, values, p))
print('quantile = ', quantile, f(quantile, values, p))

#local_search(median, values)
newton(quantile, values, p)
#gd(median, values)
