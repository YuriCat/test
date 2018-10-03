# 焼きなまし
import numpy as np

def loss_func(x):
  return x ** 4 - x ** 2

x = np.random.randn()
loss = loss_func(x)
best_x, best_loss = x, loss
for t in range(1000):
  lr = 1.0 / (10 + t)
  # 次の状態を選ぶ
  diff = np.random.randn()
  next_x = x + lr * diff
  next_loss = loss_func(next_x)
  if next_loss < loss or np.random.random() < 0.1:
    x, loss = next_x, next_loss
    if loss < best_loss:
      best_x, best_loss = x, loss
  print(best_x, best_loss)
