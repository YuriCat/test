# じゃんけんの逆問題
import math
import numpy as np
np.random.seed(0)

K = 3 # 役の数 >= 3
M = 2 # プレーヤー数 >= 2

def softmax(x):
  p = np.exp(x)
  if len(p.shape) == 1:
    return p / np.sum(p)
  else:
    return np.array([ip / np.sum(ip) for ip in p])
def random_theta():
  theta = np.random.randn(M, K)
  return theta - np.mean(theta)

answer_theta = random_theta()
answer_pi = softmax(answer_theta)

def pi2w(v0, v1):
  w, d = 0, 0
  for i in range(K):
    w += v0[i] * v1[(i + 1) % K]
    d += v0[i] * v1[i]
  return w + d * 0.5
  #A = np.eye(K, K) * 0.5 + np.vstack([np.eye(K, K)[1:,:], np.eye(K, K)[:1,:]])
  #return np.sum(A * (np.reshape(v0, [3, 1]) * np.reshape(v1, [1, 3])))
  
def pi2dwdpi(pi0, pi1):
  # (微分変数)
  return np.array([pi1[(k + 1) % K] + pi1[k] * 0.5 for k in range(K)])
def pi2dpidt(pi0):
  # (母変数, 微分変数)
  return pi0.reshape((K, 1)) * (np.eye(K, K) - pi0) # softmaxの微分 y_kk (I - y_k)
def pis2w(pi):
  return np.array([[pi2w(pi[i], pi[j]) for j in range(M)] for i in range(M)])
def pis2dwdpi(pi):
  # (プレーヤー, プレーヤー, 微分変数)
  return np.array([[pi2dwdpi(pi[i], pi[j]) for j in range(M)] for i in range(M)])
def pis2dpidt(pi):
  return np.array([pi2dpidt(pi[i]) for i in range(M)])

N, X = np.zeros((M, M)), np.zeros((M, M))
R = []

for i in range(100000):
  p0index, p1index = np.random.randint(M), np.random.randint(M)
  p0, p1 = answer_pi[p0index], answer_pi[p1index]
  wp = pi2w(p0, p1)
  result = 1 if np.random.rand() < wp else 0
  R.append((p0index, p1index, result))
  if result == 1:
    X[p0index][p1index] += 1
  else:
    X[p1index][p0index] += 1
  N[p0index][p1index] += 1

print("theoretical wp = ")
print(pis2w(answer_pi))
print("real wp = ")
print(X / (N + N.T))

# 逆推定

def w2loss(w):
  return (-X * np.log(w)).sum() / np.sum(N)
print(w2loss(pis2w(answer_pi)))

def gd(theta):
  lr = 1e-3 / np.sqrt(np.mean(N))
  theta_decay = 3e-5
  best_theta, best_loss = theta, float('inf')
  for _ in range(100000):
    pi = softmax(theta)
    w = pis2w(pi)
    dwdpi = pis2dwdpi(pi)
    dpidt = pis2dpidt(pi)
    c = -X / w
    cc = c.reshape((M, M, 1))
    loss = w2loss(w)
    print(loss)
    if loss < best_loss:
      best_theta, best_loss = theta, loss
    grad = (cc * dwdpi).reshape((M, M, K, 1)) * dpidt.reshape((M, 1, K, K))
    next_theta = theta * (1 - theta_decay) - lr * grad.sum(axis=1).sum(axis=1)
    # 拘束条件
    next_theta -= np.mean(next_theta, axis=1).reshape((M, 1))
    if float('nan') in next_theta:
      print('NaN!!!')
      break
    # 更新
    theta = next_theta
  print("best loss = ", best_loss)
  return best_theta

def sgd(theta):
  lr = 1e-6
  theta_decay = 3e-6
  best_theta, best_loss = theta, float('inf')
  for _ in range(10000):
    p0index, p1index, result = R[np.random.randint(len(R))]
    players = np.array([p0index, p1index])
    theta2 = theta[players,:]
    pi = softmax(theta2)
    w = pis2w(pi)
    dwdpi, dpidt = pis2dwdpi(pi), pis2dpidt(pi)
    x = np.array([result, 1 - result])
    c = (-X / w).reshape((M, M, 1))
    loss = w2loss(w)
    print(loss)
    if loss < best_loss:
      best_theta, best_loss = theta, loss
    grad = ((c * dwdpi).reshape((2, 2, K, 1)) * dpidt.reshape((2, 1, K, K))).sum(axis=1).sum(axis=1)
    for i, index in enumerate(players):
      next_theta = theta[index] * (1 - theta_decay) - lr * grad[i]
      theta[index] = next_theta - next_theta.mean()
  return best_theta

def gd_chainer():
  import chainer
  import chainer.functions as F

  class Rating(chainer.Chain):
    def __init__(self, K, M):
      super().__init__()
      # 定数
      self.K, self.M = K, M
      self.A = np.eye(K, K) * 0.5 + np.vstack([np.eye(K, K)[1:,:], np.eye(K, K)[:1,:]])
      with self.init_scope():
        self.theta = chainer.Parameter(np.random.randn(M, K).astype(np.float32))
    def __call__(self):
      pi = F.softmax(self.theta, axis=-1)
      print(self.theta)
      print(pi)
      w = []
      for i in range(self.M):
        wi = []
        for j in range(self.M):
          pdot = F.matmul(F.reshape(pi[i], [self.K, 1]), F.reshape(pi[j], [1, self.K]))
          wij = F.sum(self.A * pdot)
          wi.append(wij)
        wi = F.stack(wi)
        w.append(wi)
      w = F.reshape(F.stack(w), [1, self.M, self.M])

      loss_mat = X * F.log(w) - (N - X) * F.log(1 - w)
      print("loss mat = ", loss_mat)
      loss = -F.sum(loss_mat)
      print("loss scalar = ", loss)
      return loss

  model = Rating(K, M)
  optimizer = chainer.optimizers.SGD().setup(model)
  optimizer.add_hook(chainer.optimizer.WeightDecay(1e-2))

  for i in range(100):
    loss = model()
    model.cleargrads()
    loss.backward()
    optimizer.update()

  print("answer theta = ")
  print(players)
  print("real wp = ")
  print(X / N)
  print("experimental wp = ")
  print(pis2w(pi))
  return players


def annealing():
  x = theta
  l2_coef = 0.1
  lr = 0.001
  temp = 1

  print(softmax(x))
  loss = w2loss(pis2w(softmax(x))) + l2_coef * (x ** 2).sum()
  best_x, best_loss = x, loss
  for t in range(100000):
    # 次の状態を選ぶ
    diff = np.random.randn(M, K)
    diff = np.array([d - np.mean(d) for d in diff])
    next_x = x + diff * lr
    next_loss = w2loss(pis2w(softmax(x))) + l2_coef * (x ** 2).sum()
    prob = 1 if next_loss < loss else np.exp((loss - next_loss) / temp)
    if np.random.random() < prob:
      x, loss = next_x, next_loss
      if loss < best_loss:
        best_x, best_loss = x, loss
    print(best_loss, loss)

#best_theta = gd(random_theta())
best_theta = sgd(random_theta())

print("best pi = ")
print(softmax(best_theta))
print("answer pi =")
print(answer_pi)
print("best wp = ")
print(pis2w(softmax(best_theta)))
print("real wp = ")
print(X / (N + N.T))