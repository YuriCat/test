# じゃんけんの逆問題
import math
import numpy as np

np.random.seed(0)

K = 3 # 役の数
M = 3 # プレーヤー数

'''players = []
while len(players) < M:
  # K - 1 次元の中の正 K 角形内に含まれるものをOKとする
  alpha = 2 * math.pi / K
  beta = (K - 1) * 2 * math.pi / K
  root3 = math.sqrt(3)

  r = np.random.random(K - 1) * 2 - 1

  a = (r[0] + 1) / 2 + r[1] / root3
  b = (r[0] + 1) / 2 - r[1] / root3

  if 0 <= a and 0 <= b and a + b <= 1:
    players.append(np.array([a, b, 1 - a - b]))
'''

def softmax(x):
  p = np.exp(x)
  if len(p.shape) == 1:
    return p / np.sum(p)
  else:
    return np.array([ip / np.sum(ip) for ip in p])

players = np.random.randn(M, K)
players -= np.mean(players)
print(players)
for i in range(M):
  players[i] = softmax(players[i])
print(players)

def pi2w(v0, v1):
  w, d = 0, 0
  for i in range(K):
    w += v0[i] * v1[(i + 1) % K]
    d += v0[i] * v1[i]
  return w + d * 0.5
  #A = np.eye(K, K) * 0.5 + np.vstack([np.eye(K, K)[1:,:], np.eye(K, K)[:1,:]])
  #return np.sum(A * (np.reshape(v0, [3, 1]) * np.reshape(v1, [1, 3])))
  
def pi2dwdpi(pi0, pi1):
  return np.array([pi1[(k + 1) % K] + pi1[k] * 0.5 for k in range(K)])
def pi2dpidt(pi0):
  return pi0 * (np.eye(K, K) - pi0) # softmaxの微分 y_kk (I - y_k)
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
  wp = pi2w(p0, p1)
  result = 0 if np.random.rand() < wp else 1
  if result == 0:
    X[p0index][p1index] += 1
  N[p0index][p1index] += 1

print("theoretical wp = ")
print(pis2w(players))
print("real wp = ")
print(X / N)


# 逆推定
theta = np.random.randn(M, K)
pi = np.array([softmax(t) for t in theta])

def w2loss(w):
  return (-X * np.log(w) - (N - X) * np.log(1 - w)).sum() / np.sum(N)
print(w2loss(pis2w(players)))
print(w2loss(pis2w(pi)))


'''for _ in range(1000):
  print(theta)
  print(pi)

  w = pis2w(pi)
  dwdpi = pis2dwdpi(pi)
  dpidt = pis2dpidt(pi)

  print(w.shape)
  print(dwdpi.shape)
  print(dpidt.shape)

  c = X / w - (N - X) / (1 - w)
  cc = np.broadcast_to(c, (K, M, M)).transpose([1, 2, 0])
  grad = np.sum(cc * dwdpi, axis=1) * np.sum(dpidt, as 

  next_theta = theta + 0.01 / np.sqrt(np.mean(N)) * grad
  # 拘束条件
  next_theta -= np.mean(next_theta)
  next_pi = np.array([softmax(nt) for nt in next_theta])
  # 更新
  pi = next_pi
  theta = next_theta'''

'''import chainer
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
print(pis2w(pi))'''

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

print("best pi = ")
print(softmax(best_x))
print("answer = pi")
print(players)
print("theoretical wp = ")
print(pis2w(softmax(best_x)))
print("real wp = ")
print(X / N)

