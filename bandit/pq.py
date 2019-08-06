import numpy as np
import matplotlib.pyplot as plt

#p = np.array([0.1, 0.2, 0.7])
#p = np.array([0.001, 0.009, 0.99])
#p = np.array([0.3, 0.3, 0.4])
p = np.array([0.7, 0.2, 0.1])
p = np.array([0.99, 0.009, 0.001])
print(np.sum(p))

def entropy(p):
    return -np.dot(p, np.log2(p))

def p2a(p, alpha=0.5):
    return alpha * (np.log2(p) + entropy(p))

a = p2a(p)
print(a)
print(np.dot(p, a))

# pv比率は事前分布の大きさ
# 定数alphaはどこから来る?

# aの重み付け分散
m = np.dot(p, a ** 2)
print(m)

real = np.array([0.2, 0.5, 0.8])

v = 0.5
q_est = v + p2a(p)

class BanditBase:
    def __init__(self, p, v):
        self.n = np.ones_like(p) / len(p)
        self.q_sum = v * self.n
        self.q2_sum = ((self.q_sum / self.n) ** 2 + 1 ** 2) * self.n
        self.p, self.v = p, v

    def update(self, action, reward):
        self.q_sum[action] += reward
        self.q2_sum[action] += reward ** 2
        self.n[action] += 1

class UCB1(BanditBase):
    def bandit(self):
        ucb1 = self.q_sum / self.n + np.sqrt(2 * np.log(self.n.sum()) / self.n)
        return np.argmax(ucb1)

class PUCB(BanditBase):
    def bandit(self):
        pucb = self.q_sum / self.n + self.p * np.sqrt(2.0 * self.n.sum()) / (self.n + 1)
        pucb = self.q_sum / self.n + self.p * np.sqrt(2.0) / (self.n + 1)
        return np.argmax(pucb)

class UCBRoot(BanditBase):
    def bandit(self):
        ucb1 = self.q_sum / self.n + np.sqrt(2 * np.sqrt(self.n.sum()) / self.n)
        return np.argmax(ucb1)

class Thompson(BanditBase):
    def bandit(self):
        stddev = ((self.q2_sum / self.n - (self.q_sum / self.n) ** 2) / self.n) ** 0.5
        r = self.q_sum / self.n + stddev * np.random.randn(len(self.n))
        return np.argmax(r)

class PriorThompson(BanditBase):
    def __init__(self, p, v):
        super().__init__(p, v)
        q = v + p2a(p)
        self.n = np.ones_like(p)
        self.q_est = q * self.n
        self.q2_est = ((self.q_est / self.n) ** 2 + 1 ** 2) * self.n

    def bandit(self):
        stddev = ((self.q2_est / self.n - (self.q_est / self.n) ** 2) / self.n) ** 0.5
        r = self.q_est / self.n + stddev * np.random.randn(len(self.n))
        return np.argmax(r)

    def update(self, action, reward):
        super().update(action, reward)
        self.q_est[action] += reward
        self.q2_est[action] += reward ** 2

def test(algo , steps):
    reward_sum, rewards = 0, []
    for i in range(steps):
        action = algo.bandit()
        reward = real[action] + np.random.randn() * 0.3
        algo.update(action, reward)
        reward_sum += reward
        rewards.append(reward_sum / (i + 1))
    return rewards

def mtest(Algo, steps=256):
    result = np.zeros((steps))
    n = 256
    for i in range(n):
        algo = Algo(p, v)
        result += test(algo, steps)
    return result / n


plt.plot(mtest(UCB1), label='ucb1')
plt.plot(mtest(PUCB), label='pucb')
plt.plot(mtest(Thompson), label='thompson')
plt.plot(mtest(PriorThompson), label='pthompson')
plt.plot(mtest(UCBRoot), label='ucbroot')
plt.legend()
plt.ylim(0, 1)
plt.show()