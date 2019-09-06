import numpy as np
import matplotlib.pyplot as plt

p = np.array([0.1, 0.2, 0.7])
#p = np.array([0.001, 0.009, 0.99])
#p = np.array([0.3, 0.3, 0.4])
#p = np.array([0.7, 0.2, 0.1])
#p = np.array([0.99, 0.009, 0.001])
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

real = np.array([0.7, 0.5, 0.3])
real_real = np.array([0.5, 0.3, 0.9])

v = 0.5
q_est = v + p2a(p)

mode = 'B'

PRIOR = 8
N_PRIOR = 1
process = 4

print('num process = %d' % process)

def sample(i, n = 0):
    # 平均値をずらす
    #coef = np.tanh(n / 16)
    coef = 0#int(n > 64)
    mean = real[i] * (1 - coef) + real_real[i] * coef

    # サンプリング
    if mode == 'B':
        return 1 if np.random.random() < mean else 0
    else:
        return mean + np.random.randn() * 0.3

def noise():
    return np.random.random(len(p)) * 1e-6

class BanditBase:
    def __init__(self, p, v):
        self.n = np.ones_like(p) * N_PRIOR / len(p)
        self.p, self.v = p, v
        self.att = 1#0.97

        if mode == 'B':
            self.a, self.b = np.ones_like(p) * PRIOR / 2, np.ones_like(p) * PRIOR / 2
        else:
            mu_mean0, mu_std0 = 0, 1 # muの事前分布(正規分布)
            sigma_mean0, sigma_std0 = 1, 2 # sigmaの事前分布(逆ガンマ分布)

            self.mu = mu_mean0
            self.kappa = (1 / mu_std0) ** 2
            self.alpha = (sigma_mean0 / sigma_std0) ** 2 + 2
            self.beta = (self.alpha - 1) * sigma_mean0

    def update(self, action, reward):
        self.n[action] += 1
        if mode == 'B':
            self.a *= self.att
            self.b *= self.att
            if reward > 0.5:
                self.a[action] += 1
            else:
                self.b[action] += 1
        else:
            self.alpha = self.alpha + 1 / 2
            self.beta = self.beta + self.kappa / (self.kappa + 1) * ((reward - self.mu) ** 2) / 2
            self.mu = (self.kappa * self.mu + reward) / (self.kappa + 1)
            self.kappa += 1

    def prior(self):
        prob = p ** (self.att ** self.n.sum())
        return prob / prob.sum()

    def mean(self):
        if mode == 'B':
            return self.a / (self.a + self.b)
        else:
            return self.mu

    def variance(self):
        if mode == 'B':
            return self.a * self.b / ((self.a + self.b) ** 2) / (self.a + self.b + 1)
        else:
            return self.q2_sum / self.n - (self.mean() ** 2) / self.n

    def sample(self):
        if mode == 'B':
            return np.random.beta(self.a, self.b)
        else:
            return self.mean() + np.random.randn(len(p)) * ((self.variance() / self.n) ** 0.5)

class UCB1(BanditBase):
    def bandit(self):
        ucb1 = self.mean() + np.sqrt(2 * np.log(self.n.sum() + 1) / self.n)
        return np.argmax(ucb1 + noise())

class PUCB(BanditBase):
    def bandit(self):
        pucb = self.mean() + self.prior() * np.sqrt(2.0 * self.n.sum()) / (self.n + 1)
        #pucb = self.mean() + self.p * np.sqrt(2.0) / (self.n + 1)
        return np.argmax(pucb)

class UCBRoot(BanditBase):
    def bandit(self):
        ucbr = self.mean() + np.sqrt(2 * np.sqrt(self.n.sum()) / self.n)
        return np.argmax(ucbr + noise())

class Thompson(BanditBase):
    def bandit(self):
        return np.argmax(self.sample())

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

class BiasedThompson(Thompson):
    def bandit(self):
        psum, ba = 0, None
        prior = self.prior()
        if self.n.sum() == N_PRIOR:
            return np.random.choice(np.arange(len(p)), p=prior)
        prior /= np.max(prior)
        for _ in range(32):
            a1 = super().bandit()
            r = np.random.random(2)
            if r[0] < prior[a1]:
                return a1
            if r[1] * (psum + prior[a1]) >= psum:
                ba = a1
            psum += prior[a1]
        return ba

def test(algo, steps):
    reward_sum, rewards = 0, []
    for i in range(steps):
        action = algo.bandit()
        reward = sample(action, algo.n[action])
        algo.update(action, reward)
        reward_sum += reward
        rewards.append(reward_sum / (i + 1))
    return rewards

def mtest_(args):
    Algo, idx, n, steps = args
    np.random.seed(idx)
    result = np.zeros(steps)
    for _ in range(n):
        algo = Algo(p, v)
        result += test(algo, steps)
    return result

def mtest(Algo, steps=1024):
    n = 256
    import multiprocessing as mp
    with mp.Pool(process) as p:
        results = p.map(mtest_, [(Algo, i, n//process, steps) for i in range(process)])

    return np.stack(results).sum(axis=0) / n


plt.plot(mtest(UCB1), label='ucb1')
plt.plot(mtest(PUCB), label='pucb')
plt.plot(mtest(Thompson), label='thompson')
plt.plot(mtest(PriorThompson), label='pthompson')
plt.plot(mtest(BiasedThompson), label='bthompson')
plt.plot(mtest(UCBRoot), label='ucbroot')
plt.legend()
plt.ylim(0, 1)
plt.show()