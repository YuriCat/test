
# 方策勾配法のstatsはどういう値になるか？？

import numpy as np


A = 10
N = 1000000
loss_sum = 0

loss_sum = 0
for _ in range(N):
    policy = np.random.dirichlet([0.1] * A)
    adv = np.random.randn()
    #print((policy * 1000).astype(int))
    #print(adv)
    loss_sum += -np.log(policy[0]) * adv

print(loss_sum / N)

loss = 0
adv_sum = 0

for _ in range(N):
    policy = np.random.dirichlet([0.1] * A)
    action = np.random.choice(np.arange(A), p=policy)
    #print(policy, action)
    #adv = np.random.randn() + np.log(policy[action])
    q = np.log(policy)
    v = (policy * q).mean()
    adv = q[action] + np.random.randn() - v
    loss_sum += -np.log(policy[action]) * adv

print(loss / N, adv_sum / N)