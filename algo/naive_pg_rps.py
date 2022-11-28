
R = [
    [ 0,  2, -1],
    [-2,  0,  1],
    [ 1, -1,  0],
]

import torch
import torch.nn as nn
import torch.nn.functional as F

logit = torch.randn(1, 3, requires_grad=True)
logit_reg = logit.detach().clone()
pi_history = []

for i in range(300000):
    logit_ = logit
    if logit_.size(0) == 1:
        logit_ = logit_.repeat(2, 1)

    pi = F.softmax(logit_.detach(), -1)
    actions = torch.LongTensor([pi_.multinomial(num_samples=1, replacement=True) for pi_ in pi]).unsqueeze(-1)
    reward = R[actions[0]][actions[1]]
    targets = torch.FloatTensor([reward, -reward]).unsqueeze(1)

    advantages = targets
    loss = -F.log_softmax(logit_, -1).gather(-1, actions) * advantages

    loss.sum().backward()
    lr = 1e-3
    logit.data -= logit.grad * lr
    logit.grad.zero_()

    if i % 1000 == 0:
        logit_reg = logit.detach().clone()
        pi_reg = F.softmax(logit_reg, -1)[0]
        print(pi_reg)
        pi_history.append(pi_reg.numpy())


import numpy as np
pi_history = np.array(pi_history)

import matplotlib
import japanize_matplotlib
import matplotlib.pyplot as plt

plt.plot(pi_history[:,0])
plt.plot(pi_history[:,1])
plt.plot(pi_history[:,2])
plt.legend(['グー', 'チョキ', 'パー'])
plt.show()

