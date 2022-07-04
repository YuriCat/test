
R = [
    [ 0,  1, -1],
    [-1,  0,  1],
    [ 1, -1,  0],
]

import torch
import torch.nn as nn
import torch.nn.functional as F

log_pi = torch.randn(1, 3, requires_grad=True)
#log_pi = torch.randn(2, 3, requires_grad=True)
log_pi_reg = log_pi.detach().clone()
pi_history = []

for i in range(300000):
    log_pi_ = log_pi.detach()
    if log_pi_.size(0) == 1:
        log_pi_ = log_pi_.repeat(2, 1)
    pi = F.softmax(log_pi_.detach(), -1)
    actions = torch.LongTensor([pi_.multinomial(num_samples=1, replacement=True) for pi_ in pi]).unsqueeze(-1)
    reward = R[actions[0]][actions[1]]
    targets = torch.FloatTensor([reward, -reward]).unsqueeze(1)

    log_pi_ratio_reg = log_pi_ - log_pi_reg
    selected_b_prob = pi.gather(-1, actions)
    eta = 1

    advantages = -eta * log_pi_ratio_reg + F.one_hot(actions, 3) / selected_b_prob * targets

    clip = 1e4
    loss = -log_pi * torch.clamp(advantages, -clip, clip)

    loss.sum().backward()
    nn.utils.clip_grad_norm_(log_pi, 4.0)
    lr = 1e-3
    log_pi.data -= log_pi.grad * lr
    log_pi.grad.zero_()

    if i % 1000 == 0:
        log_pi_reg = log_pi.detach().clone()
        pi_reg = F.softmax(log_pi_reg, -1)[0]
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

