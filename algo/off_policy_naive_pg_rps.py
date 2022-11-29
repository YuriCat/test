
R = [
    [ 0,  2, -1],
    [-2,  0,  1],
    [ 1, -1,  0],
]

import random
import torch
import torch.nn as nn
import torch.nn.functional as F

logit = torch.randn(1, 3, requires_grad=True)
logit_regs = [torch.zeros(1, 3).detach()] * 31 + [logit.detach().clone()]
pi_history = []

replay_buffer = []
for _ in range(10000):
    pi = F.softmax(torch.zeros(2, 3).detach(), -1)
    actions = torch.LongTensor([pi_.multinomial(num_samples=1, replacement=True) for pi_ in pi]).unsqueeze(-1)
    reward = R[actions[0]][actions[1]]
    targets = torch.FloatTensor([reward, -reward]).unsqueeze(1)   
    replay_buffer.append((pi, actions, targets))


for i in range(300000):
    logit_ = logit
    if logit_.size(0) == 1:
        logit_ = logit_.repeat(2, 1)

    pi = F.softmax(logit_.detach(), -1)
    actions = torch.LongTensor([pi_.multinomial(num_samples=1, replacement=True) for pi_ in pi]).unsqueeze(-1)
    reward = R[actions[0]][actions[1]]
    targets = torch.FloatTensor([reward, -reward]).unsqueeze(1)

    replay_buffer.append((pi, actions, targets))
    replay_buffer = replay_buffer[-10000:]
    mu, actions, targets = random.choice(replay_buffer)

    advantages = targets
    selected_b_prob = mu.gather(-1, actions)
    selected_t_prob = pi.gather(-1, actions)
    rho = (selected_t_prob / selected_b_prob).prod(0, keepdim=True)

    loss = -F.log_softmax(logit_, -1).gather(-1, actions) * advantages * torch.clamp(rho, 0, 1)
    #loss += F.kl_div(F.log_softmax(logit_, -1), mu, reduction='sum')
    logit_reg = logit_regs[-random.choice([1, 2, 3, 5, 8, 13, 21])]
    loss += F.kl_div(F.log_softmax(logit_, -1), F.softmax(logit_reg, -1), reduction='sum')

    loss.sum().backward()
    lr = 1e-3
    logit.data -= logit.grad * lr
    logit.grad.zero_()

    if i % 1000 == 0:
        logit_reg_ = logit.detach().clone()
        logit_regs = logit_regs[1:] + [logit_reg_]
        pi_reg = F.softmax(logit_reg_, -1)[0]
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

