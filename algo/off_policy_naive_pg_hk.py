
R = [
    [ 0,  2, -1],
    [-2,  0,  1],
    [ 1, -1,  0],
]

B = 100000
T = 3000000
GAME = 'hk'

games = {
    'rps': (1, 3),
    'hk': (2, 6),
}
shape = games[GAME]

# nash equiv. of HK
Q = [1]
for i in range(shape[-1] - 1):
    Q.append(sum(Q) / (shape[-1] - 1))
Q = [q / sum(Q) for q in Q]
if GAME == 'hk':
    print('D : ', list(reversed(Q)))
    print('C : ', Q)

import random
import torch
import torch.nn as nn
import torch.nn.functional as F

logit = torch.randn(*shape, requires_grad=True)
logit_regs = [torch.zeros(shape).detach()] * 32
pi_history = []

replay_buffer = []
v = torch.zeros(2, 1).detach()
v_sq = torch.ones(2, 1).detach() 

def play_game(logit):
    pi = F.softmax(logit.detach(), -1)
    actions = torch.LongTensor([pi_.multinomial(num_samples=1, replacement=True) for pi_ in pi]).unsqueeze(-1)
    if GAME == 'rps':
        reward = R[actions[0]][actions[1]] 
    else:
        # D, C
        reward = (shape[-1] - 1) if actions[1] < actions[0] else (actions[1] - actions[0])
    targets = torch.FloatTensor([reward, -reward]).unsqueeze(1) 
    return pi.gather(-1, actions), actions, targets

for _ in range(B):
    episode = play_game(torch.zeros(*shape).expand(2, -1))
    replay_buffer.append(episode)


for i in range(T):
    logit_ = logit.expand(2, -1)

    episode = play_game(logit_)
    replay_buffer[random.randrange(len(replay_buffer))] = episode
    mu, actions, targets = random.choice(replay_buffer) 

    logit_ = logit.expand(2, -1)

    advantages = targets - v
    selected_b_prob = mu
    selected_t_prob = F.softmax(logit_.detach(), -1).gather(-1, actions)
    rho = selected_t_prob / selected_b_prob
    c = torch.stack([rho[1], rho[0]])
    var = (v_sq - v ** 2) ** 0.5

    loss = -F.log_softmax(logit_, -1).gather(-1, actions).mul(advantages).mul(torch.clamp(rho * c, 0, 1)).div(var + 1e-2).sum()
    #loss += 0.1 * F.kl_div(F.log_softmax(logit_, -1), F.softmax(logit_regs[-random.choice([1, 2, 3, 5, 8, 13, 21])], -1), reduction='sum')
    loss += F.log_softmax(logit_, -1).gather(-1, actions).mul(torch.clamp(selected_t_prob / mu, 1e-4, 1e4) - 1).sum()

    loss.backward()
    lr = 1e-3
    logit.data -= logit.grad * lr
    logit.grad.zero_()

    v = lr * targets + (1 - lr) * v
    v_sq = lr * (targets ** 2) + (1 - lr) * v_sq

    if i % 1000 == 0:
        logit_reg_ = logit.detach().clone()
        logit_regs = logit_regs[1:] + [logit_reg_]
        pi_reg = F.softmax(logit_reg_, -1)
        print(pi_reg)
        pi_history.append(pi_reg.numpy())
        print(v)
        print(var)

import numpy as np
pi_history = np.array(pi_history)

import matplotlib
import japanize_matplotlib
import matplotlib.pyplot as plt

if GAME == 'rps':
    plt.plot(pi_history[:, 0, 0])
    plt.plot(pi_history[:, 0, 1])
    plt.plot(pi_history[:, 0, 2])
    plt.legend(['グー', 'チョキ', 'パー'])    
else:
    for i in range(pi_history.shape[-1]):
        plt.plot(pi_history[:, 0, i])
    plt.legend(['drop' + str(i) for i in range(pi_history.shape[-1] - 1)] + ['dropなし'])
plt.show()

