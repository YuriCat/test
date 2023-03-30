
R = [
    [ 0,  2, -1],
    [-2,  0,  1],
    [ 1, -1,  0],
]

B = 30000
T = 300000
lambda_ = 0.8
GAME = 'hk'

games = {
    'rps': ((1, 1, 3), 1),
    'hk': ((5, 2, 2), 2 + 1),
}
shape, num_obses = games[GAME]

# nash equiv. of HK
Q = [1]
for i in range(shape[0]):
    Q.append(sum(Q) / shape[0])
Q = [q / sum(Q) for q in Q]
if GAME == 'hk':
    print('D : ', Q)
    print('C : ', list(reversed(Q)))

import random
import torch
import torch.nn as nn
import torch.nn.functional as F

logits = torch.randn(*shape, requires_grad=True)
logits = torch.zeros(shape, requires_grad=True)
logits_regs = [torch.zeros(shape).detach()] * 32
pi_history = []

replay_buffer = []
vs = torch.zeros(shape[0], num_obses, 1, requires_grad=True)
v = torch.zeros(2, shape[-1]).detach()
v_sq = torch.ones(2, shape[-1]).detach()


def step(state, logit):
    pi = F.softmax(logit.detach(), -1)
    actions = torch.LongTensor([pi_.multinomial(num_samples=1, replacement=True) for pi_ in pi]).unsqueeze(-1)
    if GAME == 'rps':
        obs = torch.LongTensor([[0], [0]])
        r = R[actions[0]][actions[1]]
        terminal = True
    else:
        # D, C
        t, dropped = state
        obs = torch.LongTensor([[1 if dropped else 0], [2]])
        dropped = 1 if actions[0] == 1 else dropped
        state = t + 1, dropped
        r = 0
        if actions[1] == 0 and dropped:
            r = 1
        if actions[1] == 1 and not dropped:
            r = shape[0]
        terminal = actions[1] == 1 or t >= shape[0]
    rewards = torch.FloatTensor([r, -r]).unsqueeze(-1) 
    return obs, pi.gather(-1, actions), actions, rewards, state, terminal

def play_game(logits):
    obses, probs, actions, rewards = [], [], [], []
    state = 0, 0
    for logit in logits:
        obs, prob, action, reward, state, terminal = step(state, logit.expand(2, -1))
        obses.append(obs)
        probs.append(prob)
        actions.append(action)
        rewards.append(reward)
        if terminal:
            break
    return torch.stack(obses), torch.stack(probs), torch.stack(actions), torch.stack(rewards)


for _ in range(B):
    episode = play_game(torch.zeros(*shape).expand(-1, 2, -1))
    replay_buffer.append(episode)


for i in range(T):
    episode = play_game(logits)

    replay_buffer[random.randrange(len(replay_buffer))] = episode
    obses, b_probs, actions, rewards = random.choice(replay_buffer)

    mask = F.pad(torch.ones(len(actions), 1, 1), [0, 0, 0, 0, 0, shape[0] - len(actions)])
    obses = F.pad(obses, [0, 0, 0, 0, 0, shape[0] - len(obses)])
    b_probs = F.pad(b_probs, [0, 0, 0, 0, 0, shape[0] - len(b_probs)], value=1)
    actions = F.pad(actions, [0, 0, 0, 0, 0, shape[0] - len(actions)])
    rewards = F.pad(rewards, [0, 0, 0, 0, 0, shape[0] - len(rewards)])

    logits_ = logits.expand(-1, 2, -1)
    t_probs = F.softmax(logits_.detach(), -1).gather(-1, actions)
    rhos = t_probs / b_probs
    cs = rhos.prod(1, keepdim=True)

    targets_ = [torch.zeros(2, 1)]
    vs_ = vs.gather(1, obses)
    vs_nograd = vs_.detach() * mask
    for t in range(rewards.shape[0] - 1, -1, -1):
        lamb_ = torch.clamp(cs[t] * lambda_, 0, 1)
        targets_ = [(1 - lamb_) * vs_nograd[t] + lamb_ * (rewards[t] + targets_[0])] + targets_
    targets = torch.stack(targets_[:-1])
    
    advantages = torch.clamp(cs[0], 1) * (rewards + torch.stack(targets_[1:]) - vs_nograd)
    #v_targets = advantages + vs_nograd
    v_targets = rewards + torch.stack(targets_[1:])

    vars = torch.clamp(v_sq - v ** 2 + 1, 1e-2, 1e2) ** 0.5

    loss = -F.log_softmax(logits_, -1).gather(-1, actions).mul(advantages).div(vars.unsqueeze(0)).mul(mask).sum()
    #loss += 0.1 * F.kl_div(F.log_softmax(logit_, -1), F.softmax(logit_regs[-random.choice([1, 2, 3, 5, 8, 13, 21])], -1), reduction='sum')
    loss += F.log_softmax(logits_, -1).gather(-1, actions).mul(torch.clamp(rhos, 1e-4, 1e4) - 1).mul(mask).sum()

    loss += (vs_ - v_targets).pow(2).mul(mask).sum().mul(2.5)

    loss.backward()
    lr = 1e-3
    logits.data -= torch.clamp(logits.grad, -10, 10) * lr
    logits.grad.zero_()
    vs.data -= vs.grad * lr
    vs.grad.zero_()

    v = lr * rewards.sum(0) + (1 - lr) * v
    v_sq = lr * (rewards ** 2).sum(0) + (1 - lr) * v_sq

    if i % 1000 == 0:
        logits_reg_ = logits.detach().clone()
        logits_regs = logits_regs[1:] + [logits_reg_]
        pi_reg = F.softmax(logits_reg_, -1)
        print('pi =', pi_reg)
        pi_history.append(pi_reg.numpy())
        print('vs =', vs[:, :, 0].detach())
        print('v =', v[:, 0])
        print('var =', vars[:, 0])
        print('obses =', obses.squeeze(-1))
        print('actions =', actions.squeeze(-1))
        print('v_targets =', v_targets.squeeze(-1))
        print('baselines =', vs_nograd.squeeze(-1))
        print('advantages =', advantages.squeeze(-1))

import numpy as np
pi_history = np.array(pi_history)

import matplotlib
import japanize_matplotlib
import matplotlib.pyplot as plt

if GAME == 'rps':
    plt.plot(pi_history[:, 0, 0, 0])
    plt.plot(pi_history[:, 0, 0, 1])
    plt.plot(pi_history[:, 0, 0, 2])
    plt.legend(['グー', 'チョキ', 'パー'])    
else:
    for i in range(pi_history.shape[1]):
        plt.plot(pi_history[:, i, 1, 1])
    plt.legend(['check t=' + str(i) for i in range(pi_history.shape[1])])
plt.show()

