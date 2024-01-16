
R = [
    [ 0,  2, -1],
    [-2,  0,  1],
    [ 1, -1,  0],
]

T = 10000000
GAME = 'rps'

games = {
    'rps': (1, 3),
    'hk': (2, 11),
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

import numpy as np

states = [(None, None), (None, 0), (None, 1), (None, 2)]
vs = {s: 0.0 for s in states}
ns = {s: 0 for s in states}

regrets = [[0, 0, 0] for _ in range(2)]
pi_sum = np.ones((2, 3))
pi_history = []



def softmax(pi, t=1.0):
    m = np.max(pi)
    pi = [np.exp((p - m) / t) for p in pi]
    s = np.sum(pi)
    return [p / s for p in pi]

def mc(player, state):
    if None not in state:
        return R[state[0]][state[1]]
    regret = regrets[player]
    rpositive = [np.maximum(r, 0) for r in regret]
    rp_sum = sum(rpositive)
    if rp_sum > 0:
        pi = [rp_ / rp_sum for rp_ in rpositive]
    else:
        pi = [1 / len(regret) for _ in regret]

    action = np.random.choice(list(range(3)), p=pi)
    reward = -mc(1 - player, (state[1], action))

    def compute_value(st):
        if None not in st:
            return R[st[0]][st[1]]
        return vs[st] / ns[st] if ns[st] > 0 else 0

    q = [-compute_value((state[1], action)) for action in range(3)]
    vs[state] = 0
    for action in range(3):
        vs[state] += pi[action] * q[action]
    ns[state] = 1
    v = vs[state] / ns[state] if ns[state] > 0 else 0
    regrets[player][action] += reward - v
    pi_sum[player] += np.array(pi)

    return reward

def cfr(player, state, likelihood):
    if None not in state:
        return R[state[0]][state[1]]
    regret = regrets[player]
    rpositive = [np.maximum(r, 0) for r in regret]
    rp_sum = sum(rpositive)
    if rp_sum > 0:
        pi = [rp_ / rp_sum for rp_ in rpositive]
    else:
        pi = [1 / len(regret) for _ in regret]

    q = []
    for action in range(3):
        q.append(-cfr(1 - player, (state[1], action), likelihood * pi[action]))

    v = vs[state] / ns[state] if ns[state] > 0 else 0
    vs[state] = 0
    for action in range(3):
        regrets[player][action] += likelihood * (q[action] - v)
        vs[state] += pi[action] * q[action]
    ns[state] = 1
    pi_sum[player] += np.array(pi)

    return vs[state] / ns[state]

def cfrplus(player, state, likelihood):
    if None not in state:
        return R[state[0]][state[1]]
    regret = regrets[player]
    rpositive = [np.maximum(r, 0) for r in regret]
    rp_sum = sum(rpositive)
    if rp_sum > 0:
        pi = [rp_ / rp_sum for rp_ in rpositive]
    else:
        pi = [1 / len(regret) for _ in regret]

    q = []
    for action in range(3):
        q.append(-cfrplus(1 - player, (state[1], action), likelihood * pi[action]))

    v = vs[state] / ns[state] if ns[state] > 0 else 0
    vs[state] = 0
    for action in range(3):
        regrets[player][action] = np.maximum(0, regrets[player][action] + likelihood * (q[action] - v))
        vs[state] += pi[action] * q[action]
    ns[state] = 1
    pi_sum[player] += np.array(pi)

    return vs[state] / ns[state]

def tablecfr(player, state, likelihood):
    pis = {s: None for s in vs.keys()}
    likelihoods = {s: 0 for s in vs.keys()}

    def compute_likelihood(pl, st, lh):
        if None not in st:
            return
        likelihoods[st] = lh
        regret = regrets[pl]
        rpositive = [np.maximum(r, 0) for r in regret]
        rp_sum = sum(rpositive)
        if rp_sum > 0:
            pi = [rp_ / rp_sum for rp_ in rpositive]
        else:
            pi = [1 / len(regret) for _ in regret]
        pis[st] = pi
        for action in range(3):
            compute_likelihood(1 - pl, (st[1], action), lh * pi[action])

    def compute_value(st):
        if None not in st:
            return R[st[0]][st[1]]
        return vs[st] / ns[st] if ns[st] > 0 else 0

    compute_likelihood(player, state, likelihood)
    for state in states:
        player = 0 if state[1] is None else 1
        q = [-compute_value((state[1], action)) for action in range(3)]
        pi = pis[state]
        likelihood = likelihoods[state]
        v = vs[state] / ns[state] if ns[state] > 0 else 0
        vs[state] = 0
        for action in range(3):
            regrets[player][action] += likelihood * (q[action] - v)
            vs[state] += pi[action] * q[action]
        ns[state] = 1
        pi_sum[player] += np.array(pi)

def quantized():
    N = 5
    policies = {0: np.zeros((N, N, N, 3)), 1: np.zeros((N, N, N, 3))}
    values = {0: np.zeros((N, N, N)), 1: np.zeros((N, N, N))}

    def get_value(vtable, likelihood):
        lindex = [np.minimum(int(lh * (N - 1)), N - 2) for lh in likelihood]
        lfrac = [lindex[i_] + 1 - lh * (N - 1) for i_, lh in enumerate(likelihood)]
        ret = 0
        for x in range(2):
            for y in range(2):
                for z in range(2):
                    coef = np.abs(lfrac[0] - x) * np.abs(lfrac[1] - y) * np.abs(lfrac[2] - z)
                    ret += vtable[lindex[0] + x, lindex[1] + y, lindex[2] + z] * coef
        #print(vtable[lindex[0], lindex[1], lindex[2]], ret)
        #input()
        return ret

    for player in [1, 0]:
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    p = [i / N, j / N, k / N]
                    p = [(p_ / sum(p) if sum(p) > 0 else 1.0 / 3) for p_ in p]

                    if player == 1:
                        q = [np.sum([p[a] * R[action][a] for a in range(3)]) for action in range(3)]
                        values[player][i, j, k] = max(q)
                    else:
                        max_value = -1000
                        best_policy = None
                        for ii in range(N):
                            for jj in range(N):
                                for kk in range(N):
                                    policy = [ii / N, jj / N, kk / N]
                                    policy = [(policy_ / sum(policy) if sum(policy) else 1.0 / 3) for policy_ in policy]

                                    #likelihood = [p[a] * policy[a] for a in range(3)]
                                    likelihood = [1 * policy[a] for a in range(3)]
                                    likelihood = [lh / sum(likelihood) for lh in likelihood]
                                    v = -get_value(values[player + 1], likelihood)
                                    if v > max_value:
                                        max_value = v
                                        best_policy = policy
                        values[player][i, j, k] = max_value
                        policies[player][i, j, k] = np.array(best_policy)

    print(values[0][0, 0, 0])
    print(policies[0][0, 0, 0])



for t in range(T):
    #mccfr(0, (None, None))
    #cfr(0, (None, None), 1)
    #tablecfr(0, (None, None), 1.0)
    cfrplus(0, (None, None), 1)

    #quantized()

    if t % 10000 == 0:
        pi_mean = pi_sum / pi_sum.sum(1, keepdims=True)
        pi_history.append(pi_mean)
        print(vs)
        print(pi_mean)
        print(regrets)
        #input()

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
