
import numpy as np

T = 1000 # ステップ数
K = 3    # 候補数
sigma = 0.1
delta = 0.2

#pi = np.array([0.7, 0.2, 0.1])
pi = np.array([1e-2, 0.2, 0.9])
v = np.array([0.7, 0.5, 0.3])

def sample(mean):
    return 1 if np.random.rand() < mean else 0


n = np.zeros(K)
alpha, beta = np.ones(K) * 2, np.ones(K) * 2

pi_rejection = pi / pi.max()

def rejection_thompson():
    # 棄却サンプリングによる方法
    while True:
        a = np.argmax(np.random.beta(alpha, beta))
        if np.random.random() < pi_rejection[a]:
            return a

def rejection_thompson2():
    # 棄却サンプリングによる方法2
    M = np.max(pi) / np.min(pi)
    while True:
        a = np.argmax(np.random.beta(alpha, beta))
        #print(1)
        if np.random.random() * M < pi_rejection[a]:
            return a

cumulative_pi = np.stack([0, *np.cumsum(pi)])
print(cumulative_pi)

def binary_thompson():
    # 2分サンプリングによる方法
    import bisect
    st, ed = 0, cumulative_pi[-1]
    st_idx, ed_idx = 0, len(pi) - 1
    while st_idx != ed_idx:
        mid = (st + ed) / 2
        mid_idx = bisect.bisect(cumulative_pi, mid) - 1
        pos1 = mid + (st - mid) * np.random.random()
        pos2 = mid + (ed - mid) * np.random.random()
        idx1 = bisect.bisect(cumulative_pi, pos1) - 1
        idx2 = bisect.bisect(cumulative_pi, pos2) - 1
        al = [alpha[idx1], alpha[idx2]]
        be = [beta[idx1], beta[idx2]]
        r = np.random.beta(al, be)
        if r[0] >= r[1]:
            ed, ed_idx = mid, mid_idx
        else:
            st, st_idx = mid, mid_idx
    
    return st_idx


for t in range(T):
    #a = rejection_thompson()
    a = binary_thompson()

    r = sample(v[a])
    if r < 0.5: beta[a] += 1
    else: alpha[a] += 1

    n[a] += 1
    print(n)
