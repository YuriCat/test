
T = 400
N = 40000
G = 2, 1


import random
import math

win = [0 for _ in range(T)]

for i in range(N):
    s = [0, 0]
    for t in range(T):
        if random.random() < G[0] / T:
            s[0] += 1
        if random.random() < G[1] / T:
            s[1] += 1
        if s[0] > s[1]:
            win[t] += 1
        elif s[0] == s[1]:
            win[t] += 0.5

import matplotlib.pyplot as plt

plt.plot([w / N for w in win])
plt.ylim(0, 1)
plt.show()

def rating_diff(w):
    return -math.log(1 / w - 1)

plt.plot([rating_diff(w / N) for w in win])
plt.show()
