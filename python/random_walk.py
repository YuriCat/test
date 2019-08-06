import numpy as np

import numpy as np
M, T = 10000, 10000
x, y = np.zeros(M), np.empty(T)
for t in range(T):
 x += np.random.randint(0, 2, M)
 y[t] = np.linalg.norm(x - t / 2)
y *= 2 / (M ** 0.5)

'''
r = np.random.choice([-1, 1], (M, T))
p = np.cumsum(r, axis=1)
y = np.linalg.norm(p, axis=0) / (M ** 0.5)'''

'''a = np.where(rand(10000,10000)<0.5 , 1, -1)
a = a.cumsum(axis=1)
a = np.power(a,2)
a = np.sum(a, axis=0)
a = np.sqrt(a)'''

from matplotlib import pyplot as plt

a = np.arange(10000)
plt.plot(a, y)
plt.plot(a, a ** 0.5)
plt.savefig('tmp.png')
