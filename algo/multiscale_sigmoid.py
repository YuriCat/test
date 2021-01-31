
import numpy as np

def mssigm(x, a):
    return 2 / (1 + np.exp(-x / a)) - 1


x = np.arange(0, 10, 0.5).reshape((-1, 1))
a = np.array([[1, 4]])

v = mssigm(x, a)
xs, ys = v[:,0], v[:,1]


import matplotlib.pyplot as plt

plt.scatter(xs, ys)
plt.show()


plt.scatter(x, v.sum(-1))
plt.show()
