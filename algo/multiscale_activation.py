
import numpy as np
import bisect

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def mstanh(x, a):
    return 2 * sigmoid(x / a) - 1

def msinvsigm(x, a):
    return 2 * sigmoid(-a / x)

def msinvexp(x, a):
    return np.exp(-a / x)

x = np.arange(0, 32, 0.1).reshape((-1, 1))
a = np.array([[1, 2, 4, 8]])

v = mstanh(x, a)
w = msinvsigm(x, a)
z = msinvexp(x, a)

import matplotlib.pyplot as plt

for i in range(a.shape[1]):
   plt.scatter(x, v[:,i])
for i in range(a.shape[1]):
   plt.scatter(x, w[:,i])
for i in range(a.shape[1]):
   plt.scatter(x, z[:,i])
plt.show()

plt.scatter(z[:,0], z[:,2])
plt.show()
