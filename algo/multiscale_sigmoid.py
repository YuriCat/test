
import numpy as np
import bisect

def mstanh(x, a):
    return 2 / (1 + np.exp(-x / a)) - 1

def spectram(x, a):
    sa = np.cumsum(a)
    ans = []
    for y in x:
        index = bisect.bisect_left(sa, y) - 1
        ans.append([(y[0] - b) / (b + 1) if index == i else (1.0 if i < index else 0.0) for i, b in enumerate(sa)])
    return np.array(ans)


x = np.arange(0, 32, 1).reshape((-1, 1))
a = np.array([[1, 2, 4, 8]])

v = mstanh(x, a)
w = spectram(x, a)

print(v)
print(w)


import matplotlib.pyplot as plt

plt.scatter(v[:,0], v[:,1])
plt.show()

plt.scatter(x, v.mean(-1))
plt.scatter(x, mstanh(x, [5]))
plt.show()

for y in v.T:
    plt.scatter(x[1:], np.diff(y))
plt.show()


plt.scatter(w[:,0], w[:,1])
plt.show()

plt.scatter(x, w.sum(-1))
plt.show()

for y in w.T:
    plt.scatter(x[1:], np.diff(y))
plt.show()
