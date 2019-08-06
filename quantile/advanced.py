import numpy as np
from matplotlib import pyplot as plt

y = [-4, -3, -2, -1, 0, 0, 6]

def loss(x, coef):
    sum = np.zeros_like(x)
    for val in y:
        sum += np.abs(x - val) ** coef
    return sum / len(y)

x = np.arange(-8, 8, 0.01)

coefs = np.arange(0, 2.5 + 1e-8, 0.1)
coefs = coefs ** 2

for c in coefs:
    plt.plot(x, loss(x, c))

plt.ylim(0, 3)
plt.show()