# エントロピー正則化

import math

H = 0.1

er = lambda p: (1 - p) - p - H * ((1 - p) * math.log2(1 - p) + p * math.log2(p))


for i in range(1, 20):
    p = 0.1 ** i
    v = er(p)

    print(p, v)


from matplotlib import pyplot as plt

x = [0.9 ** i for i in range(100)]
y = [1 / (1 + 2 ** (-2 / x[i])) for i in range(100)]

print(x)
print(y)

plt.scatter(x, y)
plt.show()