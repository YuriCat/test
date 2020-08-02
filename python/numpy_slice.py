import numpy as np

a = np.array([[1, 2], [3, 4]])
#b = a[0,:]
#b[0] = -1

c = np.array([[5, 6], [7, 8]])

d = np.concatenate([a, c])
d[0, 0] = -2

print(d)
print(a)
print(c)

e = a[[0, 1, 0, 1]]
e[0, 0] = -3

print(e)
print(a)
