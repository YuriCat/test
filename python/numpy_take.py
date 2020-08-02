
# numpyの多次元スライス

import numpy as np

a = np.array([
    [1, 2], [3, 4], [5, 6], [7, 8],
])

b = np.array([
    [0, 1], [1, 0],
])

c = np.array([
    [[0], [1]], [[1], [0]], [[0], [2]],
])

print(np.take(a, b).shape)
print(np.take(a, b))
print(np.take(a, b, axis=1).shape)
print(np.take(a, b, axis=1))

print(np.take(a, c).shape)
print(np.take(a, c))
#print(np.take(a, c, axis=1).shape)
#print(np.take(a, c, axis=1))

#print(np.take_along_axis(a, b, axis=0).shape)
#print(np.take_along_axis(a, b, axis=1).shape)

a = np.arange(34 * 3, dtype=np.int32).reshape(34, 3)
b = np.zeros((6, 6), dtype=np.int32)

print(a[b].shape)
print(a[b])
