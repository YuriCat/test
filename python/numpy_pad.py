import numpy as np

a = np.array([1, 2, 3])
b = np.array([[1, 2], [3, 4]])

print(np.pad(a, (0, 2), 'constant', constant_values=0))
print(np.pad(a, (1, 2), 'constant', constant_values=1))

print(np.pad(b, [(0, 0), (0, 2)], 'constant', constant_values=0))
print(np.pad(b, [(0, 0), (0, 2)], constant_values=0))
