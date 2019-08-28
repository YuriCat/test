import numpy as np

a = [1, 1]
b = [1, 1]

c = {'k':a}
d = {'k':b}

print(np.allclose(a, b))
print(np.allclose(c, d))
