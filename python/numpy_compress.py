# numpyでフラグにより配列を圧縮

import numpy as np

a = [1.0, 0, 1.0, 0]
b = [1, 2, 3, 4]

c = np.compress(a, b)
print(c)

