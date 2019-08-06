import numpy as np
import torch

a = np.array([[1, 2], [3, 4]])

b = np.tile(np.reshape(a, [2, 1, 2]), [1, 3, 1])
print(b)

c = torch.FloatTensor(a)

d = c.view(2, 1, 2).repeat(1, 3, 1)
print(d)
