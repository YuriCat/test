
import numpy as np
import torch
import torch.nn.functional as F

a = np.array([[1, 2], [3, 4]])

ta = torch.Tensor(a)

print(torch.chunk(ta, 2, 0))
print(torch.chunk(ta, 2, 1))

