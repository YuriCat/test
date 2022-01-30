
import torch

a = torch.zeros((1, 2, 3))
print(a.shape)

b = a.flatten(0, 1)
print(b.shape)

