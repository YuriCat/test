
import numpy as np
import torch

x = torch.FloatTensor(np.random.random((1, 2, 3)))
print(x.shape)
print(x)

a = torch.LongTensor(np.zeros((1, 10, 3), dtype=np.int64))
z = x.gather(1, a)
print(a.shape, z.shape)
print(z)

a = torch.LongTensor(np.zeros((1, 4, 2), dtype=np.int64))
z = x.gather(1, a)
print(a.shape, z.shape)
print(z)

#a = torch.LongTensor(np.zeros((1, 3, 4, 2), dtype=np.int64))
#z = x.unsqueeze(1).gather(1, a)
#print(a.shape, z.shape)
#print(z)

a = torch.LongTensor(np.zeros((1, 3, 10), dtype=np.int64))
z = x.permute(0, 2, 1).gather(2, a).permute(0, 2, 1)
print(a.shape, z.shape)
print(z)
