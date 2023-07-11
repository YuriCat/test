
import numpy as np
import torch
import torch.nn.functional as F

x_ = np.random.randn(8).astype(np.float32)

x = torch.tensor(x_, requires_grad=True)
log_p = F.log_softmax(x, -1)
loss = -x[0]
loss.backward()
print(x.grad)

x = torch.tensor(x_, requires_grad=True)

log_p = F.log_softmax(x, -1)
loss = -log_p[0]
loss.backward()
print(x.grad)

x = torch.tensor(x_, requires_grad=True)

loss = -x.mul(F.one_hot(torch.LongTensor([0]), 8) - F.softmax(x.detach(), -1)).sum()
loss.backward()
print(x.grad)

x = torch.tensor(x_, requires_grad=True)

loss = -F.log_softmax(x, -1).mul(F.one_hot(torch.LongTensor([0]), 8) - F.softmax(x.detach(), -1)).sum()
loss.backward()
print(x.grad)
