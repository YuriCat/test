import numpy as np
import torch
import torch.nn.functional as F

x_ = np.random.randn(8).astype(np.float32)
x = torch.tensor(x_, requires_grad=True)
p = F.log_softmax(x, -1)

p[0].backward()
print(x.grad)
x.grad.zero_()

x[0].backward()
print(x.grad)
