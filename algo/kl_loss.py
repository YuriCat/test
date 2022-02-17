
import numpy as np
import torch
import torch.nn.functional as F

x_ = np.random.randn(8).astype(np.float32)
x = torch.tensor(x_, requires_grad=True)
p = F.log_softmax(x)
q = F.log_softmax(torch.randn(8))

kl_div = F.kl_div(p, torch.exp(q), reduction='sum')
print(kl_div)

kl_div.backward()
print(x.grad)

x = torch.tensor(x_, requires_grad=True)
p = F.log_softmax(x)

ex_kl_loss = 0
for _ in range(100000):
    index = torch.exp(q).multinomial(num_samples=1, replacement=True)
    p_ = p.detach()
    ex_kl_loss += (q[index] - p_[index]) * torch.exp(p_[index]) / torch.exp(q[index]) * -p[index]

ex_kl_loss.backward()
print(x.grad / 100000)
