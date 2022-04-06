
import numpy as np
import torch
import torch.nn.functional as F

x_ = np.random.randn(8).astype(np.float32)
x = torch.tensor(x_, requires_grad=True)
p = F.log_softmax(x, -1)
p_ = p.detach()
q = F.log_softmax(torch.randn(8), -1)

print(p)
print(q)

kl_div = F.kl_div(q, torch.exp(p), reduction='sum')
print(kl_div)

kl_div.backward()
print(x.grad)

x = torch.tensor(x_, requires_grad=True)
p = F.log_softmax(x, -1)

divided_kl_loss = 0
for index in range(len(x)):
    divided_kl_loss += (q[index] - p_[index]) * torch.exp(p_[index]) * -p[index]

divided_kl_loss.backward()
print(x.grad)

x = torch.tensor(x_, requires_grad=True)
p = F.log_softmax(x, -1)

ex_kl_loss = 0
for _ in range(40000):
    index = torch.exp(q).multinomial(num_samples=1, replacement=True)
    rho = torch.clamp(torch.exp(p_[index]) / torch.exp(q[index]), 0, 1e8)
    ex_kl_loss += (q[index] - p_[index]) * rho * -p[index]

ex_kl_loss.backward()
print(x.grad / 40000)

x = torch.tensor(x_, requires_grad=True)
p = F.log_softmax(x, -1)

inv_kl_div = F.kl_div(p, torch.exp(q), reduction='sum')
print(inv_kl_div)

inv_kl_div.backward()
print(x.grad)
