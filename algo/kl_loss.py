
import numpy as np
import torch
import torch.nn.functional as F

x_ = np.random.randn(8).astype(np.float32)
x = torch.tensor(x_, requires_grad=True)
p = F.log_softmax(x, -1)
p_ = p.detach()
q = F.log_softmax(torch.randn(8), -1)

print('target log-policy')
print(p)
print('behaviour log-policy')
print(q)

print('### Reversed KL Loss ###')

print('full')
kl_div = F.kl_div(q, torch.exp(p), reduction='sum')
print(kl_div)

kl_div.backward()
print(x.grad)

print('divided')
x = torch.tensor(x_, requires_grad=True)
p = F.log_softmax(x, -1)

divided_kl_loss = 0
for index in range(len(x)):
    divided_kl_loss += (q[index] - p_[index]) * torch.exp(p_[index]) * -p[index]

divided_kl_loss.backward()
print(x.grad)

print('expected')
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

print('### Normal KL Loss ###')

print('full')
x = torch.tensor(x_, requires_grad=True)
p = F.log_softmax(x, -1)

kl_div = F.kl_div(p, torch.exp(q), reduction='sum')
print(kl_div)

kl_div.backward()
print(x.grad)

print('divided')
x = torch.tensor(x_, requires_grad=True)
p = F.log_softmax(x, -1)

divided_kl_loss = 0
for index in range(len(x)):
    divided_kl_loss += (torch.exp(p_[index]) - torch.exp(q[index])) * x[index]

divided_kl_loss.backward()
print(x.grad)

print('expected')
x = torch.tensor(x_, requires_grad=True)
p = F.log_softmax(x, -1)

ex_kl_loss = 0
for _ in range(40000):
    index = torch.exp(q).multinomial(num_samples=1, replacement=True)
    rho = torch.clamp(torch.exp(p_[index]) / torch.exp(q[index]), 0, 1e8)
    ex_kl_loss += (rho - 1) * x[index]

ex_kl_loss.backward()
print(x.grad / 40000)

print('expected-p')
x = torch.tensor(x_, requires_grad=True)
p = F.log_softmax(x, -1)

ex_kl_loss_p = 0
for _ in range(40000):
    index = torch.exp(q).multinomial(num_samples=1, replacement=True)
    rho = torch.clamp(torch.exp(p_[index]) / torch.exp(q[index]), 0, 1e8)
    ex_kl_loss_p += (rho - 1) * p[index]

ex_kl_loss_p.backward()
print(x.grad / 40000)

