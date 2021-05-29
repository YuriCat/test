
import torch
import torch.nn.functional as F

a, b = torch.rand(5, 10), torch.rand(5, 10)
a, b = a / a.sum(-1, keepdim=True), b / b.sum(-1, keepdim=True)

print('prob a =', a)
print('prob b =', b)

kl = F.kl_div(a.log(), b, reduction='none').sum(-1)
d = ((a - b) ** 2).sum(-1)

print(torch.exp(-d))

print(torch.exp(-kl))

print(torch.exp(-kl ** 2))
