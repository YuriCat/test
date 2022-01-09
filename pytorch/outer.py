
import torch

# https://pytorch.org/docs/stable/generated/torch.outer.html

# without batch dimension

v1 = torch.arange(1., 5.)
v2 = torch.arange(1., 4.)

print(v1)
print(v2)
print(torch.outer(v1, v2))

# with batch dimension

v1 = v1.view(1, -1)
v2 = v2.view(1, -1)

bmm = torch.bmm(v1.unsqueeze(2), v2.unsqueeze(1))
print(bmm)
