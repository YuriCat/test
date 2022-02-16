
import torch
import torch.nn.functional as F

def sigplus(x):
    return F.softplus(x) * torch.sigmoid(x)

def relu(x):
    return F.relu(x)

x = torch.arange(-10, 10, 0.05)
y = sigplus(x)
z = relu(x)
al = F.softplus(x)

import matplotlib.pyplot as plt

plt.scatter(x.detach().numpy(), z.detach().numpy())
plt.scatter(x.detach().numpy(), al.detach().numpy())
plt.scatter(x.detach().numpy(), y.detach().numpy())
plt.show()
