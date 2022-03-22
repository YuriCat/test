
import torch
import torch.nn as nn
import torch.optim as optim

net = nn.LayerNorm([8])

print(net.__dict__)

x = torch.FloatTensor([[0, 1, 2, 3, 4, 5, 6, 7]])

print(x)
print(net(x))

net1 = nn.LayerNorm([2, 2, 2])

print(x.view(-1, 2, 2))
print(net1(x.view(-1, 2, 2)))


opt = optim.SGD(net.parameters(), lr=1)
loss = net(x).pow(2).sum()
loss.backward()
opt.step()

print(net.__dict__)

print(net(x))
