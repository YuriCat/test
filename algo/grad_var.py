
# 方策勾配の大きさを調べる

import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(1, 3, bias=False)
        nn.init.constant_(self.fc.weight, 0)
        self.fc_p = nn.Linear(1, 1)

    def forward(self, x):
        h = self.fc(x)
        h = self.fc_p(h.view(-1, 1)).view(x.size(0), -1)
        return h


x = torch.FloatTensor([[0], [1]])
act = torch.LongTensor([[0], [1]])
adv = torch.FloatTensor([[0.5], [0.5]])

net = Net()

p = F.log_softmax(net(x))
pa = p.gather(1, act)
loss = -(pa * adv).sum()

print(p)
print(loss)

loss.backward()
print(net.fc.weight.grad)
#print(net.fc.bias.grad)
print(net.fc_p.weight.grad)
print(net.fc_p.bias.grad)
