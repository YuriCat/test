
# インデックスによって重みを切り替えるレイヤー

import math
import numpy as np
import torch


class SwitchingLinear(torch.nn.Module):
    def __init__(self, num_classes, in_features, out_features, bias=True):
        super().__init__()
        self.num_classes = num_classes
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.parameter.Parameter(torch.Tensor(num_classes, out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, input, index):
        weight = self.weight[index]
        return torch.bmm(weight, input.unsqueeze(-1)).squeeze(-1)
        #return torch.mv(weight.permute(1, 0, 2).reshape(self.out_features, -1), input.view(-1)).view(-1, self.out_features)


net = SwitchingLinear(3, 1, 2)


x = torch.FloatTensor([[1], [1], [1]])
xid = torch.LongTensor([0, 1, 2])

print(x)
print(net(x, xid))


# 速度比較

num_classes = 100

sfc = SwitchingLinear(num_classes, 1000, 1000)
fc = torch.nn.Linear(num_classes * 1000, 1000)

x_np = np.random.randn(num_classes, 1000)
x = torch.FloatTensor(x_np)
xid = torch.LongTensor(np.arange(num_classes))

bx_np = np.zeros((num_classes, num_classes * 1000))
for i in range(num_classes):
    bx_np[i, 1000 * i: 1000 * (i + 1)] = x_np[i]
bx = torch.FloatTensor(bx_np).contiguous()


import time

print(x.size())
t = time.time()
sfc(x, xid)
print('sfc', time.time() - t)

print(bx.size())
t = time.time()
fc(bx)
print('fc', time.time() - t)
