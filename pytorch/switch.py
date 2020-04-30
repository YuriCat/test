
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


net = SwitchingLinear(3, 1, 2)


x = torch.FloatTensor([[1], [1], [1]])
xid = torch.LongTensor([0, 1, 2])

print(x)
print(net(x, xid))
