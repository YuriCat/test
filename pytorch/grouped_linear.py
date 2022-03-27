
import torch
import torch.nn as nn


class GroupedLinear(nn.Linear):
    def __init__(self, in_features, out_features, groups, bias=True,
                 device=None, dtype=None):
        super().__init__(in_features, out_features//groups, bias, device, dtype)
        self.groups = groups
        self.out_features = out_features // groups

    def forward(self, input):
        x = input.view(-1, self.groups, input.size(-1) // self.groups)
        w = self.weight.view(self.groups, -1, self.out_features).repeat(input.size(0), 1, 1)
        print(x.shape, w.shape)
        o = torch.bmm(x, w).view(-1, self.groups * self.out_features)
        if self.bias is not None:
            o = o + self.bias
        return o
