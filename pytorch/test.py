import torch
import torch.nn as nn

class A(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = 'A'
    def eval(self):
        super().eval()
        self.name = 'EVAL'
    def forward(self, x):
        print(self.name)
        return x

class B(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = A()
    def forward(self, x):
        x = self.base(x)
        return x

b = B()
b.eval()
b.forward(0)