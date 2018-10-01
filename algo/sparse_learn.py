# 疎行列の線形モデルの学習

import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.v = nn.Linear(65536, 1)
    def forward(self, x):
        v_out = self.v(x)
        return v_out

def make_input():
    i = torch.LongTensor([[1, 0], [2, 0], [50000, 0]]) # 座標
    v = torch.FloatTensor([3,      4,      50000]) # 値
    t = torch.sparse.FloatTensor(i, v, torch.Size([65536, 1]))
    return v

def learn():
    model = Net()
    for i in range(100):
        loss = 
    