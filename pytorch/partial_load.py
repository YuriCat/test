
import torch
import torch.nn as nn
import torch.nn.functional as F

class NetA(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(1, 1)
    def forward(self, x):
        return self.fc(x)

class NetB(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 1)
    def forward(self, x):
        return self.fc(x)

a = NetA()
b = NetB()

loaded = a.state_dict()
initialized = b.state_dict()

print(loaded)
print(initialized)

import copy

# 残り部分が初期化の時、値が小さめになるように調整?
# それか0でも良い?
e = 0.2

partially_loaded = copy.deepcopy(initialized)
for k, v in loaded.items():
    if k in partially_loaded:
        target = partially_loaded[k]
        if v.shape == target.shape:
            partially_loaded[k] = v
        else:
            padding_shape = sum([(0, target.shape[d] - v.shape[d]) for d in reversed(range(v.ndim))], ())
            flag = torch.ones_like(v)
            padded_v = F.pad(v, padding_shape, 'constant', 0)
            padded_flag = F.pad(flag, padding_shape, 'constant', 0)

            print(padded_v.shape, target.shape, padded_flag.shape)
            
            merged = padded_v + target * (1 - padded_flag) * e
            partially_loaded[k] = merged

print(partially_loaded)
