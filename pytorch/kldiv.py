# https://stackoverflow.com/questions/49886369/kl-divergence-for-two-probability-distributions-in-pytorch

import torch
import torch.nn.functional as F

# this is the same example in wiki
P = torch.Tensor([0.36, 0.48, 0.16])
Q = torch.Tensor([0.333, 0.333, 0.333])

print((P * (P / Q).log()).sum())
# tensor(0.0863), 10.2 µs ± 508

print(F.kl_div(Q.log(), P, None, None, 'sum'))
# tensor(0.0863), 14.1 µs ± 408 ns
