import numpy as np

import torch
import torch.nn as nn

gru = torch.nn.GRU(2, hidden_size=4, num_layers=3, batch_first=False)
lstm = torch.nn.LSTM(2, hidden_size=4, num_layers=3, batch_first=False)

x = np.random.randn(8, 5, 2)
x = torch.FloatTensor(x)

o, h = gru(x)
print(type(o), type(h))
print(o.size(), h.size())

o, h = gru(x, h)
print(type(o), type(h))
print(o.size(), h.size())

o, h = lstm(x)
print(type(o), type(h))
print(o.size(), h[0].size(), h[1].size())

o, h = lstm(x, h)
print(type(o), type(h))
print(o.size(), h[0].size(), h[1].size())