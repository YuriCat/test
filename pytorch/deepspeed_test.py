import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import deepspeed

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(20, 100)
        self.fc2 = nn.Linear(100, 1)

    def forward(self, x):
        h = self.fc1(x)
        h = F.relu_(h)
        h = self.fc2(h)
        return h

net = Net()
optimizer = optim.Adam(net.parameters(), lr=1e-4)
B = 1000

deepspeed_args = {
    "train_batch_size": 8,
    "gradient_accumulation_steps": 1,
    "optimizer": {
        "type": "Adam",
        "params": {
            "lr": 1e-4
        }
    },
    "fp16": {
        "enabled": True
    },
    "zero_optimization": True
}

net_engine, optimizer, _, _ = deepspeed.initialize(args=deepspeed_args, model=net, model_parameters=net.parameters())

for e in range(10000):
    x = torch.FloatTensor(np.random.randn(B, 20))
    y_ = torch.FloatTensor(np.random.randn(B, 1))

    #y = net(x)
    y = net_engine(x)
    loss = (y - y_).pow(2).mean()
    #loss.backward()
    net_engine.backward(loss)
    #optimizer.step()
    net_engine.step()
    print(e, loss.item())

