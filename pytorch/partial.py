
# ある層の一部の重みだけ更新

import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class Model(torch.nn.Module):
    def __init__(self):
        # input: [-1, 2, 1]
        # output: [-1, 1]
        super().__init__()
        self.embedding = nn.Embedding(7, 4)
        self.fc = nn.Linear(8, 1)

    def forward(self, x):
        h = self.embedding(x)
        h = h.view(h.size(0), -1)
        return self.fc(h)

model = Model()
model.eval()
model.embedding.train()

optim = optim.Adam(lr=1e-2, params=model.embedding.parameters())
print(list(model.parameters()))

x = np.random.randint(7, size=(16, 2, 1))
y = np.random.random((16, 1))

for i in range(1000):
    o = model.forward(torch.from_numpy(x))
    loss = (o - torch.from_numpy(y)).pow(2).sum()

    optim.zero_grad()
    loss.backward()
    optim.step()

model.eval()
print(list(model.parameters()))