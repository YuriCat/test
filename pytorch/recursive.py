
# 再帰的関係を学習する

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class RecursiveModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 4)
        self.fc2 = nn.Linear(4, 2, bias=False)
        self.emb = nn.Embedding(4, 2)

    def forward(self, x, sid):
        h = torch.cat([x, self.emb(sid)], axis=-1)
        h = self.fc1(h)
        h = F.relu(h)
        return self.fc2(h)

    def embed_forward(self, xid):
        return self.emb(xid)

X = np.array([
    [0.1, 0.3],
    [0.3, 0.7],
    [0.3, 0.2],
    [0.2, 0.8],
], dtype=np.float32)
XID = np.array([0, 1, 2, 3])
FID = np.array([1, 0, 2, 2])

model = RecursiveModel()
optim = optim.SGD(model.parameters(), lr=1e-2)

model.train()
for i in range(100000):
    x = torch.from_numpy(X)
    xid = torch.from_numpy(XID)
    fid = torch.from_numpy(FID)

    o = model(x, fid)

    loss = ((o - model.embed_forward(xid)) ** 2).sum()

    optim.zero_grad()
    loss.backward()
    optim.step()

    print('out = \n', model(x, fid).detach().numpy())
    print('emb = \n', list(model.parameters())[-1].detach().numpy())
