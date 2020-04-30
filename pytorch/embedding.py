# 埋め込みベクトルの仕様確認

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


emb = nn.Embedding(4, 2, padding_idx=0)

idx = np.array([[[0, 1], [0, 2]]])
tidx = torch.LongTensor(idx)
print(emb(tidx))


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(4, 2, padding_idx=0)
        self.fc = nn.Linear(2, 2)
    def forward(self, x):
        return self.fc(self.emb(x))

net = Net()
print(list(net.parameters()))

print(net(tidx))
torch.save(net.state_dict(), 'tmp.pth')

net = Net()
net.load_state_dict(torch.load('tmp.pth'))
print(net(tidx))

import pickle
pnet = pickle.dumps(net)
net = pickle.loads(pnet)
print(net(tidx))

net.train()
loss = net(tidx).sum() + net(tidx).sum()
opt = optim.SGD(net.parameters(), lr=1)
opt.zero_grad()
loss.backward()
opt.step()

print(list(net.parameters()))
print(net(tidx))
