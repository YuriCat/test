import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class A(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 4)
        self.bn = nn.BatchNorm1d(4, eps=1)
        self.fc2 = nn.Linear(4, 1)
        self.fca = nn.Linear(4, 4)

    def forward(self, x):
        h = self.fc1(x)
        h = self.bn(h)
        h = self.fca(x)
        h = self.bn(h)
        return self.fc2(h)

batch = np.random.random((4, 4)).astype(np.float32)
dummy_batch = np.ones((4, 4)).astype(np.float32) * -100
model = A()

optim = optim.SGD(model.parameters(), lr=0)

model.train()
for i in range(1000):
    #o = model.forward(torch.from_numpy(batch))
    #print(batch)
    #print(dummy_batch)
    with torch.no_grad():
        model.eval()
        model.forward(torch.from_numpy(dummy_batch))
        model.train()
    o = model.forward(torch.from_numpy(batch))
    loss = (o ** 2).sum()

    if i == 0:
        print('train', o.detach().numpy().reshape([-1]))

    optim.zero_grad()
    loss.backward()
    optim.step()

    if i % 10 == 0:
        model.eval()
        with torch.no_grad():
            o = model.forward(torch.from_numpy(batch))
        print('eval ', o.numpy().reshape([-1]))
        print('mean', model.bn._buffers['running_mean'], 'var', model.bn._buffers['running_var'])
        model.train()
