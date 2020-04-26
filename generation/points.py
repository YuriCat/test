import numpy as np
import torch
import pyro

class Net(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.rg = torch.distributions.Bernoulli(0.5)
        self.fc = torch.nn.Linear(dim, dim, bias=False)
    def forward(self, batch_size):
        h = self.rg.sample((batch_size, self.dim))
        return self.fc(h)

dim = 4
net = Net(dim)

def true_data(bs, dim):
    a = np.random.randint(2, size=(bs, 1))
    return np.tile(a, [1, dim])

def loss_function(x, y):
    return ((((x - y) ** 2).sum(-1) + 1e-6) ** 0.5)

def annealing(x, y):
    # アニーリングで距離最小のマッチングを作成
    indice = np.arange(len(x))
    loss_each = loss_function(x, y) # 0で良いけどワッサースタイン距離を知るために一応
    loss = loss_each.sum()
    best_indice, best_loss = None, float('inf')
    steps = 3000
    for s in range(steps):
        i, j = np.random.randint(len(x), size=2)
        idxi, idxj = indice[i], indice[j]
        nlossi = loss_function(x[i], y[idxj])
        nlossj = loss_function(x[j], y[idxi])
        d = 0
        d += nlossi + nlossj
        d -= loss_each[i] + loss_each[j]
        if d < 0 or np.random.random() < 0.15:
            indice[i], indice[j] = idxj, idxi
            loss_each[i], loss_each[j] = nlossi, nlossj
            loss += d
        if loss < best_loss:
            best_indice, best_loss = np.copy(indice), loss
    print(loss)
    return best_indice

# sort
def sort_train():
    optim = torch.optim.SGD(lr=1e-2, params=net.parameters())
    bs = 16
    for i in range(1000):
        y_ = np.sort(np.random.randint(2, size=(bs, 1)), axis=0)
        y = net(bs)
        y, _ = torch.sort(y, dim=0)
        #print(y_, y)

        loss = ((y - torch.Tensor(y_)).pow(2).sum(-1) + 1e-6).pow(0.5).mean()
        #print(loss)

        optim.zero_grad()
        loss.backward()
        optim.step()


# annealing matching
def anneal_train():
    optim = torch.optim.SGD(lr=1e-3, params=net.parameters())
    bs = 128
    for i in range(1000):
        y_ = true_data(bs, dim)
        y = net(bs)

        indice = annealing(y.detach().numpy(), y_) # matching points
        y_sorted = torch.Tensor(y_[indice])

        loss = loss_function(y, y_sorted).sum()

        optim.zero_grad()
        loss.backward()
        optim.step()
        print(list(net.parameters()))

def sample_train():
    optim = torch.optim.SGD(lr=1e-2, params=net.parameters())
    bs = 1
    sn = 4
    for i in range(1000):
        y_ = true_data(bs, dim)
        y_ = np.tile(np.expand_dims(y_, 1), [1, sn, 1])
        y = net(bs * sn).view(-1, sn, dim)
        #print(y)

        #print(y_.shape, y.shape)
        print((y - torch.Tensor(y_)).pow(2).sum(-1).pow(0.5).min(-1)[0])

        loss = (y - torch.Tensor(y_)).pow(2).sum(-1).pow(0.5).min(-1)[0].mean()

        optim.zero_grad()
        loss.backward()
        optim.step()

#sort_train()
anneal_train()
#sample_train()
print(list(net.parameters()))
