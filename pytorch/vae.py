# シンプルな確率分布を学習する
# modified from https://github.com/pytorch/examples/blob/master/vae/main.py

import torch
from torch import nn, optim
from torch.nn import functional as F


def data_generator(batch_size):
    while True:
        #x = (torch.randint(2, size=(batch_size, 1)) * 2 - 1).float()
        x = torch.randn(batch_size, 1)
        yield x, None


batch_size = 256
epochs = 10
batch_cnt = 1000

train_loader = data_generator(batch_size)
test_loader = data_generator(batch_size)


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.fc11 = nn.Linear(1, 1)
        self.fc12 = nn.Linear(1, 1)
        self.fc2 = nn.Linear(1, 1)

    def encode(self, x):
        return self.fc11(x), self.fc12(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        r = mu + eps*std
        return r

    def decode(self, z):
        return self.fc2(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


model = VAE()
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    MSE = 0.5 * (recon_x - x).pow(2).sum()

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return MSE + KLD


def train(epoch):
    model.train()
    train_loss = 0
    cnt = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        if cnt >= batch_cnt:
            break
        cnt += 1
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / (batch_cnt * batch_size)))


if __name__ == "__main__":
    for epoch in range(1, epochs + 1):
        train(epoch)
        with torch.no_grad():
            #sample = torch.randint(2, size=(16, 1)).float()
            sample = torch.randn(16, 1)
            output = model.decode(sample).cpu()
            print(output)
            print(list(model.parameters()))
