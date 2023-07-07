
import torch
import torch.nn as nn

encoder = nn.TransformerEncoderLayer(d_model=128, nhead=4, dim_feedforward=256, dropout=0, batch_first=True)
decoder = nn.TransformerDecoderLayer(d_model=128, nhead=4, dim_feedforward=256, dropout=0, batch_first=True, norm_first=True)

x = torch.randn((1, 10, 128))
y = torch.randn((1, 20, 128))

h = encoder(x)
print(h.shape)
print(h[0, :2, :8])

z = decoder(y, h)
print(z.shape)

print(z[0, :6, :8])

h_ = torch.cat([h[:, -5:], h[:, :5]], 1)

print(h_.shape)
print(decoder(y, h_).shape)
print(decoder(y, h_)[0, :6, :8])

y_ = torch.cat([y[:, -5:], y[:, :5]], 1)
#y_ = y[:, :3]

print(decoder(y_, h).shape)
print(decoder(y_, h)[0, :6, :8])
