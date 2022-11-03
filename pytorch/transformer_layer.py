
import torch
import torch.nn as nn

encoder = nn.TransformerEncoderLayer(d_model=128, nhead=4, dim_feedforward=256, batch_first=True)
decoder = nn.TransformerDecoderLayer(d_model=128, nhead=4, dim_feedforward=256, batch_first=True)

x = torch.zeros((1, 10, 128))
y = torch.zeros((1, 20, 128))

h = encoder(x)
print(h.shape)
print(h[0, :2, :8])

z = decoder(y, h)
print(z.shape)

print(z[0, :3, :8])

