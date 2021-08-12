import torch
import torch.nn as nn

embed = nn.EmbeddingBag(3, 5, mode='sum')

x = torch.LongTensor([[0, 1]])

y = embed(x)

print(embed.weight.data)
print(x)
print(y)
