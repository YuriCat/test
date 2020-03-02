# 埋め込みベクトルの仕様確認

import numpy as np
import torch
import torch.nn as nn

emb = nn.Embedding(4, 2, padding_idx=0)

idx = np.array([[[0, 1]]])
tidx = torch.LongTensor(idx)

e = emb(tidx)

print(e)
