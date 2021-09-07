
# https://github.com/cavalleria/cavaface/blob/master/head/metrics.py

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ArcFace(nn.Module):
    def __init__(self, in_features, out_features, s=64.0, m_arc=0.50, m_am=0.0):
        super(ArcFace, self).__init__()

        self.s = s
        self.m_arc = m_arc
        self.m_am = m_am

        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

        self.cos_margin = math.cos(m_arc)
        self.sin_margin = math.sin(m_arc)
        self.min_cos_theta = math.cos(math.pi - m_arc)

    def forward(self, embbedings, label):
        # 入力
        # embeddings: バッチサイズ x 埋め込み次元
        # label: バッチサイズ x 1

        # 出力
        # バッチサイズ x クラス数

        embbedings = F.normalize(embbedings, dim=1)
        kernel_norm = F.normalize(self.weight, dim=0)
        cos_theta = torch.mm(embbedings, kernel_norm).clamp(-1, 1)
        sin_theta = torch.sqrt(1.0 - torch.pow(cos_theta, 2))
        cos_theta_m = cos_theta * self.cos_margin - sin_theta * self.sin_margin


        cos_theta_m = torch.where(
            cos_theta > self.min_cos_theta, cos_theta_m, cos_theta.float() - self.m_am,
        )
        index = torch.zeros_like(cos_theta)

        index.scatter_(1, label.data.view(-1, 1), 1)
        index = index.byte().bool()
        output = cos_theta * 1.0
        output[index] = cos_theta_m[index]
        output *= self.s

        return output


embbeding = nn.Embedding(8, 3)

model = ArcFace(3, 8)
model.train()
opt = optim.SGD(list(model.parameters()) + list(embbeding.parameters()), lr=1e-2)


labels = torch.LongTensor(np.arange(0, 4))

for _ in range(100):
    print(embbeding.weight.detach().numpy())

    loss = model(embbeding(labels), labels.unsqueeze(-1)).sum()

    loss.backward()
    opt.step()
