
# https://github.com/ndrplz/ConvLSTM_pytorch/blob/master/convlstm.py

import numpy as np
import torch
import torch.nn as nn

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super().__init__()

        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding     = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias        = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, input_size, batch_size):
        return (
            torch.zeros(batch_size, self.hidden_dim, *input_size),
            torch.zeros(batch_size, self.hidden_dim, *input_size)
        )

class DRC(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, kernel_size, bias):
        super().__init__()
        self.num_layers = num_layers

        blocks = []
        for _ in range(self.num_layers):
            blocks.append(ConvLSTMCell(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                kernel_size=kernel_size,
                bias=bias)
            )
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x, hidden, num_repeats):
        if hidden is None:
            hidden = self.init_hidden(x.shape[2:], x.shape[0])

        hs = [hidden[0][:,i] for i in range(self.num_layers)]
        cs = [hidden[1][:,i] for i in range(self.num_layers)]
        for _ in range(num_repeats):
            for i, block in enumerate(self.blocks):
                hs[i], cs[i] = block(x, (hs[i], cs[i]))

        return hs[-1], (torch.stack(hs, dim=1), torch.stack(cs, dim=1))

    def init_hidden(self, input_size, batch_size):
        hs, cs = [], []
        for block in self.blocks:
            h, c = block.init_hidden(input_size, batch_size)
            hs.append(h)
            cs.append(c)

        return torch.stack(hs, dim=1), torch.stack(cs, dim=1)


net = DRC(num_layers=3, input_dim=8, hidden_dim=16, kernel_size=(3, 3), bias=True)
x = torch.randn(5, 8, 4, 4)
h = net.init_hidden((4, 4), 5)

y, h = net(x, h, num_repeats=3)
print(y.size(), h[0].size(), h[1].size())
y, h = net(x, h, num_repeats=10)
print(y.size(), h[0].size(), h[1].size())