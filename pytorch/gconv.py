import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True, init='xavier'):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        if init == 'uniform':
            print("| Uniform Initialization")
            self.reset_parameters_uniform()
        elif init == 'xavier':
            print("| Xavier Initialization")
            self.reset_parameters_xavier()
        elif init == 'kaiming':
            print("| Kaiming Initialization")
            self.reset_parameters_kaiming()
        else:
            raise NotImplementedError

    def reset_parameters_uniform(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def reset_parameters_xavier(self):
        nn.init.xavier_normal_(self.weight.data, gain=0.02) # Implement Xavier Uniform
        if self.bias is not None:
            nn.init.constant_(self.bias.data, 0.0)

    def reset_parameters_kaiming(self):
        nn.init.kaiming_normal_(self.weight.data, a=0, mode='fan_in')
        if self.bias is not None:
            nn.init.constant_(self.bias.data, 0.0)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.einsum('jk,ikl->ijl', adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

# https://discuss.pytorch.org/t/kronecker-product/3919
def kronecker(A, B):
    return torch.einsum("ab,cd->acbd", A, B).view(A.size(0) * B.size(0), A.size(1) * B.size(1))

class BatchGraphConvolution(nn.Module):
    def __init__(self, gconv_layer):
        super().__init__()
        self.gconv = gconv_layer
    
    def forward(self, input, adj):
        adj_ = kronecker(torch.eye(input.size(0)), adj)
        output = self.gconv(input.view(-1, self.gconv.in_features), adj_)
        return output.view(input.size(0), -1, self.gconv.out_features)

# グラフデータ
g0 = np.array([0, 1, 2]).reshape([-1, 1])
g1 = np.array([0, 1, 3]).reshape([-1, 1])
g = np.stack([g0, g1])
adj = np.ones((len(g0), len(g0))) - np.eye(len(g0))

layer = GraphConvolution(1, 1, False)
blayer = BatchGraphConvolution(layer)

_g0 = torch.FloatTensor([g0])
_g1 = torch.FloatTensor([g1])
_g = torch.FloatTensor(g)
_adj = torch.FloatTensor(adj)


print(layer(_g0, _adj))
print(layer(_g1, _adj))
print(layer(_g, _adj))