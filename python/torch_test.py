import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

class ResidualBlock(nn.Module):
    def __init__(self, c0, c1):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Conv2d(c0, c1, 3, stride=1, padding=1, bias=True)
        #self.bn = nn.BatchNorm2d(c1)

    def forward(self, h):
        #return F.relu(self.bn(self.conv(s)))
        return F.relu(h + self.conv(h))

class Net(nn.Module):
    def __init__(self):
        # パラメータ設定
        self.board_x, self.board_y = 5, 5
        self.num_actions = 5
        c = 8
        self.num_input_channels = 4
        l = 4
        
        # ネットワーク定義
        super(Net, self).__init__()

        # 第1層
        self.l1_conv1 = nn.Conv2d(self.num_input_channels, c, 3, stride=1, padding=1, bias=True)
        self.l1_conv2 = nn.Conv2d(self.num_input_channels, c, 1, stride=1, bias=False)

        # 共有層
        self.common = []
        for _ in range(1, l):
            #self.common.append(ResidualBlock(c, c))
            self.common.append(nn.Conv2d(c, c, 3, stride=1, padding=1, bias=True))
        self.common = nn.ModuleList(self.common)

        # Policy
        self.conv_p = nn.Conv2d(c, 2, 1, stride=1)
        #self.bn_p = nn.BatchNorm2d(2)
        self.fc_p = nn.Linear(self.board_x * self.board_y * 2, self.num_actions)

        # Value
        self.conv_v = nn.Conv2d(c, 1, 1, stride=1)
        #self.bn_v = nn.BatchNorm2d(1)
        #self.bn_v = nn.BatchNorm1d(self.board_x * self.board_y * 1)
        self.fc_v1 = nn.Linear(self.board_x * self.board_y * 1, 4)
        self.fc_v2 = nn.Linear(4, 1)

    def forward(self, x):
        h = x.view(-1, 4, self.board_x, self.board_y)

        h = F.relu(self.l1_conv1(h) + self.l1_conv2(h))
        
        for block in self.common:
            #h = block(h)
            h = F.relu(h + block(h))
            print(h)

        h_p = self.conv_p(h)
        #h_p = self.bn_p(h_p)
        h_p = h_p.view(-1, self.board_x * self.board_y * 2)
        h_p = self.fc_p(F.relu(h_p))

        h_v = self.conv_v(h)
        #h_v = self.bn_v(h_v)
        h_v = h_v.view(-1, self.board_x * self.board_y * 1)
        #s_v = self.bn_v(s_v)
        h_v = self.fc_v2(F.relu(self.fc_v1(F.relu(h_v))))

        return F.log_softmax(h_p, dim=1), F.tanh(h_v)

class MiniNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(5 * 5 * 4, 5)
        self.fc2 = nn.Linear(5 * 5 * 4, 1)
    def forward(self, x):
        h = x.view(-1, 100)
        h_p = self.fc1(h)
        h_v = self.fc2(h)
        return F.log_softmax(h_p, dim=1), F.tanh(h_v)


def make_sample_batch():
    b = np.ones((16, 5, 5, 4))
    p = np.zeros((16, 5))
    p[:,0] = 1
    v = np.zeros((16, 1))
    return b, p, v

def make_sample_input():
    return np.zeros((5, 5, 4))

def loss_p(targets, outputs):
    return -torch.sum(targets*outputs)/targets.size()[0]

def loss_v(targets, outputs):
    return torch.sum((targets-outputs.view(-1))**2)/targets.size()[0]

def train(net):
    num_epochs = 100
    optimizer = optim.Adam(net.parameters())
    for epoch in range(num_epochs):
        net.train()
        b, p_, v_ = make_sample_batch()
        b = Variable(torch.FloatTensor(b))
        p_ = Variable(torch.FloatTensor(p_))
        v_ = Variable(torch.FloatTensor(v_))
        p, v = net(b)
        l_p, l_v = loss_p(p_, p), loss_v(v_, v)
        total_loss = l_p + l_v

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
    return net
    

def predict(net):
    b = make_sample_input()
    with torch.no_grad():
        b = Variable(torch.FloatTensor(b))
    net.eval()
    p, v = net(b)
    p, v = torch.exp(p).data.cpu().numpy()[0], v.data.cpu().numpy()[0]
    print(p, v)

filepath = 'model'

net = Net()#MiniNet()

#net = train(net)
#predict(net)
#torch.save({'state_dict' : net.state_dict()}, filepath)

checkpoint = torch.load(filepath)
print(checkpoint['state_dict'])
net.load_state_dict(checkpoint['state_dict'])
predict(net)