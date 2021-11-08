import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim


# InfoGANによる潜在表現付き強化学習

class Environment:
    def __init__(self):
        self.sequence = []

    def outcome(self):
        counts = [self.sequence.count(i) for i in range(10)]
        return (max(counts) / 10) ** 2

    def step(self, action):
        self.sequence.append(action)

    def feature(self):
        sequence = self.sequence + [10] * 10
        return np.array([np.eye(11)[v][:10] for v in sequence[:10]], dtype=np.float32)


class Agent(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(20, 64)
        self.transfomer = nn.TransformerEncoderLayer(d_model=64, nhead=1, dropout=0.0, batch_first=True)
        self.fc_p = nn.Linear(64, 10)
        self.fc_v = nn.Linear(64, 1)

        self.fcz1 = nn.Linear(20, 64)
        self.transfomerz = nn.TransformerEncoderLayer(d_model=64, nhead=1, dropout=0.0, batch_first=True)
        self.fcz2 = nn.Linear(64, 10)

    def forward(self, x, z):
        z = z.unsqueeze(1).repeat(1, 10, 1)
        h = torch.cat([x, z], dim=2)
        h = F.relu(self.fc1(h))
        h = self.transfomer(h)
        h, _ = torch.max(h, dim=1)
        h_p = self.fc_p(h)
        h_v = self.fc_v(h)

        h = F.softmax(h_p, -1).unsqueeze(1).repeat(1, 10, 1)
        h = torch.cat([x, h], dim=2)
        h = F.relu(self.fcz1(h))
        h = self.transfomerz(h)
        h, _ = torch.max(h, dim=1)
        h_z = self.fcz2(h)

        return h_p, torch.tanh(h_v), h_z


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 64)
        self.transfomer = nn.TransformerEncoderLayer(d_model=64, nhead=1, dropout=0.0, batch_first=True)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        h = x
        h = F.relu(self.fc1(h))
        h = self.transfomer(h)
        h, _ = torch.max(h, dim=1)
        h = self.fc2(h)
        return h


def softmax(x):
    x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return x / x.sum(axis=-1, keepdims=True)


def generate(net):
    tragectory = []
    env = Environment()
    z = np.eye(10)[random.randrange(10)]
    zt = torch.FloatTensor(z).unsqueeze(0)
    for _ in range(10):
        with torch.no_grad():
            x = env.feature()
            xt = torch.FloatTensor(x).unsqueeze(0)
            p, _, _ = net(xt, zt)
            p = p.squeeze(0).numpy()
            prob = softmax(p)
        action = random.choices(list(range(10)), weights=prob)[0]
        env.step(action)
        tragectory.append((x, p, action))
    # terminal state
    tragectory.append((env.feature(), None, None))

    return tragectory, z, env.outcome()


def create_batch(episodes, batch_size):
    data = []
    for _ in range(batch_size):
        traj, z, oc = random.choice(episodes)
        feature, p, action = random.choice(traj[:-1])
        last_feature, _, _ = traj[-1]
        data.append((feature, p, [action], z, [oc], last_feature))

    def to_torch(x):
        if x.dtype == np.int32 or x.dtype == np.int64:
            return torch.LongTensor(x)
        else:
            return torch.FloatTensor(x)

    return [to_torch(np.array(d)) for d in zip(*data)]


def compute_loss(net, batch):
    features, bps, actions, zs, ocs, last_features = batch

    # 方策勾配法ロス
    ps, vs, est_zs = net(features, zs)
    bps_selected = F.softmax(bps, -1).gather(1, actions).detach()
    ps_selected = F.softmax(ps, -1).gather(1, actions).detach()
    clipped_rhos = torch.clamp(ps_selected / bps_selected, 0, 1)
    loss_p = -(clipped_rhos * (ocs - vs.detach()) * F.log_softmax(ps, -1)).sum()
    loss_v = (vs - ocs).pow(2).sum() / 2

    # 分類ロス
    #print(F.softmax(est_zs))
    loss_z = -(zs * F.log_softmax(est_zs, -1)).sum()

    #print(loss_p, loss_v, loss_z)

    return loss_p + loss_v + loss_z


net = Agent()
inv = Encoder()
optim = torch.optim.Adam(params=net.parameters(), lr=1e-4, weight_decay=1e-6)

episodes = []
for e in range(1000):
    new_episodes = []
    for i in range(100):
        episode = generate(net)
        new_episodes.append(episode)
    episodes += new_episodes

    # new episodesの選択回数分布を表示
    dist = [0 for _ in range(10)]
    oc_sum = 0
    for traj, _, oc in new_episodes:
        for _, _, action in traj[:-1]:
            dist[action] += 1
        oc_sum += oc
    print(dist)
    print(oc_sum / len(new_episodes))

    for s in range(30):
        b = create_batch(episodes, 64)
        loss = compute_loss(net, b)

        optim.zero_grad()
        loss.backward()
        optim.step()
