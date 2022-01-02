import torch
import mctorch.nn as mnn
import mctorch.optim as moptim

# Random data with high variance in first two dimension
X = torch.diag(torch.FloatTensor([3,2,1])).matmul(torch.randn(3,200))

# 1. Initialize Parameter
manifold_param = mnn.Parameter(manifold=mnn.Stiefel(3,2))

# 2. Define Cost - squared reconstruction error
def cost(X, w):
    wTX = torch.matmul(w.transpose(1,0), X)
    wwTX = torch.matmul(w, wTX)
    return torch.sum((X - wwTX)**2)

# 3. Optimize
optimizer = moptim.rAdagrad(params = [manifold_param], lr=1e-2)

for epoch in range(30):
    cost_step = cost(X, manifold_param)
    print(cost_step)
    cost_step.backward()
    optimizer.step()
    optimizer.zero_grad()
