
import copy
import torch
import torch.nn as nn

a = nn.Linear(1, 1)

state_dict1 = a.state_dict()
state_dict2 = {k: v.cpu() for k, v in state_dict1.items()}
state_dict3 = copy.deepcopy(state_dict2)

print(a.state_dict())

a.weight.data += 1

print(a.state_dict())
print(state_dict1)
print(state_dict2)
print(state_dict3)
