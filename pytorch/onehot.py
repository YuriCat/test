
import torch

# https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507/3

y = torch.LongTensor([[0], [1], [2]])

y_onehot = torch.FloatTensor(3, 3)
y_onehot.zero_()
y_onehot.scatter_(1, y, 1)

print(y)
print(y_onehot)
