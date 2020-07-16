# https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html

import torch
import torch.nn as nn
import torch.optim as optim

class A(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 4)
        self.fc2 = nn.Linear(4, 4)
        self.bn = nn.BatchNorm1d(4, eps=1)
        self.fca = nn.Linear(4, 1)
        self.fcb = nn.Linear(4, 1)

    def forward(self, x):
        a, b = x
        h = self.fc1(a) + self.fc2(b)
        h = self.bn(h)
        h = h[:, :4]
        h = h.view(*h.size()[:1], *h.size()[1:])
        return self.fca(h), self.fcb(h)


torch_model = A()
torch_model.eval()

x = [
    torch.randn(4, 4, requires_grad=True),
    torch.randn(4, 4, requires_grad=True)
]

torch_out = torch_model(x)
in_names = ['a', 'b']
out_names = ['output.p', 'output.v']

torch.onnx.export(
    torch_model,                  # model being run
    x,                            # model input (or a tuple for multiple inputs)
    "tmp.onnx",                   # where to save the model (can be a file or file-like object)
    export_params=True,           # store the trained parameter weights inside the model file
    opset_version=10,             # the ONNX version to export the model to
    do_constant_folding=True,     # whether to execute constant folding for optimization
    input_names = in_names,       # the model's input names
    output_names = out_names,     # the model's output names
    verbose=True,
    dynamic_axes={k: {0 : 'batch_size'} for k in (in_names + out_names)},    # variable lenght axes
)
