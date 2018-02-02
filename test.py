from __future__ import division
import torch
import torch.nn as nn
from torch.autograd import Variable

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(2,1,5)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0),-1).mean(1)
        return x


m = Model()
x = Variable(torch.rand(10,2,5,5), requires_grad = True)
y = m(x)

grad1 = torch.autograd.grad(outputs = y, inputs = x,
    grad_outputs = y.data.new(y.size()).fill_(1),
    create_graph = True, retain_graph = True, only_inputs = True)[0]

grad2 = torch.autograd.grad(outputs = y.sum(), inputs = x,
    create_graph = True, retain_graph = True, only_inputs = True)[0]

print(grad1-grad2)