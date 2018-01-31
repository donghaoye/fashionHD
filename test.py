from __future__ import division
import torch
import torch.nn as nn
from torch.autograd import Variable

class Model(nn.Module):
    def forward(self, x):
        return [x, x+1]

model = Model()
x = Variable(torch.ones(2,2))
y = nn.parallel.data_parallel(model, x, device_ids = [0,1])
print(y)
