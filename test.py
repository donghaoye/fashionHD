from __future__ import division
import torch
import torch.nn as nn
from torch.autograd import Variable

class Encoder(nn.Module):
    def __init__(self, ndowns):
        super(Encoder, self).__init__()
        c_out = 1
        layers = []
        for i in range(ndowns):
            c_in = c_out
            c_out = c_out*2
            layers += [nn.Conv2d(c_in, c_out, 4, 2, 1)]
            if i < 7:
                layers += [nn.InstanceNorm2d(c_out)]
            layers += [nn.ReLU()]

        self.down = nn.Sequential(*layers)
        self.res = nn.Conv2d(c_out, 1, 1)

    def forward(self, x):
        x = self.down(x)
        x = torch.nn.functional.upsample(x, (256,256), mode='bilinear')
        x = self.res(x)
        return x




x = torch.rand(10,1,256,256)
y = torch.rand(10,1,256,256)
crit = nn.L1Loss()

for ndowns in range(1, 9):
    vx = Variable(x, requires_grad=True)
    vy = Variable(y)
    model = Encoder(ndowns)

    loss = crit(model(vx), vy)
    loss.backward()
    print('ndowns: %d, grad_norm: %f' % (ndowns, vx.grad.data.norm()))


