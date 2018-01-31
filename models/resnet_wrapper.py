# borrow from pytorch

import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from torchvision.models.resnet import BasicBlock, Bottleneck, model_urls
import math



def create_resnet_conv_layers(network = 'resnet18', input_nc = 3, pretrained = False):
    '''
    Create resnet (without last fc layer) with arbitrary input channel number.
    '''

    if network == 'resnet18':
        model = ResNetConv(BasicBlock, [2,2,2,2], input_nc)
    elif network == 'resnet34':
        model = ResNetConv(BasicBlock, [3,4,6,3], input_nc)
    elif network == 'resnet50':
        model = ResNetConv(Bottleneck, [3,4,6,3], input_nc)
    elif network == 'resnet101':
        model = ResNetConv(Bottleneck, [3,4,23,3], input_nc)
    elif network == 'resnet152':
        model = ResNetConv(Bottleneck, [3,8,36,3], input_nc)
    else:
        raise ValueError('cannot recognize "%s" for creaing resnet conv layers' % network)

    if pretrained:
        if input_nc == 3:
            param = model_zoo.load_url(model_urls[network])
            param.pop('fc.weight')
            param.pop('fc.bias')
            model.load_state_dict(param)
        else:
            raise ValueError('only support loading pretrained weights when input_nc == 3 (RGB input)')

        return model



class ResNetConv(nn.Module):
    '''
    ResNet with modifications:
        - Remove the last fc layer
        - Output is a feature map, where each pixel is a self.output_nc dimension vector.
        - Support arbitrary input channel (but only support ImageNet-pratrain weights when input_nc = 3)
    '''
    def __init__(self, block, layers, input_nc):

        self.inplanes = 64
        super(ResNetConv, self).__init__()
        self.conv1 = nn.Conv2d(input_nc, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # self.avgpool = nn.AvgPool2d(7, stride=1)
        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.output_nc = 512 * block.expansion


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def extract_feat(self, x):
        output = {'f_0': x}
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        output['f_1'] = x;

        x = self.layer1(x)
        output['f_2'] = x
        x = self.layer2(x)
        output['f_3'] = x
        x = self.layer3(x)
        output['f_4'] = x
        x = self.layer4(x)
        output['f_5'] = x
        return output


