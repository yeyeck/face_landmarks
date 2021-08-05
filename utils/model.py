import torch
from torch import nn
from torch.nn.modules import loss
from torch.nn.modules.activation import Hardswish
from torchvision import models
from torchvision.models.mobilenetv3 import ConvBNActivation, InvertedResidual, InvertedResidualConfig
from functools import partial


class Model(nn.Module):
    def __init__(self, out_features=196, width_mult=1.0):
        super(Model, self).__init__()
        norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)
        invert_res = partial(InvertedResidual, norm_layer= norm_layer)
        invert_res_config = partial(InvertedResidualConfig, width_mult=width_mult)
        # main net
        self.convba_0 = ConvBNActivation(3, 16, stride=2, norm_layer=norm_layer, activation_layer=nn.Hardswish)
        self.inver_1 = invert_res(invert_res_config(16, 3, 16, 16, True, 'RE', 2, 1))
        self.inver_2 = invert_res(invert_res_config(16, 3, 72, 24, False, "RE", 1, 1))
        self.inver_3 = invert_res(invert_res_config(24, 3, 88, 24, False, "RE", 1, 1))    # c1
        self.inver_4 = invert_res(invert_res_config(24, 5, 96, 40, True, "HS", 2, 1))
        self.inver_5 = invert_res(invert_res_config(40, 5, 240, 40, True, "HS", 1, 1))
        self.inver_6 = invert_res(invert_res_config(40, 5, 240, 40, True, "HS", 1, 1))
        self.inver_7 = invert_res(invert_res_config(40, 5, 120, 48, True, "HS", 1, 1))
        self.inver_8 = invert_res(invert_res_config(48, 5, 144, 48, True, "HS", 1, 1))
        self.inver_9 = invert_res(invert_res_config(48, 5, 288, 96, True, 'RE', 2, 1))
        self.inver_10 = invert_res(invert_res_config(96, 5, 576, 96, True, 'HS', 1, 1))
        self.inver_11 = invert_res(invert_res_config(96, 5, 576, 96, True, 'HS', 1, 1))
        self.convba_12 = ConvBNActivation(96, 256, kernel_size=1, stride=1, norm_layer=norm_layer)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.header = nn.Sequential(
            nn.Linear(256, 512),
            nn.Hardswish(),
            nn.Linear(512, out_features=out_features)
        )

        self.auxiliarynet = AuxiliaryNet()
        
    def forward(self, x):
        x = self.convba_0(x)
        x = self.inver_1(x)
        x = self.inver_2(x)
        x = self.inver_3(x)
        a = self.auxiliarynet(x)
        x = self.inver_4(x)
        x = self.inver_5(x)
        x = self.inver_6(x)
        x = self.inver_7(x)
        x = self.inver_8(x)
        x = self.inver_9(x)
        x = self.inver_10(x)
        x = self.inver_11(x)
        x = self.convba_12(x)
        x = self.avgpool(x)
        x = x.view(-1, 256)
        x = self.header(x)
        return torch.cat((x, a), 1)

class AuxiliaryNet(nn.Module):
    def __init__(self):
        super(AuxiliaryNet, self).__init__()
        norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)
        self.conv0 = ConvBNActivation(24, 64, stride=2, norm_layer=norm_layer)
        self.conv1 = ConvBNActivation(64, 64, norm_layer=norm_layer)
        self.conv2 = ConvBNActivation(64, 32, stride=2, norm_layer=norm_layer)
        self.conv3 = ConvBNActivation(32, 64, norm_layer=norm_layer)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(64, 16)
        self.fc2 = nn.Linear(16, 3)
    
    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.avgpool(x)
        x = x.view(-1, 64)
        x = self.fc1(x)
        x = self.fc2(x)
        return x



def create_model(output_features=196):
    return Model(output_features)


if __name__ == '__main__':
    model = Model(width_mult=1.2)
    print(model)