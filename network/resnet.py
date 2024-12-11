import torch.nn as nn
from torchvision import models


class ResNet(nn.Module):
    def __init__(self, net_name, weights):
        super(ResNet, self).__init__()

        self.resnet = getattr(models,net_name)(weights=weights)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])


    def forward(self, x):
        x = self.resnet(x)
        x = x.flatten(start_dim=1)
        return x


if __name__ == '__main__':
    resnet18 = ResNet('resnet50', weights=None)
    print(resnet18)
    import torch
    inputs = torch.randn(32, 3, 1024, 512)
    print(resnet18(inputs).shape)
    # print(resnet18)