import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ConvNeXt_Tiny_Weights


class ConvNeXt(nn.Module):
    def __init__(self, net_name, weights):
        super(ConvNeXt, self).__init__()
        self.convnext = models.__dict__[net_name](weights=weights)
        self.features = nn.Sequential(*(list(self.convnext.children())[:-1]))

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(start_dim=1)
        return x


if __name__ == '__main__':

    convnext_tiny = ConvNeXt('convnext_tiny', ConvNeXt_Tiny_Weights.DEFAULT)
    inputs = torch.randn(32, 3, 224, 224)
    print(convnext_tiny(inputs).shape)
