from . import resnet,vit,convnext,swin_transformer
import torch.nn as nn
import torchvision.models as models

backbone_dict = {
    'resnet18': [resnet.ResNet('resnet18', weights=None), 512],
    'resnet34': [resnet.ResNet('resnet34', weights=None), 512],
    'resnet50': [resnet.ResNet('resnet50', weights=None), 2048],

    'convnext_tiny':[convnext.ConvNeXt('convnext_tiny',weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1),768],

    'vit_b': [vit.vit_base(), 768],
    'vit_s': [vit.vit_small(), 384],
    'swin_transformer': [swin_transformer.swin_tiny_patch4_window7_224(pretrain=False),768]
}


class Projector(nn.Module):
    def __init__(self, input_dim, hidden_size, feature_dim):
        super(Projector, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, feature_dim)
        )

    def forward(self, x):
        return self.layers(x)


class Classifier(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        return self.fc(x)

