"""
Contains PyTorch model code to instantiate a TinyVGG model.
"""

import torch

from torch import nn 
from torchvision import models

class DenseNet(nn.Module):
    """
    Transfer learning using densenet121 architecture.
    See the documentation here: https://pytorch.org/vision/main/models/generated/torchvision.models.densenet121.html
    """
    def __init__(self):
        super().__init__()
        self.name = 'DenseNet'
        
        Dense = models.densenet121(weights='DEFAULT')
        # Freeze all base layers in the "features" section of the model (the feature extractor) by setting requires_grad=False
        for param in Dense.features.parameters():
            param.requires_grad = False
        
        # Recreate the classifier layer and seed it to the target device
        Dense.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.2, inplace=True), 
            torch.nn.Linear(in_features=1024, 
                            out_features=10, # same number of output units as our number of classes
                            bias=True)
        )
        
        self.net = Dense

    def forward(self, x):
        return self.net(x)

class EfficientNet_B0(nn.Module):
    """
    Transfer learning using EfficientNet_B0 architecture.
    See the documentation here: https://pytorch.org/vision/main/models/generated/torchvision.models.efficientnet_b0.html
    """
    def __init__(self):
        super().__init__()
        self.name = 'EfficientNet_B0'
        
        EfficientNet_B0_weights = models.EfficientNet_B0_Weights.DEFAULT 
        EfficientNet_B0 = models.efficientnet_b0(weights=EfficientNet_B0_weights)
        # Freeze all base layers in the "features" section of the model (the feature extractor) by setting requires_grad=False
        for param in EfficientNet_B0.features.parameters():
            param.requires_grad = False
        
        # Recreate the classifier layer and seed it to the target device
        EfficientNet_B0.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.2, inplace=True), 
            torch.nn.Linear(in_features=1280, out_features=10)
        )
        
        self.net = EfficientNet_B0

    def forward(self, x):
        return self.net(x)

class MobileNet_V3(nn.Module):
    """
    Transfer learning using large MobileNetV3 architecture.
    See the documentation here: https://pytorch.org/vision/main/models/generated/torchvision.models.mobilenet_v3_large.html
    """
    def __init__(self):
        super().__init__()
        self.name = 'MobileNet_V3'
        
        MobileNet_V3 = models.mobilenet_v3_large(weights='DEFAULT')
        # Freeze all base layers in the "features" section of the model (the feature extractor) by setting requires_grad=False
        for param in MobileNet_V3.features.parameters():
            param.requires_grad = False
        
        MobileNet_V3.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.2, inplace=True), 
            torch.nn.Linear(in_features=960, 
                            out_features=10, # same number of output units as our number of classes
                            bias=True)
        )
        
        self.net = MobileNet_V3

    def forward(self, x):
        return self.net(x)