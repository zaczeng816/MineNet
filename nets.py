import torch
import torch.nn as nn
from torchvision.models import resnet50, densenet121, densenet169, densenet201, efficientnet_b0, efficientnet_b7

class VisionTransformer(nn.Module):
    def __init__(self, num_classes, num_channels, image_size, patch_size):
        super(VisionTransformer, self).__init__()
        pass

    def forward(self, x):
        pass

class SwinTransformer(nn.Module):
    def __init__(self, num_classes, num_channels, image_size, patch_size):
        super(SwinTransformer, self).__init__()
        pass

    def forward(self, x):
        pass

class ConvNeXt(nn.Module):
    def __init__(self, num_classes, num_channels):
        super(ConvNeXt, self).__init__()
        pass

    def forward(self, x):
        pass

class ResNet50(nn.Module):
    def __init__(self, num_classes, num_channels):
        super(ResNet50, self).__init__()
        self.model = resnet50(weights=None)  # Don't load pre-trained weights
        self.model.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)

class DenseNet121(nn.Module):
    def __init__(self, num_classes, num_channels):
        super(DenseNet121, self).__init__()
        self.model = densenet121(weights=None)
        self.model.features.conv0 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_ftrs = self.model.classifier.in_features
        self.model.classifier = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)

class DenseNet169(nn.Module):
    def __init__(self, num_classes, num_channels):
        super(DenseNet169, self).__init__()
        self.model = densenet169(weights=None)
        self.model.features.conv0 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_ftrs = self.model.classifier.in_features
        self.model.classifier = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)

class DenseNet201(nn.Module):
    def __init__(self, num_classes, num_channels):
        super(DenseNet201, self).__init__()
        self.model = densenet201(weights=None)
        self.model.features.conv0 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_ftrs = self.model.classifier.in_features
        self.model.classifier = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)

class EfficientNetB0(nn.Module):
    def __init__(self, num_classes, num_channels):
        super(EfficientNetB0, self).__init__()
        self.model = efficientnet_b0(weights=None)
        self.model.features[0][0] = nn.Conv2d(num_channels, 32, kernel_size=3, stride=2, padding=1, bias=False)
        num_ftrs = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)

class EfficientNetB7(nn.Module):
    # out of memory error
    def __init__(self, num_classes, num_channels):
        super(EfficientNetB7, self).__init__()
        self.model = efficientnet_b7(weights=None)
        self.model.features[0][0] = nn.Conv2d(num_channels, 64, kernel_size=3, stride=2, padding=1, bias=False)
        num_ftrs = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)