import torch
import torch.nn as nn
from torchvision.models import resnet50, densenet121, densenet169, densenet201, efficientnet_b0, efficientnet_b7, vit_b_16, swin_t, convnext_tiny
import sys
import os

# Add the parent directory to the system path
sys.path.append("Vim/vim")
from models_mamba import VisionMamba, PatchEmbed

class VisionMambaNet(nn.Module):
    def __init__(self, num_classes, num_channels):
        super(VisionMambaNet, self).__init__()
        self.model = VisionMamba( num_classes=num_classes,
            img_size=512,
            channels=num_channels,
            patch_size=16, stride=8, embed_dim=384, depth=24,
            rms_norm=True, residual_in_fp32=True, fused_add_norm=True, final_pool_type='mean', 
            if_abs_pos_embed=True, if_rope=False, if_rope_residual=False, bimamba_type="v2", if_cls_token=True, if_devide_out=True, use_middle_cls_token=True
        )

        self.model.patch_embed.proj = nn.Conv2d(num_channels, 384, kernel_size=16, stride=8)
        
        self.model.head = nn.Sequential(
            nn.Linear(384, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.model(x)



class VisionTransformer(nn.Module):
    def __init__(self, num_classes, num_channels):
        super(VisionTransformer, self).__init__()
        self.model = vit_b_16(weights=None, image_size=512)
        self.model.conv_proj = nn.Conv2d(num_channels, 768, kernel_size=16, stride=16)
        self.model.heads = nn.Linear(768, num_classes)

    def forward(self, x):
        return self.model(x)

class SwinTransformer(nn.Module):
    def __init__(self, num_classes, num_channels):
        super(SwinTransformer, self).__init__()
        self.model = swin_t(weights=None)
        self.model.features[0][0] = nn.Conv2d(num_channels, 96, kernel_size=4, stride=4)
        self.model.head = nn.Linear(768, num_classes)

    def forward(self, x):
        return self.model(x)

class ConvNeXt(nn.Module):
    def __init__(self, num_classes, num_channels):
        super(ConvNeXt, self).__init__()
        self.model = convnext_tiny(weights=None)
        self.model.features[0][0] = nn.Conv2d(num_channels, 96, kernel_size=4, stride=4)
        self.model.classifier[2] = nn.Linear(768, num_classes)

    def forward(self, x):
        return self.model(x)

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