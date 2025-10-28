import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights

class SimpleCNN(nn.Module):
    """
    A simple CNN architecture for image classification
    """

    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        # Convolutional layers: progressively increase number of filters (3 -> 32 -> 64 -> 128)
        # 3x3 kernels with padding=1 maintain spatial dimensions before pooling
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)  # Additional layer for CIFAR-100
        self.pool = nn.MaxPool2d(2, 2)  # 2x2 pooling reduces spatial dimensions by half
        # Fully connected layers: flatten feature maps and classify
        self.fc1 = nn.Linear(256 * 2 * 2, 512)  # Adjusted for additional conv layer
        self.fc2 = nn.Linear(512, 256)  # Additional FC layer for better representation
        self.fc3 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)  # Dropout for regularization
        self.batch_norm1 = nn.BatchNorm2d(32)  # Batch normalization for better training
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.batch_norm3 = nn.BatchNorm2d(128)
        self.batch_norm4 = nn.BatchNorm2d(256)

    def forward(self, x):
        x = self.pool(F.relu(self.batch_norm1(self.conv1(x))))
        x = self.pool(F.relu(self.batch_norm2(self.conv2(x))))
        x = self.pool(F.relu(self.batch_norm3(self.conv3(x))))
        x = self.pool(F.relu(self.batch_norm4(self.conv4(x))))
        x = x.view(-1, 256 * 2 * 2)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


class ResNetForCIFAR(nn.Module):
    """
    Modified ResNet architecture for CIFAR datasets
    """

    def __init__(self, num_classes=10, resnet_type='resnet18'):
        super(ResNetForCIFAR, self).__init__()

        # 使用新的weights参数替代pretrained
        if resnet_type == 'resnet18':
            # 不使用预训练权重
            self.resnet = models.resnet18(weights=None)
        elif resnet_type == 'resnet34':
            self.resnet = models.resnet34(weights=None)
        elif resnet_type == 'resnet50':
            self.resnet = models.resnet50(weights=None)
        else:
            raise ValueError(f"Unsupported ResNet type: {resnet_type}")

        # Modify first layer for CIFAR (32x32 images instead of 224x224)
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet.maxpool = nn.Identity()  # Remove maxpool for CIFAR

        # Replace final classifier
        if resnet_type in ['resnet18', 'resnet34']:
            num_features = self.resnet.fc.in_features
        else:  # resnet50
            num_features = self.resnet.fc.in_features

        self.resnet.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.resnet(x)


def create_model(num_classes, device, model_type='simple'):
    """
    Create and initialize the model

    Args:
        num_classes: Number of classes (10 for CIFAR-10, 100 for CIFAR-100)
        device: Device to put model on
        model_type: Type of model ('simple', 'resnet18', 'resnet34', 'resnet50')
    """
    if model_type == 'simple':
        model = SimpleCNN(num_classes=num_classes)
    elif model_type.startswith('resnet'):
        model = ResNetForCIFAR(num_classes=num_classes, resnet_type=model_type)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    model = model.to(device)
    return model


def get_model_parameters(model):
    """Get total number of parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
