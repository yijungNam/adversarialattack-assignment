import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


def get_cifar10_pretrained_model():

    model = resnet18(weights=ResNet18_Weights.DEFAULT)


    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 10)

    return model