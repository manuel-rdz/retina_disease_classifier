import timm
import torch.nn as nn


def create_model(model_name, n_classes, pretrained=True, requires_grad=False):
    model = timm.create_model(model_name, pretrained=pretrained)

    for param in model.parameters():
        param.requires_grad = requires_grad
    
    if 'vit' in model_name:
        model.head = nn.Linear(model.head.in_features, n_classes)
    elif 'efficientnet' in model_name:
        model.classifier = nn.Linear(model.classifier.in_features, n_classes)

    return model
