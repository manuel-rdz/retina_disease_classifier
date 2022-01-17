import torch
import timm
import torch.nn as nn
import numpy as np
import torch.optim as op

from loss.AsymetricLoss import AsymmetricLossOptimized
from optimizer.ranger21 import Ranger21
from pytorch_ranger import Ranger


def create_model(model_name, n_classes, input_size, pretrained=True, requires_grad=False):

    if any(name in model_name for name in ['beit', 'vgg16', 'resnet101']):
        model = timm.create_model(model_name, pretrained=pretrained)
    else:
        model = timm.create_model(model_name, pretrained=pretrained, img_size=input_size)

    for param in model.parameters():
        param.requires_grad = requires_grad
    
    if 'distilled' in model_name:
        model.head = nn.Linear(model.head.in_features, n_classes)
        model.head_dist = nn.Linear(model.head_dist.in_features, n_classes)
    elif any(name in model_name for name in ['vit', 'xcit', 'deit', 'beit', 'swin']):
        model.head = nn.Linear(model.head.in_features, n_classes)
    elif 'efficientnet' in model_name:
        model.classifier = nn.Linear(model.classifier.in_features, n_classes)
    elif 'vgg16' in model_name:
        model.head.fc = nn.Linear(model.head.fc.in_features, n_classes)
    elif 'resnet101' in model_name:
        model.fc = nn.Linear(model.fc.in_features, n_classes)

    return model


def get_loss_function(loss, weights=[]):
    if loss == 'ASL':
        return AsymmetricLossOptimized(gamma_neg=2, gamma_pos=1)
    if loss == 'BCE':
        return nn.BCEWithLogitsLoss()
    if loss == 'WBCE':
        if len(weights) == 0:
            weights = np.ones(34)
        return nn.BCEWithLogitsLoss(pos_weight=torch.from_numpy(weights))

    return None


def get_optimizer(optimizer, params, lr):
    if optimizer == 'Adam':
        return op.Adam(params, lr)
    if optimizer == 'SGD':
        return op.SGD(params, lr)
    if optimizer == 'Ranger':
        return Ranger(params, lr)
    if optimizer == 'Ranger21':
        return Ranger21(params, lr, num_batches_per_epoch=163, num_epochs=100, use_madgrad=True)
    