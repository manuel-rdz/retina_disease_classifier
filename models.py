from optimizers.AsymetricLoss import AsymmetricLossOptimized

import pytorch_lightning as pl
import torch.optim as optim
import torchmetrics as tm
import torch.nn as nn
import torch
import timm

class RetinaClassifier(pl.LightningModule):
    def __init__(self, model_name='vit', n_classes=29, requires_grad=False):
        super().__init__()

        self.model = timm.create_model(model_name, pretrained = True)
        
        #model = models.resnet50(progress=True, pretrained=pretrained)
        # to freeze the hidden layers
        if requires_grad == False:
            for param in self.model.parameters():
                param.requires_grad = False
        # to train the hidden layers
        elif requires_grad == True:
            for param in self.model.parameters():
                param.requires_grad = True
        # make the classification layer learnable
        
        # Add last layer that contains the amount of classes to predict
        self.model.head = nn.Linear(self.model.head.in_features, n_classes)
        # we have 25 classes in total
        #model.fc = nn.Linear(2048, n_classes)
        #self.base_model = model

        self.loss = AsymmetricLossOptimized(gamma_neg=2, gamma_pos=1)
        self.n_classes = n_classes

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch

        b = x.size()
        x = x.view(b, -1)

        logits = self(x)

        J = self.loss(logits, y)

        torch.sigmoid_(logits)
        #auc_score = tm.functional.auroc(logits, y, num_classes=self.n_classes)

        #auc_score = 0
        #for i in range(logits.size()[1]):
        #   auc_score += tm.functional.auroc(logits[:, i], y[:, i])

        #auc_score /= logits.size()[1]
        #acc = tm.functional.auc(torch.sigmoid(logits), y)

        acc = tm.functional.accuracy(logits, y)

        pbar = {'train_acc': acc}
        print('Training acc: ', acc)
        
        return {
            'loss': J,
            'train_acc': acc,
            'progress_bar': pbar}
    
    def validation_step(self, batch, batch_idx):
        results = self.training_step(batch, batch_idx)
        results['progress_bar']['val_acc'] = results['progress_bar']['train_acc']
        del results['progress_bar']['train_acc']
        return results

    def validation_epoch_end(self, val_step_outputs):
        avg_val_loss = torch.tensor([x['loss'] for x in val_step_outputs]).mean()
        avg_val_acc = torch.tensor([x['progress_bar']['val_acc'] for x in val_step_outputs]).mean()

        print('val auc score')
        print(avg_val_acc)

        pbar = {'avg_val_acc': avg_val_acc}

        return {'val_loss': avg_val_loss, 'progress_bar': pbar}

    def test_step(self, batch, batch_idx):
        print(self.validation_step(batch, batch_idx))