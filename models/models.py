from models.utils import create_model, get_loss_function, get_optimizer

import pytorch_lightning as pl
import torch.optim as optim
import torchmetrics as tm
import torch.nn as nn
import torch
import timm
import numpy as np


class RetinaClassifier(pl.LightningModule):
    def __init__(self, model_name, n_classes, loss='', optimizer='', requires_grad=False, lr=0.001, weights=[]):
        super().__init__()

        self.model = create_model(model_name, n_classes, True, requires_grad)

        self.lr = lr
        self.loss = get_loss_function(loss, weights)
        self.n_classes = n_classes
        self.optimizer = optimizer

        self.predictions = np.empty((0, n_classes), dtype=np.float32)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = get_optimizer(self.optimizer, self.model.parameters(), self.lr)
        lr_scheduler = {
            'scheduler': optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True),
            'monitor': 'avg_val_loss'}
        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        x, y = batch

        b = x.size()
        x = x.view(b, -1)

        logits = self(x)


        # Required for BCEwithLogits to work
        y = y.type(torch.float16)

        J = self.loss(logits, y)

        #torch.sigmoid_(logits)
        #auc_score = tm.functional.auroc(logits, y, num_classes=self.n_classes)

        #auc_score = 0
        #for i in range(logits.size()[1]):
        #   auc_score += tm.functional.auroc(logits[:, i], y[:, i])

        #auc_score /= logits.size()[1]
        #acc = tm.functional.auc(torch.sigmoid(logits), y)

        #acc = tm.functional.accuracy(logits, y)

        #self.log("train_loss", J, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        #self.log('acc', acc, on_step=True, on_epoch=True, prog_bar=True, #logger=True)  


        #pbar = {'train_acc': acc}
        
        return {
            'loss': J}
        #    'train_acc': acc}
        #    'progress_bar': pbar}
    
    def validation_step(self, batch, batch_idx):
        x, y = batch

        b = x.size()
        x = x.view(b, -1)

        preds = self(x)

        # Required for BCEwithLogits to work
        y = y.type(torch.float16)

        J = self.loss(preds, y)

        #torch.sigmoid_(preds)

        #self.predictions = np.concatenate((self.predictions, preds.detach().cpu().numpy()), 0)

        #results = self.validation_step(batch, batch_idx)
        #results['test_acc'] = results['val_acc']
        #del results['val_acc']

        return {
            'val_loss': J,
        }
        
        
        #results = self.training_step(batch, batch_idx)
        #results['val_acc'] = results['train_acc']
        #del results['train_acc']
        #results['progress_bar']['val_acc'] = results['progress_bar']['train_acc']
        #del results['progress_bar']['train_acc']
        #return results

    def validation_epoch_end(self, val_step_outputs):
        avg_val_loss = torch.tensor([x['val_loss'] for x in val_step_outputs]).mean()
        #avg_val_acc = torch.tensor([x['val_acc'] for x in val_step_outputs]).mean()

        #print('val auc score')
        #print(avg_val_acc)

        self.log("avg_val_loss", avg_val_loss, on_epoch=True, prog_bar=True, logger=True)
        #self.log('avg_val_acc', avg_val_acc, on_epoch=True, prog_bar=True, logger=True)  

        #pbar = {'avg_val_acc': avg_val_acc}

        return {'avg_val_loss': avg_val_loss}
                #'avg_val_acc': avg_val_acc} #, 'progress_bar': pbar}

    def test_step(self, batch, batch_idx):
        x, y = batch

        b = x.size()
        x = x.view(b, -1)

        preds = self(x)

        #J = self.loss(preds, y)

        torch.sigmoid_(preds)

        self.predictions = np.concatenate((self.predictions, preds.detach().cpu().numpy()), 0)

        #results = self.validation_step(batch, batch_idx)
        #results['test_acc'] = results['val_acc']
        #del results['val_acc']

        #return results

    def test_epoch_end(self, test_step_outputs):
        pass
        #avg_test_loss = torch.tensor([x['loss'] for x in test_step_outputs]).mean()
        #avg_test_acc = torch.tensor([x['test_acc'] for x in test_step_outputs]).mean()

        #avg_metrics = {'avg_test_loss': avg_test_loss}
        #, 'avg_test_acc': avg_test_acc}

        #self.log_dict(avg_metrics, on_epoch=True, prog_bar=True, logger=True)

        #return avg_metrics

    def clean_predictions(self):
        self.predictions = np.empty((0, self.n_classes), dtype=np.float32)