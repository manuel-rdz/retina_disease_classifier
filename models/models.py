from utils.metrics import get_scores, get_short_metrics_message
from models.utils import create_model, get_loss_function, get_optimizer, get_lr_scheduler
from torch_poly_lr_decay import PolynomialLRDecay

import pytorch_lightning as pl
import torch.optim as optim
import torch
import config
import os
import numpy as np


class RetinaClassifier(pl.LightningModule):
    def __init__(self, model_name, n_classes, input_size, loss='BCE', optimizer='', lr_scheduler='',requires_grad=False, lr=0.001,
                 threshold=0.0005, weights=None, output_path='', automatic_optimization=True):
        super().__init__()

        self.model = create_model(model_name, n_classes, input_size, True, requires_grad)

        self.lr = lr
        self.loss = get_loss_function(loss, weights)
        self.n_classes = n_classes
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.threshold = threshold

        self.predictions = np.empty((0, n_classes), dtype=np.float32)
        self.target = np.empty((0, n_classes), dtype=np.int16)

        self.best_model_score = 0.0
        self.output_path = output_path

        self.automatic_optimization = automatic_optimization

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = get_optimizer(self.optimizer, self.model.parameters(), self.lr)
        lr_scheduler = get_lr_scheduler(self.lr_scheduler, optimizer, 'avg_val_loss', self.threshold)

        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        x, y = batch

        b = x.size()
        x = x.view(b, -1)

        logits = self(x)

        # Required for BCEwithLogits to work
        y = y.type(torch.float16)

        if not self.automatic_optimization:
            return self.manual_training_step(logits, y)

        J = self.loss(logits, y)

        return {
            'loss': J}
        #    'train_acc': acc}
        #    'progress_bar': pbar}

    def manual_training_step(self, logits, y):
        opt = self.optimizers()
        sch = self.lr_schedulers()

        opt.zero_grad()

        J = self.loss(logits, y)

        self.manual_backward(J)
        opt.step()
        sch.step()

        return {
            'loss': J
        }

    def validation_step(self, batch, batch_idx):
        x, y = batch

        b = x.size()
        x = x.view(b, -1)

        preds = self(x)

        # Required for BCEwithLogits to work
        y = y.type(torch.float16)

        J = self.loss(preds, y)

        self.predictions = np.concatenate((self.predictions, preds.detach().cpu().numpy()), 0)
        self.target = np.concatenate((self.target, y.detach().cpu().numpy()), 0)

        # torch.sigmoid_(preds)

        # self.predictions = np.concatenate((self.predictions, preds.detach().cpu().numpy()), 0)

        # results = self.validation_step(batch, batch_idx)
        # results['test_acc'] = results['val_acc']
        # del results['val_acc']

        return {
            'val_loss': J,
        }

        # results = self.training_step(batch, batch_idx)
        # results['val_acc'] = results['train_acc']
        # del results['train_acc']
        # results['progress_bar']['val_acc'] = results['progress_bar']['train_acc']
        # del results['progress_bar']['train_acc']
        # return results

    def validation_epoch_end(self, val_step_outputs):
        avg_val_loss = torch.tensor([x['val_loss'] for x in val_step_outputs]).mean()

        self.log("avg_val_loss", avg_val_loss, on_epoch=True, prog_bar=True, logger=True)

        avg_metrics, _, _, _ = get_scores(self.target, self.predictions, config.normal_column_idx)

        model_score = ((avg_metrics[2] + avg_metrics[3]) / 2.0 + avg_metrics[0]) / 2.0

        if model_score > self.best_model_score:
            short_msg = get_short_metrics_message(*avg_metrics)
            print(short_msg)

            f = open(os.path.join(self.output_path, "final_scores.txt"), "w")
            f.write(short_msg)
            f.close()

            self.best_model_score = model_score

        # clear preds and target
        self.clean_metrics_arrays()

        return {'avg_val_loss': avg_val_loss}
        # 'avg_val_acc': avg_val_acc} #, 'progress_bar': pbar}

    def test_step(self, batch, batch_idx):
        x, y = batch

        b = x.size()
        x = x.view(b, -1)

        preds = self(x)

        # J = self.loss(preds, y)

        torch.sigmoid_(preds)

        self.predictions = np.concatenate((self.predictions, preds.detach().cpu().numpy()), 0)

        # results = self.validation_step(batch, batch_idx)
        # results['test_acc'] = results['val_acc']
        # del results['val_acc']

        # return results

    def test_epoch_end(self, test_step_outputs):
        pass
        # avg_test_loss = torch.tensor([x['loss'] for x in test_step_outputs]).mean()
        # avg_test_acc = torch.tensor([x['test_acc'] for x in test_step_outputs]).mean()

        # avg_metrics = {'avg_test_loss': avg_test_loss}
        # , 'avg_test_acc': avg_test_acc}

        # self.log_dict(avg_metrics, on_epoch=True, prog_bar=True, logger=True)

        # return avg_metrics

    def clean_metrics_arrays(self):
        self.predictions = np.empty((0, self.n_classes), dtype=np.float32)
        self.target = np.empty((0, self.n_classes), dtype=np.int16)
