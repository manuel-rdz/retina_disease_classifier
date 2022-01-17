from models.models import RetinaClassifier
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from data.modules import RetinaDataModule
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from resampling import utils as res_utils


import pandas as pd
import numpy as np
import argparse
import yaml
import os
import time
import random
import config


config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')

parser = argparse.ArgumentParser(description='Training args')

parser.add_argument('--train_data', help='path to train dataset')
parser.add_argument('--val_data', help='path to val dataset')
parser.add_argument('--train_imgs', nargs='+', help='path to train images')
parser.add_argument('--val_imgs', nargs='+', help='path to val images')
parser.add_argument('--test_imgs', help='path to test images')
parser.add_argument('--output_path', help='path to save trained models')
parser.add_argument('--batch_size', help='training batch size')
parser.add_argument('--lr', help='initial lr')
parser.add_argument('--num_classes', help='number of classes to predict')
parser.add_argument('--model', help='model to use')
parser.add_argument('--img_size', help='image size')
parser.add_argument('--start_col', help='column where labels start')
parser.add_argument('--seed', default=42, help='seed')
parser.add_argument('--num_workers', default=4, help='num workers for dataloader')
parser.add_argument('--gpus', default=1, help='number of gpus to use for training')
parser.add_argument('--auto_gpus', default=False, help='let pytorch check how many gpus there are available')
parser.add_argument('--pin_memory', default=False, help='pin memory for dataloader')
parser.add_argument('--epochs', default=50, help='number of epochs to run')
parser.add_argument('--auto_batch_size', default=False, help='use pl function to find automatically the biggest batch size possible')
parser.add_argument('--auto_lr', default=False, help='use pl function to find automtically the best initial lr')
parser.add_argument('--limit_train_batches', default=1.0, help='limit the number of batches to use during trainig')
parser.add_argument('--limit_val_batches', default=1.0, help='limit the number of validation batches to use during training')
parser.add_argument('--resampling', default='None', help='Name of the resampling algorithm to be applied on the dataset')
parser.add_argument('--resampling_percentage', default=10, help='Percentage of resampling the dataset')
parser.add_argument('--folds', default=5, help='Folds to train')
parser.add_argument('--loss', default='ASL', help='Loss function to use for the model training')
parser.add_argument('--optimizer', default='Adam', help='Optimizer to use during training')
parser.add_argument('--scheduler_threshold', default=1e-4, help='min threshold value for schedulers')

def _parse_args():
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text

def create_output_folder():
    # Added random number to avoid collisions duing running on server
    random.seed(time.perf_counter_ns())
    no_folder = str(random.randint(100000, 1000000))

    folder_name = time.strftime('%Y%m%d-%H%M%S') + '-' + args.model + '-' + no_folder
    output_path = os.path.join(args.output_path, folder_name)
    os.makedirs(output_path, exist_ok=True)
    return output_path

def get_class_weights(y_true):
    y_pos = np.sum(y_true, axis=0)
    weights = y_pos.max() / y_pos

    return np.array(weights)

def save_train_val_idxs(train_idx, val_idx, path):
    np.savetxt(os.path.join(path, 'val_idx.csv'),
    val_idx,
    delimiter=', ',
    fmt='% s')

    np.savetxt(os.path.join(path, 'train_idx.csv'),
    train_idx,
    delimiter=', ',
    fmt='% s')


def train_model(train_x, train_y, val_x, val_y, out_path):
    lr_monitor = LearningRateMonitor(
        logging_interval='step',
        log_momentum=True,
    )
    
    early_stopping = EarlyStopping(
        monitor='avg_val_loss', 
        patience=17, 
        verbose=True,
        min_delta=args.scheduler_threshold, 
        mode='min') 

    checkpoint = ModelCheckpoint(
        monitor="avg_val_loss",
        dirpath=out_path,
        filename=args.model + '-{epoch:02d}-{avg_val_loss:.3f}',
        save_top_k=1,
        mode="min",
    )

    model = RetinaClassifier(
        model_name=args.model, 
        n_classes=args.num_classes,
        input_size=args.img_size,
        lr=args.lr,
        loss=args.loss,
        optimizer=args.optimizer,
        threshold=args.scheduler_threshold,
        output_path=out_path,
        weights=get_class_weights(train_y)
    )

    trainer = Trainer(
        gpus=args.gpus,
        auto_select_gpus=args.auto_gpus,
        auto_scale_batch_size=args.auto_batch_size,
        auto_lr_find=args.auto_lr, 
        deterministic=True,
        precision=16, 
        max_epochs=args.epochs,
        callbacks=[checkpoint, lr_monitor, early_stopping], 
        limit_train_batches=args.limit_train_batches, 
        limit_val_batches=args.limit_val_batches)

    data_module = RetinaDataModule(
        df_train=train_x.join(train_y),
        df_val=val_x.join(val_y),
        train_img_path=args.train_imgs,
        val_img_path=args.val_imgs,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        start_col_labels=args.start_col,
    )

    if args.auto_lr:
        trainer.tune(model, datamodule=data_module)
        args.lr = model.lr
        args.auto_lr = False

    print('Using batch size: ', data_module.batch_size)
    print('Using learning rate: ', model.lr)

    trainer.fit(model, data_module)


if __name__ == '__main__':
    args, args_text = _parse_args()
    seed_everything(42, workers=True)

    output_dir = create_output_folder()

    # Save current args in the output dir
    args_file = open(os.path.join(output_dir, 'args.yaml'), 'w')
    args_file.write(args_text)
    args_file.close()
    
    if args.folds == 0:
        train_data = pd.DataFrame(np.empty(0))
        val_data = pd.DataFrame(np.empty(0))

        fold_path = os.path.join(output_dir, 'fold_0')
        os.mkdir(fold_path)

        if args.val_data is None:
            data = pd.read_csv(args.train_data)

            # limit data to only a subset of classes
            data = data.iloc[:, :args.start_col + args.num_classes]

            folds = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)

            for (train_idx, val_idx) in folds.split(data, data.iloc[:, args.start_col:]):
                train_data = data.iloc[train_idx, :]
                val_data = data.iloc[val_idx, :]
                break

            save_train_val_idxs(train_idx, val_idx, fold_path)
        else:
            train_data = pd.read_csv(args.train_data)
            val_data = pd.read_csv(args.val_data)

        # Create train and val datasets
        train_x = train_data.iloc[:, :args.start_col]
        train_y = train_data.iloc[:, args.start_col:]

        val_x = val_data.iloc[:, :args.start_col]
        val_y = val_data.iloc[:, args.start_col:]
        
        train_x, train_y = res_utils.resample_dataset(train_x, train_y, args.resampling, args.resampling_percentage)

        train_model(train_x, train_y, val_x, val_y, fold_path)    
    else:
        folds = MultilabelStratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

        data = pd.read_csv(args.train_data)

        for fold_i, (train_idx, val_idx) in enumerate(folds.split(data, data.iloc[:, args.start_col:])):
            fold_path = os.path.join(output_dir, 'fold_' + str(fold_i))
            os.mkdir(fold_path)

            # Create train and val datasets
            train_x = data.iloc[train_idx, :args.start_col]
            train_y = data.iloc[train_idx, args.start_col:]

            val_x = data.iloc[val_idx, :args.start_col]
            val_y = data.iloc[val_idx, args.start_col:]

            train_x, train_y = res_utils.resample_dataset(train_x, train_y, args.resampling, args.resampling_percentage)

            save_train_val_idxs(train_idx, val_idx, fold_path)

            train_model(train_x, train_y, val_x, val_y, fold_path)

# for checkpoints
# ! ls lightning_logs