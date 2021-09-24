from models import RetinaClassifier
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from data.modules import RetinaDataModule
from pytorch_lightning import Trainer, seed_everything

import pandas as pd
import argparse
import yaml


config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')

parser = argparse.ArgumentParser(description='Training args')

parser.add_argument('--data_dir', help='path to dataset')
parser.add_argument('--train_imgs', nargs='+', help='path to train images')
parser.add_argument('--val_imgs', nargs='+', help='path to val images')
parser.add_argument('--test_imgs', help='path to test images')
parser.add_argument('--batch_size', help='training batch size')
parser.add_argument('--num_classes', help='number of classes to predict')
parser.add_argument('--model', help='model to use')
parser.add_argument('--img_size', help='image size')
parser.add_argument('--start_col', help='column where labels start')
parser.add_argument('--seed', default=42, help='seed')
parser.add_argument('--num_workers', default=4, help='num workers for dataloader')
parser.add_argument('--pin_memory', default=False, help='pin memory for dataloader')
parser.add_argument('--epochs', default=50, help='number of epochs to run')

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


if __name__ == '__main__':
    args, args_text = _parse_args()
    seed_everything(42, workers=True)

    folds = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)

    data = pd.read_csv(args.data_dir)

    for fold_i, (train_idx, val_idx) in enumerate(folds.split(data, data.iloc[:, args.start_col:])):
        data_module = RetinaDataModule(
            df_train=data.iloc[train_idx, :],
            df_val=data.iloc[val_idx, :],
            train_img_path=args.train_imgs,
            val_img_path=args.val_imgs,
            img_size=args.img_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            )

        model = RetinaClassifier(args.model, args.num_classes)
        
        trainer = Trainer(gpus=1, deterministic=True, max_epochs=args.epochs, limit_train_batches=5, limit_val_batches=5)
        trainer.fit(model, data_module)


# for checkpoints
# ! ls lightning_logs