import pytorch_lightning as pl
import pandas as pd
import argparse
import yaml
import numpy as np
import torch
import os
import glob
import config

from data.modules import RetinaDataModule
from models.models import RetinaClassifier
from utils.metrics import get_short_metrics_message, get_scores


config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')

parser = argparse.ArgumentParser(description='Training args')

parser.add_argument('--data_dir', help='path to dataset')
parser.add_argument('--test_imgs', nargs='+', help='path to test images')
parser.add_argument('--model_path', help='model to use for inference')
parser.add_argument('--batch_size', help='training batch size')
parser.add_argument('--num_classes', help='number of classes to predict')
parser.add_argument('--img_size', help='image size')
parser.add_argument('--start_col', help='column where labels start')
parser.add_argument('--seed', default=42, help='seed')
parser.add_argument('--num_workers', default=4, help='num workers for dataloader')
parser.add_argument('--pin_memory', default=False, help='pin memory for dataloader')
parser.add_argument('--tta', default=1, help='number of times to apply tta')
parser.add_argument('--model_name', help='model to load')
parser.add_argument('--output_path', help='path to output the generated csv')
parser.add_argument('--folds', help='number of folds to use for cross validation')
parser.add_argument('--gpus', default=1, help='number of gpus to use for evaluation')
parser.add_argument('--auto_gpus', default=False, help='let pythorch check how many gpus are available')
parser.add_argument('--transforms', default='riadd', help='testing transforms to apply')


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


def get_model(model_path, model_name, n_classes, input_size):
    if '.ckpt' in model_path:
        return RetinaClassifier.load_from_checkpoint(model_path, model_name=model_name, n_classes=n_classes, input_size=input_size)
    elif '.pth.tar' in model_path:
        retina_model = RetinaClassifier(model_name=model_name, n_classes=n_classes, input_size=input_size)

        state_dict = torch.load(model_path, map_location='cpu')
        retina_model.model.load_state_dict(state_dict["state_dict"], strict=True)

        return retina_model
    else:
        print('get_model::unknown model extension ', model_path)
        return None

    
def get_predictions(model, data_module, tta):
    trainer = pl.Trainer(gpus=args.gpus, auto_select_gpus=args.auto_gpus, deterministic=True, limit_test_batches=1.0, precision=16)

    if tta == 0:
        trainer.test(model, data_module)
        return model.predictions

    y_pred = np.zeros(0)

    for i in range(tta):
        trainer.test(model, data_module)

        if len(y_pred) == 0:
            y_pred = model.predictions
        else:
             y_pred = np.add(y_pred, model.predictions)

        model.clean_predictions()

    y_pred /= tta

    return y_pred


if __name__ == '__main__':
    args, args_text = _parse_args()

    pl.seed_everything(args.seed)

    #bin_auc, bin_f1, labels_auc, labels_map, labels_f1
    avg_metrics = np.zeros(5)
    scores_auc = scores_map = scores_f1 = np.zeros(args.num_classes)

    data = pd.read_csv(args.data_dir)
    #NORMAL_COL_IDX = list(data.columns.values).index('NORMAL') - args.start_col

    # limit data to only a subset of classes
    data = data.iloc[:, :args.start_col + args.num_classes]

    if args.folds > 0:
        y_pred = np.zeros((data.shape[0], args.num_classes))

        for fold in range(args.folds):
            model_path = glob.glob(os.path.join(args.model_path, 'fold_' + str(fold), '*.ckpt'))
            model = get_model(model_path[0], args.model_name, args.num_classes)
            
            val_idx = pd.read_csv(os.path.join(args.model_path, 'fold_' + str(fold) ,'val_idx.csv')).to_numpy(dtype=np.int32).squeeze()
            
            fold_y_true = data.iloc[val_idx, args.start_col:].to_numpy(dtype=np.float32)

            data_module = RetinaDataModule(
                df_test=data.iloc[val_idx],
                test_img_path=args.test_imgs,
                img_size=args.img_size,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=args.pin_memory,
                start_col_labels=args.start_col,
                stage='test',
                transforms=args.transforms,
            )

            fold_y_pred = get_predictions(model, data_module, args.tta)
            fold_avg_metrics, fold_scores_auc, fold_scores_map, fold_scores_f1 = get_scores(fold_y_true, fold_y_pred, config.normal_column_idx)

            avg_metrics = np.add(avg_metrics, fold_avg_metrics)
            scores_auc = np.add(scores_auc, fold_scores_auc)
            scores_map = np.add(scores_map, fold_scores_map)
            scores_f1 = np.add(scores_f1, fold_scores_f1)

            y_pred[val_idx] = fold_y_pred

        avg_metrics /= args.folds
        scores_auc /= args.folds
        scores_map /= args.folds
        scores_f1 /= args.folds

    else:
        if len(glob.glob(os.path.join(args.model_path, 'fold_0', '*.csv'))) > 0:
            test_idx = pd.read_csv(os.path.join(args.model_path, 'fold_0','val_idx.csv'), header=None).to_numpy(dtype=np.int32).squeeze()
        else:
            test_idx = np.arange(data.shape[0])

        y_true = data.iloc[test_idx, args.start_col:].to_numpy(dtype=np.float32)

        data_module = RetinaDataModule(
            df_test=data.iloc[test_idx],
            test_img_path=args.test_imgs,
            img_size=args.img_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            start_col_labels=args.start_col,
            use_tta=(args.tta > 0),
            stage='test',
        )

        model_path = glob.glob(os.path.join(args.model_path, 'fold_0', '*.ckpt'))
        if len(model_path) == 0:
            model_path = glob.glob(os.path.join(args.model_path, 'fold_0', '*.pth.tar'))

        model = get_model(model_path[0], args.model_name, args.num_classes, args.img_size)
        y_pred = get_predictions(model, data_module, args.tta)
    
        avg_metrics, scores_auc, scores_map, scores_f1 = get_scores(y_true, y_pred, config.normal_column_idx)

    message = get_short_metrics_message(*avg_metrics)

    f = open(os.path.join(args.output_path, "final_scores.txt"), "w")
    f.write(message)
    f.close()

    print(message)

    np.savetxt(os.path.join(args.output_path, 'preds.csv'), 
        y_pred,
        delimiter =", ", 
        fmt ='% s')

    np.savetxt(os.path.join(args.output_path, 'scores.csv'),
        np.column_stack((scores_auc, scores_map, scores_f1)),
        header='auc, map, f1',
        delimiter=', ',
        fmt='% s')