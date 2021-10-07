import pytorch_lightning as pl
import pandas as pd
import argparse
import yaml
import numpy as np
import torch
import os
import glob

from data.modules import RetinaDataModule
from models.models import RetinaClassifier
from utils.metrics import auc_score, mAP_score


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


def get_model(model_path, model_name, n_classes):
    if '.ckpt' in model_path:
        return RetinaClassifier.load_from_checkpoint(model_path, model_name=model_name, n_classes=n_classes)
    elif '.pth.tar' in model_path:
        retina_model = RetinaClassifier(model_name=model_name, n_classes=n_classes)

        state_dict = torch.load(model_path, map_location='cpu')
        retina_model.model.load_state_dict(state_dict["state_dict"], strict=True)

        return retina_model
    else:
        print('unknown extension ', model_path)
        return None

    
def get_predictions(model, data_module, tta):
    trainer = pl.Trainer(gpus=1, deterministic=True, limit_test_batches=1.0, precision=16)

    y_pred = np.zeros(0)

    for i in range(tta):
        trainer.test(model, data_module)
        
        #auc_avg_score, auc_scores = auc_score(y_true, model.predictions)
        #map_avg_score, map_scores = mAP_score(y_true, model.predictions)

        #print('AUC_scores:')
        #print(auc_avg_score, auc_scores)
        #print('mAP scores:')
        #print(map_avg_score, map_scores)

        if len(y_pred) == 0:
            y_pred = model.predictions
        else:
             y_pred = np.add(y_pred, model.predictions)

        model.clean_predictions()

    y_pred /= tta

    return y_pred


def get_scores(y_true, y_pred):
    auc_bin, scores_auc = auc_score(y_true[:, 0], y_pred[:, 0])
    map_bin, scores_map = mAP_score(y_true[:, 0], y_pred[:, 0])
    
    auc, scores_auc = auc_score(y_true[:, 1:], y_pred[:, 1:])
    mAP, scores_mAP = mAP_score(y_true[:, 1:], y_pred[:, 1:])
    task2_score = (auc + mAP) / 2

    final_score = (auc_bin + task2_score) / 2

    msg = '----- Multilabel scores -----\n'
    msg += 'auc_score: {}\n'.format(auc)
    msg += 'mAP: {}\n'.format(mAP)
    msg += 'task score: {}\n'.format(task2_score)
    msg += '----- Binary scores -----\n'
    msg += 'auc: {}\n'.format(auc_bin)
    msg += 'mAP: {}\n'.format(map_bin)
    msg += '----- Final Score -----\n'
    msg += str(final_score)

    scores_auc.insert(0, auc_bin)
    scores_mAP.insert(0, map_bin)

    return msg, scores_auc, scores_mAP


if __name__ == '__main__':
    args, args_text = _parse_args()

    pl.seed_everything(args.seed)

    data = pd.read_csv(args.data_dir)

    data_module = RetinaDataModule(
        df_test=data,
        test_img_path=args.test_imgs,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        stage='test',
    )

    y_true = data.iloc[:, args.start_col:].to_numpy(dtype=np.float32)
    y_pred = np.zeros(y_true.shape)

    if args.folds > 0:
        for fold in range(args.folds):
            model_path = glob.glob(os.path.join(args.model_path, 'fold_' + str(fold), '*.ckpt'))
            print(model_path)
            model = get_model(model_path[0], args.model_name, args.num_classes)
            val_idx = pd.read_csv(os.path.join(args.model_path, 'fold_' + str(fold) ,'val_idx.csv')).to_numpy(dtype=np.int32)
            fold_pred = get_predictions(model, data_module, args.tta)

            y_pred[val_idx, :] = fold_pred[val_idx, :]
    else:
        model = get_model(args.model_path, args.model_name, args.num_classes)
        y_pred = get_predictions(model, data_module, args.tta)
    
    msg, scores_auc, scores_map = get_scores(y_true, y_pred)

    np.savetxt(os.path.join(args.output_path, 'preds.csv'), 
        y_pred,
        delimiter =", ", 
        fmt ='% s')

    np.savetxt(os.path.join(args.output_path, 'scores.csv'),
        np.column_stack((np.array(scores_auc), np.array(scores_map))),
        header='auc, map',
        delimiter=', ',
        fmt='% s')

    f = open(os.path.join(args.output_path, "final_scores.txt"), "w")
    f.write(msg)
    f.close()

    print(msg)