import pytorch_lightning as pl
import pandas as pd
import argparse
import yaml
import numpy as np
import torch
import os

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
        return RetinaClassifier.load_from_checkpoint(model_path, model_name=model_name, num_classes=n_classes)
    elif '.pth.tar' in model_path:
        retina_model = RetinaClassifier(model_name=model_name, n_classes=n_classes)

        state_dict = torch.load(model_path, map_location='cpu')
        retina_model.model.load_state_dict(state_dict["state_dict"], strict=True)

        return retina_model
    else:
        print('unknown extension ', model_path)
        return None


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

    model = get_model(args.model_path, args.model_name, args.num_classes)
    
    trainer = pl.Trainer(gpus=1, deterministic=True, limit_test_batches=1.0, precision=16)

    y_true = data.iloc[:, args.start_col:].to_numpy(dtype=np.float32)
    y_pred = np.empty((0))

    for i in range(args.tta):
        trainer.test(model, data_module)
        
        auc_avg_score, auc_scores = auc_score(y_true, model.predictions)
        map_avg_score, map_scores = mAP_score(y_true, model.predictions)

        print('AUC_scores:')
        print(auc_avg_score, auc_scores)
        print('mAP scores:')
        print(map_avg_score, map_scores)

        if len(y_pred) == 0:
            y_pred = model.predictions
        else:
             y_pred = np.add(y_pred, model.predictions)

        model.clean_predictions()

    y_pred /= args.tta
    
    auc_bin, scores_auc = auc_score(y_true[:, 0], y_pred[:, 0])
    map_bin, scores_map = mAP_score(y_true[:, 0], y_pred[:, 0])
    
    auc, scores_auc = auc_score(y_true[:, 1:], y_pred[:, 1:])
    mAP, scores_mAP = mAP_score(y_true[:, 1:], y_pred[:, 1:])
    task2_score = (auc + mAP) / 2

    final_score = (auc_bin + task2_score) / 2
    print('----- Multilabel scores -----')
    print('auc_score: ', auc)
    print('mAP: ', mAP)
    print('task score: ', task2_score)
    print('----- Binary scores -----')
    print('auc: ', auc_bin)
    print('mAP: ', map_bin)
    print('----- Final Score -----')
    print(final_score)

    scores_auc.insert(0, auc_bin)
    scores_mAP.insert(0, map_bin)

    np.savetxt(os.path.join(args.output_path, 'auc_scores.csv'), 
        scores_auc,
        delimiter =", ", 
        fmt ='% s')

    np.savetxt(os.path.join(args.output_path, 'map_scores.csv'), 
        scores_mAP,
        delimiter =", ", 
        fmt ='% s')
