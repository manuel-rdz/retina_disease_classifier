#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --job-name=train_resnet
#SBATCH --time=03:00:00
#SBATCH --partition=gpu
#SBATCH --account=kuex0005
#SBATCH --output=train_resnet.%j.out
#SBATCH --error=train_resnet.%j.err

module purge
module load gcc/9.3
module load python/3.9.6
module load miniconda/3
module load cuda/11.3

pip install pillow
pip install numpy
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install -U scikit-learn
pip install visdom
pip install pandas
pip install tensorboard
pip install -U albumentations
pip install pytorch-lightning
pip install scikit-multilearn
pip install pytorch-ranger
pip install -U iterative-stratification
pip install git+https://github.com/cmpark0126/pytorch-polynomial-lr-decay.git

python train.py -c args/server/all_train_args.yaml