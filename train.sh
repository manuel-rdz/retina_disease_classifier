#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --job-name=train
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --account=kuex0005
#SBATCH --output=train.%j.out
#SBATCH --error=train.%j.err

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

python train.py -c args/server/all_train_args.yaml
