---
data_dir: /home/kunet.ae/100058256/datasets/RIADD_cropped/Training_Set/RFMiD_Training_Labels.csv
train_imgs:
- /home/kunet.ae/100058256/datasets/RIADD_cropped/Training_Set/Training
val_imgs:
- /home/kunet.ae/100058256/datasets/RIADD_cropped/Training_Set/Training
test_imgs:
output_path: /home/kunet.ae/100058256/codes/trained_models/RIADD/
batch_size: 8
lr: 0.0003630780
num_classes: 29
model: vit_large_patch16_384
#efficientnet_b0
#vit_base_patch16_224
img_size: 384
start_col: 1
experiment:
seed: 42
num_workers: 4
pin_memory: True
epochs: 50
auto_batch_size: False
auto_lr: False
limit_train_batches: 1.0
limit_val_batches: 1.0

