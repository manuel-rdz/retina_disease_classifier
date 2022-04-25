import torch

from models.models import RetinaClassifier

from pytorch_grad_cam import ScoreCAM, GradCAM, GradCAMPlusPlus, EigenCAM, EigenGradCAM, AblationCAM
from pytorch_grad_cam.utils.image import preprocess_image,show_cam_on_image

import cv2

import numpy as np


def reshape_transform(tensor, height=24, width=24):
    print('to_reshape', tensor.shape)
    result = tensor[:, 1 :  , :].reshape(tensor.size(0),
        height, width, tensor.size(2))
    print('middle step', result.shape)

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    print('reshape_result', result.shape)
    return result


image_path = 'C:\\Users\\AI\\Desktop\\student_Manuel\\datasets\\RIADD_cropped\\Training_Set\\Training\\777.png'
#image_path = 'C:\\Users\\AI\\Desktop\\student_Manuel\\datasets\\ARIA\\all_images_crop\\aria_a_11_6.tif'
#image_path = 'C:\\Users\\AI\Desktop\\student_Manuel\\datasets\\STARE\\all_images_crop\\im0045.png'
model_path = 'D:\\subsets_models\\20_labels\\20211128-002800-vit_base_patch16_384-347886\\fold_0\\vit_base_patch16_384-epoch=87-avg_val_loss=0.118.ckpt'
model_name = 'vit_base_patch16_384'
n_classes = 20

model = RetinaClassifier.load_from_checkpoint(model_path, model_name=model_name, n_classes=n_classes, requires_grad=True, input_size=384)

vit = model.model

vit.eval()
vit.cuda()
print(vit)

target_layers = [vit.blocks[-1].norm1]
print(target_layers)

from pytorch_grad_cam.ablation_layer import AblationLayerVit

cam = ScoreCAM(model=model,
                target_layers=target_layers,
                use_cuda=True,
                reshape_transform=reshape_transform)
                #ablation_layer=AblationLayerVit)


rgb_img = cv2.imread(image_path, 1)[:, :, ::-1]
rgb_img = cv2.resize(rgb_img, (384, 384))
rgb_img = np.float32(rgb_img) / 255
input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

targets = None

cam.batch_size = 32

grayscale_cam = cam(input_tensor=input_tensor,
                    targets=targets,
                    eigen_smooth=False,
                    aug_smooth=False)

# Here grayscale_cam has only one image in the batch
grayscale_cam = grayscale_cam[0, :]
grayscale_cam = cv2.bitwise_not(grayscale_cam)
cam_image = show_cam_on_image(rgb_img, grayscale_cam)
cv2.imwrite('vit_eigencam_11.jpg', cam_image)

preds = torch.sigmoid_(model(input_tensor.cuda()))
print(preds)