from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from models.models import RetinaClassifier

import cv2

import numpy as np


def reshape_transform(tensor, height=24, width=24):
    result = tensor[:, 1 :  , :].reshape(tensor.size(0),
        height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


image_path = 'C:\\Users\\AI\\Desktop\\student_Manuel\\datasets\\RIADD_cropped\\Training_Set\\Training\\64.png'
model_path = 'D:\\subsets_models\\20_labels\\20211128-002800-vit_base_patch16_384-347886\\fold_0\\vit_base_patch16_384-epoch=87-avg_val_loss=0.118.ckpt'
model_name = 'vit_base_patch16_384'
n_classes = 20

eigen_smooth = False
aug_smooth = False

model = RetinaClassifier.load_from_checkpoint(model_path, model_name=model_name, n_classes=n_classes)

vit = model.model

vit.eval()
vit.cuda()
print(vit)

target_layers = [vit.blocks[-1].norm1]
print(target_layers)

cam = GradCAM(model=vit, target_layers=target_layers, reshape_transform=reshape_transform, use_cuda=True)


rgb_img = cv2.imread(image_path, 1)[:, :, ::-1]
rgb_img = cv2.resize(rgb_img, (384, 384))
rgb_img = np.float32(rgb_img) / 255
input_tensor = preprocess_image(rgb_img, mean=[0.5, 0.5, 0.5],
                                    std=[0.5, 0.5, 0.5])

print(input_tensor.shape)

target_category = None

cam.batch_size = 16


grayscale_cam = cam(input_tensor=input_tensor,
                    target_category=target_category,
                    eigen_smooth=eigen_smooth,
                    aug_smooth=aug_smooth)

# Here grayscale_cam has only one image in the batch
grayscale_cam = grayscale_cam[0, :]
cam_image = show_cam_on_image(rgb_img, grayscale_cam)
cv2.imwrite('test_cam.jpg', cam_image)