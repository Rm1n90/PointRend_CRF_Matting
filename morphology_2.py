import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import kornia
import time
import numpy as np


def dilation(img: torch.Tensor, structuring_element: torch.Tensor):

    if not torch.is_tensor(img):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(img)}")
    if not torch.is_tensor(structuring_element):
        raise TypeError(f"Structuring element type is not a torch.Tensor. Got {type(structuring_element)}")
    img_shape = img.shape
    if not (len(img_shape) == 3 or len(img_shape) == 4):
        raise ValueError(f"Expected input tensor to be of ndim 3 or 4, but got {len(img_shape)}")
    if len(img_shape) == 3:
        # unsqueeze introduces a batch dimension
        img = img.unsqueeze(0)
    else:
        if(img_shape[1] != 1):
            raise ValueError(f"Expected a single channel image, but got {img_shape[1]} channels")
    if len(structuring_element.shape) != 2:
        raise ValueError(
            f"Expected structuring element tensor to be of ndim=2, but got {len(structuring_element.shape)}")

    # Check if the input image is a binary containing only 0, 1
    unique_vals = torch.unique(img)

    if len(unique_vals) > 2:
        raise ValueError(
            f"Expected only 2 unique values in the tensor, since it should be binary, but got {len(torch.unique(img))}")
    if not ((unique_vals == 0.0) + (unique_vals == 1.0)).all():
        raise ValueError("Expected image to contain only 1's and 0's since it should be a binary image")

        # Convert structuring_element from shape [a, b] to [1, 1, a, b]
    structuring_element = structuring_element[None, None, ...]

    se_shape = structuring_element.shape
    conv1 = F.conv2d(img, structuring_element, padding=(se_shape[2] // 2, se_shape[2] // 2))
    convert_to_binary = (conv1 > 0).float()

    if len(img_shape) == 3:
        # If the input ndim was 3, then remove the fake batch dim introduced to do conv
        return torch.squeeze(convert_to_binary, 0)
    else:
        return convert_to_binary

img = cv2.imread('auxiliar/mask_234.png')[..., 0] # 0 - 255
image = torch.tensor(img)[None, None,...].cuda()
invert = torch.clamp(~image, 0, 1).float()
kernel = torch.ones((21, 21), dtype=torch.uint8).cuda()
kernel = kernel.float()

start = time.time()
torch.cuda.synchronize()
dilation_image = dilation(invert, kernel)
torch.cuda.synchronize()
end = time.time()
print(end - start)

dilated_image_np= kornia.tensor_to_image(dilation_image)
dilated_image_np = dilated_image_np.astype(np.uint8)
# import ipdb;ipdb.set_trace()
cv2.imwrite('auxiliar/disldas.png', ~(255 * dilated_image_np))


