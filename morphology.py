import torch
import kornia
import cv2
import ipdb
from torch.nn import functional as f

import torch.nn.functional as F
class Morphology:
    def __init__(self, img: torch.Tensor,device=torch.device('cuda:0')):
        self.img = img
        self.device = device

    def dilation(self, kernel: torch.Tensor):

        img_pad = torch.nn.ConstantPad2d(
            (kernel.shape[0] // 2, kernel.shape[0] // 2, kernel.shape[1] // 2, kernel.shape[1] // 2), 0)(self.img)
        windows = f.unfold(img_pad, kernel_size=kernel.shape)
        st_elem_tmp = kernel.flatten().unsqueeze(0).unsqueeze(-1)
        max_kernel = kernel.max()
        processed = windows.add(st_elem_tmp).max(dim=1, keepdims=True)[0] - max_kernel
        out = f.fold(processed, self.img.shape[-2:], kernel_size=1)

        if len(self.img.shape) == 3:
            return torch.squeeze(out, 0)
        else:
            return out

    def erosion(self, kernel: torch.Tensor):

        img_pad = torch.nn.ConstantPad2d(
            (kernel.shape[0] // 2, kernel.shape[0] // 2, kernel.shape[1] // 2, kernel.shape[1] // 2), 0)(self.img)
        windows = f.unfold(img_pad, kernel_size=kernel.shape)
        st_elem_tmp = kernel.flatten().unsqueeze(0).unsqueeze(-1)
        min_kernel = kernel.min()
        processed = windows.add(st_elem_tmp).min(dim=1, keepdims=True)[0] - min_kernel
        out = f.fold(processed, self.img.shape[-2:], kernel_size=1)

        if len(self.img.shape) == 3:
            return torch.squeeze(out, 0)
        else:
            return out


import time

img = cv2.imread('auxiliar/mask_234.png')[..., 0] # 0 - 255
image = torch.tensor(img, dtype=torch.float)[None, None,...].cuda() # 0 - 255
kernel = torch.ones((21, 21), dtype=torch.float).cuda()
morphology = Morphology(image)
for i in range(10):
    start = time.time()
    torch.cuda.synchronize()
    # dilation_image = morphology.dilation(kernel)
    eroson_image = morphology.erosion(kernel) # 0 - 255
    torch.cuda.synchronize()
    end = time.time()
    print(end - start)

# ipdb.set_trace()
# dilated_image_np= kornia.tensor_to_image(dilation_image)
eroson_image_np= kornia.tensor_to_image(eroson_image)
# cv2.imwrite('auxiliar/dilasd.png', dilated_image_np)
cv2.imwrite('auxiliar/eros.png', eroson_image_np)