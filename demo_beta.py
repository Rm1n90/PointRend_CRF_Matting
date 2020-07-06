# You may need to restart your runtime prior to this, to let your installation take effect
# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
import torch
import os
import time
from collections import deque
import shutil

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
import point_rend
import torch.nn.functional as F

cfg = get_cfg()
# Add PointRend-specific config
point_rend.add_pointrend_config(cfg)
# Load a config from file
cfg.merge_from_file("/home/media4us/PycharmProjects/pointrend_with_refinator/configs/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2  # set threshold for this model
# Use a model from PointRend model zoo: https://github.com/facebookresearch/detectron2/tree/master/projects/PointRend#pretrained-models
cfg.MODEL.WEIGHTS = "/home/media4us/PycharmProjects/pointrend_with_refinator/weights/model_final_3c3198.pkl"
predictor = DefaultPredictor(cfg)

# CRF
from convcrf import do_crf_inference, GaussCRF, default_conf
config = default_conf
config['filter_size'] = 21
config['pyinn'] = False
config['col_feats']['schan'] = 0.05

# FAMatting
from fa_matting.networks.transforms import trimap_transform, groupnorm_normalise_image
from fa_matting.networks.models import build_model
import argparse


def config_fa():
    parser = argparse.ArgumentParser()
    # Model related arguments
    parser.add_argument('--encoder', default='resnet50_GN_WS', help="encoder model")
    parser.add_argument('--decoder', default='fba_decoder', help="Decoder model")
    parser.add_argument('--weights', default='/home/media4us/PycharmProjects/pointrend_with_refinator/weights/FBA.pth')
    args = parser.parse_args()
    return args

# IndexMatting
from indexMatting.hlmobilenetv2 import hlmobilenetv2
from PIL import Image
from collections import OrderedDict
import cv2 as cv
import torch
import torch.nn as nn

RESTORE_FROM = '/home/media4us/PycharmProjects/pointrend_with_refinator/weights/indexnet_matting.pth.tar'


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
STRIDE = 32

# if not os.path.exists(RESULT_DIR):
#     os.makedirs(RESULT_DIR)

# load pretrained model
net_index = hlmobilenetv2(
    pretrained=True,
    freeze_bn=True,
    output_stride=STRIDE,
    apply_aspp=True,
    conv_operator='std_conv',
    decoder='indexnet',
    decoder_kernel_size=5,
    indexnet='depthwise',
    index_mode='m2o',
    use_nonlinear=True,
    use_context=True
)
try:
    checkpoint = torch.load(RESTORE_FROM, map_location=device)
    pretrained_dict = OrderedDict()
    for key, value in checkpoint['state_dict'].items():
        if 'module' in key:
            key = key[7:]
        pretrained_dict[key] = value
except:
    raise Exception('Please download the pretrained model!')
net_index.load_state_dict(pretrained_dict)
net_index.to(device)
net_index.eval()
INDEX_IMG_MEAN = torch.tensor([0.485, 0.456, 0.406, 0]).view((1, 4, 1, 1)).to(device)
INDEX_IMG_STD = torch.tensor([0.229, 0.224, 0.225, 1]).view((1, 4, 1, 1)).to(device)

import math
def size_align(imsize, output_stride):
    return [int(math.ceil(axis / output_stride) * output_stride) for axis in imsize]

def image_alignment(x, output_stride, odd=False):
    imsize = np.asarray(x.shape[:2], dtype=np.float)
    if odd:
        new_imsize = np.ceil(imsize / output_stride) * output_stride + 1
    else:
        new_imsize = np.ceil(imsize / output_stride) * output_stride
    h, w = int(new_imsize[0]), int(new_imsize[1])

    x1 = x[:, :, 0:3]
    x2 = x[:, :, 3]
    new_x1 = cv2.resize(x1, dsize=(w, h), interpolation=cv.INTER_CUBIC)
    new_x2 = cv2.resize(x2, dsize=(w, h), interpolation=cv.INTER_NEAREST)

    new_x2 = np.expand_dims(new_x2, axis=2)
    new_x = np.concatenate((new_x1, new_x2), axis=2)

    return new_x

def inference_indexMatting(imgs, trimaps):
    with torch.no_grad():
        # pre
        #a0_pre = time.time()
        trimap = trimaps.unsqueeze(0).unsqueeze(0).to(device)
        image = imgs.to(device)

        image = torch.cat((image, trimap), dim=1)
        h, w = image.shape[2:]
        image_batch = (image - INDEX_IMG_MEAN) / INDEX_IMG_STD
        image_batch = F.interpolate(image_batch, size=size_align((INFER_MATTING_HEIGHT, INFER_MATTING_WIDTH), STRIDE), mode='nearest')
        #a1_pre = time.time()

        # inference
        #a0_infer = time.time()
        torch.cuda.synchronize()
        outputs = net_index(image_batch)
        torch.cuda.synchronize()
        #a1_infer = time.time()

        # post
        #a0_post = time.time()
        outputs = F.interpolate(outputs, size=(h, w), mode='nearest')
        alpha = torch.clamp(outputs, 0, 1).squeeze()
        trimap = trimap.squeeze()
        mask = (trimap == 0.5).float()
        alpha = (1 - mask) * trimap + mask * alpha



        #a1_post = time.time()

        #print('time:', a1_pre - a0_pre, a1_infer - a0_infer, a1_post - a0_post)


        return alpha.cpu().numpy()

def trimap_approx(trimap_np):
    dilate_trimap_np = cv2.dilate(np.uint8(trimap_np > 0), np.ones((15, 15), dtype=np.uint8)) > 0

    # wrong detection simulation
    erode_trimap_np = cv2.erode(np.uint8(trimap_np > 0), np.ones((3, 3), dtype=np.uint8)) > 0

    out_trimap = np.float32(dilate_trimap_np)
    in_trimap = np.float32(erode_trimap_np)

    out_trimap = 0.5 * out_trimap
    out_trimap[in_trimap > 0] = 1

    return out_trimap

def get_fa_mask(crf_mask):
    # Split in fg & bg
    h, w = crf_mask.shape
    trimap = np.zeros((h, w, 2))
    trimap[crf_mask == 1, 1] = 1
    trimap[crf_mask == 0, 0] = 1

    return trimap_approx(trimap)

def scale_input_fa(x: np.ndarray, scale: float, scale_type) -> np.ndarray:
    ''' Scales inputs to multiple of 8. '''
    h, w = x.shape[:2]
    h1 = int(np.ceil(scale * h / 8) * 8)
    w1 = int(np.ceil(scale * w / 8) * 8)

    ratio = w1 / h1
    if h1 < w1:
        h1 = max(8 * 30, h1)
        w1 = h1 * ratio
    else:
        w1 = max(8 * 30, w1)
        h1 = w1 / ratio

    h1 = int(np.ceil(h1 / 8) * 8)
    w1 = int(np.ceil(w1 / 8) * 8)

    x_scale = cv2.resize(x, (w1, h1), interpolation=scale_type)
    return x_scale

def np_to_torch(x):
    return torch.from_numpy(x).permute(2, 0, 1)[None, :, :, :].float().cuda()

def pred_fa(image_np: np.ndarray, trimap_np: np.ndarray) -> np.ndarray:
    ''' Predict alpha, foreground and background.
        Parameters:
        image_np -- the image in rgb format between 0 and 1. Dimensions: (h, w, 3)
        trimap_np -- two channel trimap, first background then foreground. Dimensions: (h, w, 2)
        Returns:
        fg: foreground image in rgb format between 0 and 1. Dimensions: (h, w, 3)
        bg: background image in rgb format between 0 and 1. Dimensions: (h, w, 3)
        alpha: alpha matte image between 0 and 1. Dimensions: (h, w)
    '''

    model_fa = build_model(config_fa())
    model_fa.eval()

    h, w = trimap_np.shape[:2]

    image_scale_np = scale_input_fa(image_np, 1.0, cv2.INTER_LANCZOS4)
    trimap_scale_np = scale_input_fa(trimap_np, 1.0, cv2.INTER_LANCZOS4)

    with torch.no_grad():
        image_torch = np_to_torch(image_scale_np)
        trimap_torch = np_to_torch(trimap_scale_np)

        trimap_transformed_torch = np_to_torch(trimap_transform(trimap_scale_np))
        image_transformed_torch = groupnorm_normalise_image(image_torch.clone(), format='nchw')
        output = model_fa(image_torch, trimap_torch, image_transformed_torch, trimap_transformed_torch)
        output = cv2.resize(output[0].cpu().numpy().transpose((1, 2, 0)), (w, h), cv2.INTER_LANCZOS4)
    alpha = output[:, :, 0]
    #fg = output[:, :, 1:4]
    #bg = output[:, :, 4:7]

    alpha[trimap_np[:, :, 0] == 1] = 0
    alpha[trimap_np[:, :, 1] == 1] = 1
    #fg[alpha == 1] = image_np[alpha == 1]
    #bg[alpha == 0] = image_np[alpha == 0]
    return alpha


cnt = 0

def inference(im):
    global cnt
    cnt += 1

    im_norm = im / 255.
    image = torch.tensor(im_norm).to(device).permute(2, 0, 1).unsqueeze(0).float()

    # point rend
    t0 = time.time()
    torch.cuda.synchronize()
    outputs = predictor(im)
    torch.cuda.synchronize()

    if len(outputs['instances']) == 0:
        return np.zeros(im.shape[1:], dtype=np.float32)

    instances = outputs['instances'].get_fields()
    out = torch.zeros(instances['pred_masks'].shape[1:]).bool().cuda()
    for i in range(len(outputs['instances'])):
        if instances['pred_classes'][i] == 0:
            out |= instances['pred_masks'][i]
    out = out.float()
    t1 = time.time()
    print('time PointRend:', 1000 * (t1 - t0))

    # CRF
    out_refine_crf = torch.stack((1 - out, out), dim=0).unsqueeze(0)

    torch.cuda.synchronize()
    out_refine_crf = gausscrf(unary=out_refine_crf, img=(image - 0.5) / 0.3, num_iter=8)
    torch.cuda.synchronize()

    _, out_refine_crf = torch.max(out_refine_crf[0], dim=0)
    out_refine_crf_np = out_refine_crf.cpu().numpy()

    t2 = time.time()
    print('time CRF:', 1000 * (t2 - t1))

    # KNN
    #out_knn_np = bs.apply(im)
    #out_knn_np = np.float32(out_knn_np > 0)

    # Get the trimap
    trimap_np = trimap_approx(out_refine_crf_np)
    # out_refine_crf_np = cv2.erode(np.uint8(out_refine_crf_np > 0), np.ones((3, 3), dtype=np.uint8)) > 0
    # trimap_np = (out_knn_np + out_refine_crf_np) / 2.
    # trimap_np_05 = cv2.dilate(np.uint8(trimap_np == 0.5), np.ones((15, 15), dtype=np.uint8)) > 0
    # trimap_np_new = np.zeros_like(trimap_np, dtype=np.float32)
    # trimap_np_new[trimap_np == 1] = 1.
    # trimap_np_new[trimap_np_05] = 0.5
    # trimap_np = trimap_np_new

    trimap = torch.from_numpy(trimap_np).to(device)

    # indexMatting
    out_refine_in = inference_indexMatting(image, trimap)

    # FAMatting
    # out_refine_fa = pred_fa(im_norm, trimap_np, model_fa)

    t3 = time.time()
    print('time Matting:', 1000 * (t3 - t2))

    # Mask = pointRend + CRF + alphaMatting
    mask_full = out_refine_in

    # Mask = PointRend + CRF
    # mask_full = out_refine_crf_np

    montage = np.concatenate((
        im,
        np.tile(np.expand_dims(255 * out.cpu().numpy(), axis=-1), [1, 1, 3]),
        np.tile(np.expand_dims(255 * out_refine_crf_np, axis=-1), [1, 1, 3]),
        np.tile(np.expand_dims(255 * out_refine_in, axis=-1), [1, 1, 3]),
    ), axis=1)
    cv2.imwrite(f'montages/montage{cnt}.png', montage)

    return mask_full

cap = cv2.VideoCapture('/home/media4us/video.mp4')

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

INFER_WIDTH = width // 2
INFER_HEIGHT = height // 2

INFER_MATTING_WIDTH = INFER_WIDTH
INFER_MATTING_HEIGHT = INFER_HEIGHT

gausscrf = GaussCRF(conf=config, shape=(INFER_HEIGHT, INFER_WIDTH), nclasses=2).cuda()
fps = cap.get(cv2.CAP_PROP_FPS)

# class NaiveBS:
#     def __init__(self, images):
#         self.im_mean = np.int16(np.median(np.stack(images, axis=0), axis=0))
#
#     def __call__(self, x):
#         x = (np.abs(np.int16(x) - self.im_mean) >= 15).any(axis=-1)


# Get webcam
images = deque(maxlen=int(fps * 300))
i = 0
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break
    images.append(frame)

    # Display the resulting frame
    cv2.imshow('frame', frame )
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
    i += 1
    print(i, cv2.CAP_PROP_FPS)

#bs = NaiveBS(images)
bs = cv2.createBackgroundSubtractorMOG2(detectShadows=True, varThreshold=50, history=100)

# Inference images
if os.path.isdir('montages'):
    shutil.rmtree('montages')
os.makedirs('montages', exist_ok=True)

if os.path.isdir('frames'):
    shutil.rmtree('frames')
os.makedirs('frames', exist_ok=True)

for idx, im in enumerate(images):
    a0 = time.time()
    im_resize = cv2.resize(im, (INFER_WIDTH, INFER_HEIGHT))
    mask_resize = inference(im_resize)
    mask = cv2.resize(mask_resize, (im.shape[1], im.shape[0]), interpolation=cv2.INTER_NEAREST)

    mask_im = np.uint8(255 * mask)
    a1 = time.time()
    print(f'Elapsed time: {1000 * (a1 - a0)}')

    cv2.imwrite(f'frames/im_{idx}.jpg', im)
    cv2.imwrite(f'frames/mask_{idx}.png', mask_im)

    montage = np.concatenate((im, np.tile(np.expand_dims(mask_im, axis=-1), [1, 1, 3])), axis=1)
    cv2.imwrite(f'frames/montage_{idx}.png', montage)