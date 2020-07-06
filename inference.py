# import some common libraries
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import os
import time
from collections import deque, OrderedDict
from PIL import Image
import shutil
import math
from collections import namedtuple
import ipdb # TODO Remove

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
# import sys; sys.path.insert(1, "pointrend_with_refinator/point_rend")
import point_rend
import torch.nn.functional as f
setup_logger()

# CRF
from convcrf import do_crf_inference, GaussCRF, default_conf

# FAMatting
from fa_matting.networks.transforms import trimap_transform, groupnorm_normalise_image
from fa_matting.networks.models import build_model
import argparse

# IndexMatting
from indexMatting.hlmobilenetv2 import hlmobilenetv2

# GPU morphology
from morphology import Morphology

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def time_decor(func):
    def __func(*args, **kwargs):
        a0 = time.time()
        aux = func(*args, **kwargs)
        torch.cuda.synchronize()
        a1 = time.time()
        print(f"{func.__name__}: {1000 * (a1 - a0)}")
        return aux
    return __func


def config():
    parser = argparse.ArgumentParser()
    # Main config
    parser.add_argument('--video', default='/home/media4us/video.mp4',
                        type=str, help='Path to the video or number to use webcam')
    parser.add_argument('--background', default='/home/media4us/Pictures/rocket.jpg',
                        type=str, help='Path to background')

    parser.add_argument('--matting', choices=['index', 'FA'], default='index',
                        type=str, help='specify matting model')


    # Detectron2 config
    parser.add_argument('--detWeight', default='./weights/model_final_3c3198.pkl',
                        type=str, help='Path to Detectron2 pretrained')
    parser.add_argument('--detConfig',
                        default='./configs/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco.yaml',
                        type=str, help='Path to Detectron2 config file')
    parser.add_argument('--detScore', default=0.2,
                        type=float, help='Confidence score')

    # CRF config
    parser.add_argument('--crfKernel', default= 21,
                        type=float, help='CRF filter size')
    parser.add_argument('--crfFeats', default=0.05,
                        type=float, help='CRF column features')
    parser.add_argument('--crfPyinn', default=False,
                        type=bool, help='CRF for old python')

    # FAMatting config
    parser.add_argument('--faEncoder', default='resnet50_GN_WS',
                        type=str, help="FAMatting encoder model")
    parser.add_argument('--faDecoder', default='fba_decoder',
                        type=str, help="FAMatting Decoder model")
    parser.add_argument('--faWeights', default='weights/FBA.pth',
                        type=str, help='Path to FAMatting weight')

    # IndexMatting config
    parser.add_argument('--inWeight', default='./weights/indexnet_matting.pth.tar',
                        type=str, help='Path to IndexMatting pretrained')
    parser.add_argument('--inStride', default=32,
                        type=int, help='Index Matting Stride')
    cfg = parser.parse_args()
    return cfg

# _____________________________________PointRend____________________________#

class PointRendSegmentation:
    def __init__(self, detection_config, weights_path, score=0.5, device=torch.device('cuda:0')):
        self.device = device
        cfg = get_cfg()
        point_rend.add_pointrend_config(cfg)
        cfg.merge_from_file(detection_config)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score
        cfg.MODEL.WEIGHTS = weights_path
        self.predictor = DefaultPredictor(cfg)


    def predict(self, img):
        with torch.no_grad():
            torch.cuda.synchronize()
            outputs = self.predictor(img)
            if len(outputs['instances']) == 0:
                return np.zeros(img.shape[1:], dtype=np.float32)
            instances = outputs['instances'].get_fields()
            out = torch.zeros(instances['pred_masks'].shape[1:]).bool().cuda()
            for i in range(len(outputs['instances'])):
                if instances['pred_classes'][i] == 0:
                    out |= instances['pred_masks'][i]
            out = out.float()
            return out


# _____________________________________PalomosPose____________________________#
# import cv2
# import math
# import time
# import numpy as np
# from config_reader import config_reader
# from keras.models import load_model
# from model_simulated_RGB101_cdcl_pascal import get_testing_model_resnet101
# from human_seg.pascal_voc_human_seg_gt_7parts import human_seg_combine_argmax_rgb
#
# import keras
# import tensorflow as tf
# gpu_options = tf.GPUOptions(allow_growth=True)
# gpu_options.per_process_gpu_memory_fraction = 0.5
# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
# keras.backend.tensorflow_backend.set_session(sess)

# class PalomosPose:
#     def __init__(self, img_shape, weights_path):
#         self.pad_value = 80
#         self.img_shape = [int(0.25*s) for s in img_shape]
#         self.model = get_testing_model_resnet101()
#         self.model.load_weights(weights_path)
#         self.params, self.model_params = config_reader()
#
#     def recover_flipping_output(self, oriImg, part_ori_size):
#         part_ori_size = part_ori_size[:, ::-1, :]
#         part_flip_size = np.zeros((oriImg.shape[0], oriImg.shape[1], 7))
#         part_flip_size[:, :, :] = part_ori_size[:, :, :]
#         return part_flip_size
#
#     def predict(self, im):
#         oriImg = im
#         oriImg = (oriImg / 256.0) - 0.5
#
#         scale = 1
#         imageToTest = cv2.resize(oriImg, (self.img_shape[1], self.img_shape[0]), interpolation=cv2.INTER_CUBIC)
#
#         pad = [0,
#                0,
#                (imageToTest.shape[0] - self.model_params['stride']) % self.model_params['stride'],
#                (imageToTest.shape[1] - self.model_params['stride']) % self.model_params['stride']
#                ]
#
#         imageToTest_padded = np.pad(imageToTest, ((0, pad[2]), (0, pad[3]), (0, 0)), mode='constant',
#                                     constant_values=((0, 0), (0, 0), (0, 0)))
#
#         input_img = imageToTest[np.newaxis, ...]
#
#         print("\t[Original] Actual size fed into NN: ", input_img.shape)
#
#         output_blobs = self.model.predict(input_img)
#
#         seg = np.squeeze(output_blobs[0])
#         seg = cv2.resize(seg, (0, 0), fx=self.model_params['stride'], fy=self.model_params['stride'],
#                          interpolation=cv2.INTER_CUBIC)
#         seg = seg[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
#         seg = cv2.resize(seg, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)
#
#         seg_argmax = np.argmax(seg, axis=-1)
#         seg_max = np.max(seg, axis=-1)
#         seg_max_thres = (seg_max > 0.1).astype(np.uint8)
#         seg_argmax *= seg_max_thres
#         return torch.from_numpy(seg_argmax > 0).float().to('cuda:0')
#
#
#
#
#         # oriImg = im
#         # #flipImg = cv2.flip(oriImg, 1)
#         # oriImg = (oriImg / 256.0) - 0.5
#         #
#         # imageToTest = cv2.resize(oriImg, (self.img_shape[1], self.img_shape[0]), interpolation=cv2.INTER_CUBIC)
#         #
#         # pad = [
#         #     0,
#         #     0,
#         #     (imageToTest.shape[0] - self.model_params['stride']) % self.model_params['stride'],
#         #     (imageToTest.shape[1] - self.model_params['stride']) % self.model_params['stride']
#         # ]
#         # imageToTest_padded = np.pad(imageToTest, ((0, pad[2]), (0, pad[3]), (0, 0)), mode='constant',
#         #                             constant_values=0)
#         # input_img = imageToTest[np.newaxis, ...]
#         # output_blobs = self.model.predict(input_img)
#         # seg = np.squeeze(output_blobs[0])
#         # seg = cv2.resize(seg, (0, 0), fx=self.model_params['stride'], fy=self.model_params['stride'], interpolation=cv2.INTER_CUBIC)
#         # seg = seg[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
#         # seg = cv2.resize(seg, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)
#         #
#         # seg_argmax = np.argmax(seg, axis=-1)
#         # #seg_max = np.max(seg, axis=-1)
#         # #seg_max_thres = (seg_max > 0.1).astype(np.uint8)
#         # #seg_argmax *= seg_max_thres
#         #
#         # return torch.from_numpy(seg_argmax > 0).float().to('cuda:0')








# _____________________________________CRF___________________________________#

class CRF:
    def __init__(self, img_shape: (int, int), filter_size: int, col_feats: float, device=torch.device('cuda:0')):
        config = default_conf
        config['filter_size'] = filter_size
        config['pyinn'] = False
        config['col_feats']['schan'] = col_feats
        self.device = device
        self.gausscrf = GaussCRF(conf=config, shape=(img_shape[0], img_shape[1]), nclasses=2).to(device)

    def refine_mask(self, img, seg_mask):
        out_refine_crf = torch.stack((1 - seg_mask, seg_mask), dim=0).unsqueeze(0)
        torch.cuda.synchronize()
        out_refine_crf = self.gausscrf(unary=out_refine_crf, img=(img - 0.5) / 0.3, num_iter=8)
        torch.cuda.synchronize()

        _, out_refine_crf = torch.max(out_refine_crf[0], dim=0)
        # out_refine_crf_np = out_refine_crf.cpu().numpy()
        out_refine_crf_np = out_refine_crf[None, None, ...]

        return out_refine_crf_np


# _____________________________________FAMatting____________________________#

class FAMatting:
    FAConfig = namedtuple('FAConfig', ['encoder', 'decoder', 'weights'])

    def __init__(self, encoder, decoder, weight):
        self.model = build_model(FAMatting.FAConfig(encoder, decoder, weight))
        self.model.eval()

    def scale_input_fa(self, x, scale, scale_type):
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

    def np_to_torch(self, x):
        return torch.from_numpy(x).permute(2, 0, 1)[None, :, :, :].float().cuda()

    def refine_mask(self, image, trimap):
        h, w = trimap.shape[:2]

        trimap = np.stack((1 - trimap, trimap), axis=-1)

        image_scale_np = self.scale_input_fa(image / 255., 1.0, cv2.INTER_NEAREST)
        trimap_scale_np = self.scale_input_fa(trimap, 1.0, cv2.INTER_NEAREST)

        with torch.no_grad():
            image_torch = self.np_to_torch(image_scale_np)
            trimap_torch = self.np_to_torch(trimap_scale_np)

            trimap_transformed_torch = self.np_to_torch(trimap_transform(trimap_scale_np))
            image_transformed_torch = groupnorm_normalise_image(image_torch.clone(), format='nchw')
            output = self.model(image_torch, trimap_torch, image_transformed_torch, trimap_transformed_torch)
            output = cv2.resize(output[0].cpu().numpy().transpose((1, 2, 0)), (w, h), cv2.INTER_NEAREST)
        alpha = output[:, :, 0]
        # fg = output[:, :, 1:4]
        # bg = output[:, :, 4:7]

        alpha[trimap[:, :, 0] == 1] = 0
        alpha[trimap[:, :, 1] == 1] = 1
        # fg[alpha == 1] = image_np[alpha == 1]
        # bg[alpha == 0] = image_np[alpha == 0]
        return alpha


# _____________________________________IndexMatting____________________________#
class IndexMatting:
    def __init__(self, weights_path, stride=32, device=torch.device('cuda:0')):
        self.net_index = hlmobilenetv2(pretrained=True, freeze_bn=True, output_stride=stride,
                                  apply_aspp=True, conv_operator='std_conv', decoder='indexnet',
                                  decoder_kernel_size=5,
                                  indexnet='depthwise', index_mode='m2o', use_nonlinear=True, use_context=True)
        self.device = device
        self.stride = stride
        try:
            checkpoint = torch.load(weights_path, map_location=device)
            pretrained_dict = OrderedDict()
            for key, value in checkpoint['state_dict'].items():
                if 'module' in key:
                    key = key[7:]
                pretrained_dict[key] = value
        except:
            raise Exception('Please download the pretrained model!')
        self.net_index.load_state_dict(pretrained_dict)
        self.net_index.to(self.device)
        self.net_index.eval()
        self.in_mean = torch.tensor([0.485, 0.456, 0.406, 0]).view((1, 4, 1, 1)).to(self.device)
        self.in_std = torch.tensor([0.229, 0.224, 0.225, 1]).view((1, 4, 1, 1)).to(self.device)

    def size_align(self, im_size, output_stride):
        return [int(math.ceil(axis / output_stride) * output_stride) for axis in im_size]

    def refine_mask(self, img, trimap):
        with torch.no_grad():
            # pre
            trimap = trimap.unsqueeze(0).unsqueeze(0).to(self.device)
            image = img.to(self.device)
            image = torch.cat((image, trimap), dim=1)

            h, w = image.shape[2:]
            image_batch = (image - self.in_mean) / self.in_std
            image_batch = F.interpolate(image_batch,
                                        size=self.size_align((h, w), self.stride),
                                        mode='nearest')
            # inference
            torch.cuda.synchronize()
            outputs = self.net_index(image_batch)
            torch.cuda.synchronize()

            # post
            outputs = F.interpolate(outputs, size=(h, w), mode='nearest')
            alpha = torch.clamp(outputs, 0, 1).squeeze()
            trimap = trimap.squeeze()
            mask = (trimap == 0.5).float()
            alpha = (1 - mask) * trimap + mask * alpha

            return alpha.cpu().numpy()


class KNN:
    def __init__(self):
        pass

    def apply(self):
        pass


class MOG2:
    def __init__(self, img):
        self.img = img
        self.bs = cv2.createBackgroundSubtractorMOG2(detectShadows=True, varThreshold=50, history=100)

    def apply(self):
        self.bs.apply(self.img)


class lightBS:
    def __init__(self):
        pass

    def apply(self):
        pass



class MaskEstimator:
    def __init__(self,
                 segmentation_size,
                 point_rend_config,
                 crf_config,
                 matting_config,
                 device=torch.device('cuda:0')):
        self.segmentation_size = segmentation_size
        self.device = device
        # Define models
        self.seg_model = PointRendSegmentation(
            detection_config=point_rend_config['detector_config'], #gcfg.detConfig,
            weights_path=point_rend_config['detector_weights'], #cfg.detWeight,
            score=point_rend_config['detector_score'], #cfg.detScore,
            device=self.device
        )
        # self.seg_model = PalomosPose(
        #     img_shape=(segmentation_size[1], segmentation_size[0]),
        #     weights_path='/home/media4us/PycharmProjects/pointrend_with_refinator/weights/model_simulated_RGB_mgpu_scaling_append.0024.h5')

        self.crf_model = CRF(
            img_shape=(segmentation_size[1], segmentation_size[0]),
            filter_size=crf_config['crf_filter_size'],
            col_feats=crf_config['crf_col_feats'],
            device=self.device
        )

        if matting_config['model'] == 'index':
            self.matting_model = IndexMatting(
                weights_path=matting_config['matting_weights'],
                stride=matting_config['matting_stride'],
                device=self.device
            )
        else:
            self.matting_model = FAMatting(
                encoder=matting_config['matting_encoder'],
                decoder=matting_config['matting_decoder'],
                weight=matting_config['matting_weights'])

    @time_decor
    def trimap_approx_gpu(self, trimap: torch.Tensor):
        morph = Morphology(trimap.float(), device=self.device)
        dilate = morph.dilation(torch.ones((15, 15), dtype=torch.float, device=self.device))
        erode = morph.erosion(torch.ones((3, 3), dtype=torch.float, device=self.device))

        out_trimap = 0.5 * dilate
        out_trimap[erode > 0] = 1

        return out_trimap

    @time_decor
    def trimap_approx(self, trimap_np: np.ndarray):
        dilate_trimap_np = cv2.dilate(np.uint8(trimap_np > 0), np.ones((15, 15), dtype=np.uint8)) > 0
        # wrong detection simulation
        erode_trimap_np = cv2.erode(np.uint8(trimap_np > 0), np.ones((3, 3), dtype=np.uint8)) > 0

        out_trimap = np.float32(dilate_trimap_np)
        in_trimap = np.float32(erode_trimap_np)

        out_trimap = 0.5 * out_trimap
        out_trimap[in_trimap > 0] = 1

        return out_trimap

    def predict(self, im):
        im_resize = cv2.resize(im, self.segmentation_size)
        im_torch = torch.tensor(im_resize).to(self.device).permute(2, 0, 1).unsqueeze(0).float()
        im_norm_torch = im_torch / 255.

        point_rend_mask = self.seg_model.predict(im_resize)

        crf_mask = self.crf_model.refine_mask(im_norm_torch, point_rend_mask)

        # IndexMatting

        # trimap = torch.tensor(self.trimap_approx(crf_mask)).to(self.device)
        trimap_gpu = self.trimap_approx_gpu(crf_mask).to(self.device)

        matting_mask = self.matting_model.refine_mask(im_norm_torch, trimap_gpu.squeeze(0).squeeze(0))

        # FAMatting
        # matting_mask = self.matting_model.refine_mask(im_resize, self.trimap_approx(crf_mask))

        return cv2.resize(matting_mask, (im.shape[1], im.shape[0]), interpolation=cv2.INTER_NEAREST)




# TODO
# def tenorToNumpy(img: torch.tensor):
#     pass


def create_folder(name: str):
    if os.path.isdir(name):
        shutil.rmtree(name)
    os.makedirs(name, exist_ok=True)


# TODO
class visualize:
    def __init__(self, img):
        self.img = img

    def show(self):
        pass

    def save(self, path: str, name: str, ext: str, idx: int):
        # if self.img.max() == 255 and type(self.img) == np.uint8:
        self.ext = ext
        self.path = path
        self.idx = idx
        self.name = name
        cv2.imwrite(self.path + "/" + self.name + "__" + str(self.idx) + "." + self.ext, self.img)

    def montage(self):
        pass


def numpyToTensor(img: np.ndarray, norm=False):
    if norm:
        img_norm = img / 255.
        img = torch.tensor(img_norm).cuda().permute(2, 0, 1).unsqueeze(0).float()
    else:
        img = torch.tensor(img).cuda().permute(2, 0, 1).unsqueeze(0).float()
    return img


if __name__ == "__main__":
    cfg = config()
    DEVICE = torch.device('cuda:0')

    create_folder('montages')
    create_folder('frames')

    # Define video attributes
    is_webcam = cfg.video.isdigit()

    if is_webcam:
        cap = cv2.VideoCapture(int(cfg.video))
    else:
        cap = cv2.VideoCapture(cfg.video)

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    INFER_WIDTH = width
    INFER_HEIGHT = height

    INFER_MATTING_WIDTH = INFER_WIDTH // 2
    INFER_MATTING_HEIGHT = INFER_HEIGHT // 2

    images = deque(maxlen=int(fps * 300))
    i = 0

    im_background = cv2.imread(cfg.background)
    im_background = cv2.resize(im_background, (width, height))

    mask_seg = MaskEstimator(
        segmentation_size=(INFER_WIDTH, INFER_HEIGHT),
        point_rend_config={
            'detector_config': cfg.detConfig,
            'detector_weights': cfg.detWeight,
            'detector_score': cfg.detScore
        },
        crf_config={
            'crf_filter_size': cfg.crfKernel,
            'crf_col_feats': cfg.crfFeats
        },

        matting_config={
            'model': cfg.matting,
            'matting_stride': cfg.inStride,
            'matting_weights': cfg.inWeight
        }
        # matting_config={
        #     'model': cfg.matting,
        #     'matting_encoder': cfg.faEncoder,
        #     'matting_decoder': cfg.faDecoder,
        #     'matting_weights': cfg.faWeights,
        # }
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        images.append(frame)
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
        i += 1

    for idx, im in enumerate(images):
        im_pad = np.pad(im, ((80, 80), (80, 80), (0, 0)), 'constant', constant_values=0)
        mask = mask_seg.predict(im_pad)
        mask = mask[80:-80, 80:-80]
        mask_im = np.uint8(255 * mask)
        mask_rgb = np.tile(np.expand_dims(mask, axis=-1), [1, 1, 3])
        # mask_rgb_bool = mask_rgb > 0

        # Montage mask to the background
        back_im = np.uint8(np.float32(im) * mask_rgb + np.float32(im_background) * (1 - mask_rgb))

        cv2.imwrite(f'frames/im_{idx}.jpg', im)
        cv2.imwrite(f'frames/mask_{idx}.png', mask_im)
        cv2.imwrite(f'frames/back_{idx}.png', back_im)

        montage = np.concatenate((im, np.tile(np.expand_dims(mask_im, axis=-1), [1, 1, 3])), axis=1)
        cv2.imwrite(f'montages/montage_{idx}.png', montage)
