import os.path
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
import logging
from configs.detection_runner_config import opt

BASEDIR = os.path.dirname(__file__)


class YoloV7:
    def __init__(self, conf_thres, device='cpu'):
        # self.weights = os.path.join(BASEDIR, 'weights', 'yolov7.pt')
        self.weights = os.path.join(BASEDIR, 'weights', 'best.pt')
        self.img_size = 640
        self.conf_thres = conf_thres
        self.iou_thres = 0.45

        self.trace = not opt.no_trace
        self.device = select_device(device)  # help='cuda device, i.e. 0 or 0,1,2,3 or cpu'
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        if os.path.exists(self.weights):
            self.model = attempt_load(self.weights, map_location=device)  # load FP32 model
            self.stride = int(self.model.stride.max())  # model stride
            self.imgsz = check_img_size(self.img_size, s=self.stride)  # check img_size
            if self.trace:
                self.model = TracedModel(self.model, device, self.imgsz)

            if self.half:
                self.model.half()  # to FP16
        else:
            logging.error(f'*****[INFO] Could not load the weight.*****')
            exit()

    def detect(self, source) -> list:

        # Initialize
        set_logging()

        dataset = LoadImages(source, img_size=self.imgsz, stride=self.stride)
        # Get names
        names = self.model.module.names if hasattr(self.model, 'module') else self.model.names

        # Run inference
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(
                next(self.model.parameters())))  # run once
        old_img_w = old_img_h = self.imgsz
        old_img_b = 1

        final_data = []  # final data per image
        for path, img, im0s, vid_cap in dataset:  # directory ho vane iterate garxa each image ma

            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Warmup
            if self.device.type != 'cpu' and (
                    old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
                old_img_b = img.shape[0]
                old_img_h = img.shape[2]
                old_img_w = img.shape[3]
                for i in range(3):
                    self.model(img, augment=opt.augment)[0]

            # Inference
            with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
                pred = self.model(img, augment=opt.augment)[0]

            # Apply NMS
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=opt.classes,
                                       agnostic=opt.agnostic_nms)

            for i, det in enumerate(pred):  # detections per image

                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

                detected_num = len(det)
                if detected_num:
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    detected_objects = []
                    for *bbox, confidence, cls in reversed(det):
                        left, top, right, bottom = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                        confidence: float = float(confidence)
                        label: str = names[int(cls)]
                        cropped_image: np.ndarray = im0[top:bottom, left:right]
                        object_data: list = [(left, top, right, bottom), confidence, label, cropped_image]
                        detected_objects.append(object_data)
                    final_data.append(detected_objects)
                    final_data.append(detected_num)
                    final_data.append(im0)

                else:
                    object_data: list = [(0.0, 0.0, 0.0, 0.0), 0.0, 'N/A', None]
                    final_data.append([object_data])
                    final_data.append(detected_num)
                    final_data.append(im0)

            return final_data
