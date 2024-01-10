import os.path
import time
from pathlib import Path

import cv2
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

    def detect(self, source, view_img=False, webcam=False, save_img=False, save_dir=os.path.join(BASEDIR,
                                                                                                'resources',
                                                                                                'detection_results')):
        # Initialize
        set_logging()

        # Set Dataloader
        vid_writer = None
        if webcam:
            # view_img = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=self.imgsz, stride=self.stride)
        else:
            dataset = LoadImages(source, img_size=self.imgsz, stride=self.stride)
        # Get names and colors
        names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

        # Run inference
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(
                next(self.model.parameters())))  # run once
        old_img_w = old_img_h = self.imgsz
        old_img_b = 1

        t0 = time.time()
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
            t1 = time_synchronized()
            with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
                pred = self.model(img, augment=opt.augment)[0]
            t2 = time_synchronized()

            # Apply NMS
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=opt.classes,
                                       agnostic=opt.agnostic_nms)
            t3 = time_synchronized()

            # Process detections
            cropped_detected_images_list = []
            detected_object_count = 0
            for i, det in enumerate(pred):  # detections per image
                if webcam:  # batch_size >= 1
                    p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
                else:
                    p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *bbox, confidence, cls in reversed(det):
                        left, top, right, bottom = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                        confidence: float = float(confidence)
                        label: str = names[int(cls)]
                        cropped_image = im0[top:bottom, left:right]
                        cropped_detected_images_list.append(cropped_image)

                        if save_img or view_img:  # Add bbox to image
                            label_ = f'{names[int(cls)]} {confidence:.2f}'
                            plot_one_box(bbox, im0, label=label_, color=colors[int(cls)], line_thickness=1)

                # Print time (inference + NMS)
                print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

                detected_object_count += len(det)

                # Stream results
                if view_img:
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL)
                    cv2.imshow(str(p), im0)
                    cv2.waitKey(0)  # 1 millisecond

                # Save results (image with detections)
                if save_img:
                    if dataset.mode == 'image':
                        save_path = os.path.join(save_dir, 'images', p.name)
                        cv2.imwrite(save_path, im0)
                        print(f" The image with the result is saved in: {save_path}")
                    else:  # 'video' or 'stream'
                        save_path = os.path.join(save_dir, 'videos', p.name)

                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        vid_writer.write(im0)
        print(f'Done. ({time.time() - t0:.3f}s)')
