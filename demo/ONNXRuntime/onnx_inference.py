#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
import sys
sys.path.append("/home/ailab/sophic/thinh/hand_gesture/YOLOX")
import argparse
import os
import time
import cv2
import numpy as np

import onnxruntime

from yolox.data.data_augment import preproc as preprocess
from yolox.data.datasets import COCO_CLASSES
from yolox.utils import mkdir, multiclass_nms, demo_postprocess, vis
VOC_CLASSES = ('__background__', 'hi', 'like', 'ok')


def make_parser():
    parser = argparse.ArgumentParser("onnxruntime inference sample")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="yolox.onnx",
        help="Input your onnx model.",
    )
    parser.add_argument(
        "-i",
        "--image_path",
        type=str,
        default='test_image.png',
        help="Path to your input image.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default='demo_output',
        help="Path to your output directory.",
    )
    parser.add_argument(
        "-s",
        "--score_thr",
        type=float,
        default=0.3,
        help="Score threshould to filter the result.",
    )
    parser.add_argument(
        "--input_shape",
        type=str,
        default="640,640",
        help="Specify an input shape for inference.",
    )
    parser.add_argument(
        "--with_p6",
        action="store_true",
        help="Whether your model uses p6 in FPN/PAN.",
    )
    return parser


# if __name__ == '__main__':
args = make_parser().parse_args()

input_shape = tuple(map(int, args.input_shape.split(',')))
cap = cv2.VideoCapture('/home/ailab/sophic/thinh/hand_gesture/YOLOX/test_2.webm')
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
fps = cap.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height)))
while True:
    _,origin_img = cap.read()
    t0 = time.time()
    origin_img = cv2.resize(origin_img,(input_shape[0],input_shape[1]))
    img, ratio = preprocess(origin_img, input_shape)

    session = onnxruntime.InferenceSession(args.model)

    ort_inputs = {session.get_inputs()[0].name: img[None, :, :, :]}
    output = session.run(None, ort_inputs)
    predictions = demo_postprocess(output[0], input_shape, p6=args.with_p6)[0]

    boxes = predictions[:, :4]
    scores = predictions[:, 4:5] * predictions[:, 5:]

    boxes_xyxy = np.ones_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.
    boxes_xyxy /= ratio
    dets = multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=0.1)
    if dets is not None:
        final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
        origin_img = vis(origin_img, final_boxes, final_scores, final_cls_inds,
                        conf=args.score_thr, class_names=VOC_CLASSES)
        out.write(origin_img)
    fps = int(1/(time.time() - t0))
    print(fps)
    
    # cv2.imshow('img', origin_img)
    ch = cv2.waitKey(1)
    if ch == 27 or ch == ord("q") or ch == ord("Q"):
        break
cap.release()
out.release()
cv2.destroyAllWindows()
