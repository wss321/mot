#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

# import os
from timeit import time
import warnings
# import sys
import cv2
from PIL import Image
from yolo import YOLO

from deep_sort import generate_detections as gdet
from deep_sort.detection import Detection

# from deep_sort.detection import Detection as ddet

warnings.filterwarnings('ignore')


def get_center(x, y, w, h):
    x = int(x)
    y = int(y)
    w = int(w)
    h = int(h)
    cx = int(0.5 * (x + w))
    cy = int(0.5 * (y + h))
    return [cx, cy]


def main(yolo, writeVideo_flag=False):
    # Definition of the parameters
    max_cosine_distance = 0.5
    nn_budget = None
    nms_max_overlap = 0.8

    # deep_sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    # tracker = Tracker(metric)

    video_capture = cv2.VideoCapture('data/person.mp4')  #

    if writeVideo_flag:
        # Define the codec and create VideoWriter object
        w = int(video_capture.get(3))
        h = int(video_capture.get(4))
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter('detect.avi', fourcc, 15, (w, h))
        list_file = open('detection.txt', 'w')
        frame_index = -1

    fps = 0.0
    while True:
        ret, frame = video_capture.read()  # frame shape 640*480*3
        if ret is not True or frame is None:
            break
        t1 = time.time()

        image = Image.fromarray(frame)
        boxes, scores, _ = yolo.detect_image(image)
        features = encoder(frame, boxes)

        detections = [Detection(bbox_and_feature[0], scores[idx], bbox_and_feature[1]) for idx, bbox_and_feature in
                      enumerate(zip(boxes, features))]

        centers = [get_center(det.to_tlbr()[0], det.to_tlbr()[1], det.to_tlbr()[2], det.to_tlbr()[3]) for det in
                   detections]
        print("{}:{}".format(len(centers), centers))

        for det in detections:
            bbox = det.to_tlbr()
            center = get_center(bbox[0], bbox[1], bbox[2], bbox[3])

            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
            cv2.putText(frame, ".", (center[0], center[1]), 0,
                        5e-3 * 200, (0, 0, 255), 2)
        cv2.putText(frame, "{}".format(len(centers)), (int(image.size[0]*0.5), int(image.size[1]*0.9)), 0,
                    5e-3 * 200, (0, 0, 255), 2)
        cv2.imshow('', frame)

        if writeVideo_flag:
            # save a frame
            out.write(frame)
            frame_index = frame_index + 1
            list_file.write(str(frame_index) + ' ')
            if len(boxes) != 0:
                for i in range(0, len(boxes)):
                    list_file.write(
                        str(boxes[i][0]) + ' ' + str(boxes[i][1]) + ' ' + str(boxes[i][2]) + ' ' + str(
                            boxes[i][3]) + ' ')
            list_file.write('\n')

        fps = (fps + (1. / (time.time() - t1))) / 2
        # print("fps= %f" % (fps))

        # Press Q to stop!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    if writeVideo_flag:
        out.release()
        list_file.close()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(YOLO(iou_th=0.5, score_th=0.6), False)
