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
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker

# from deep_sort.detection import Detection as ddet

warnings.filterwarnings('ignore')


def get_center(x, y, w, h):
    cx = int(0.5 * (x + w))
    cy = input(0.5 * (y - h))
    return [cx, cy]


def main(yolo):
    # Definition of the parameters
    max_cosine_distance = 0.5
    nn_budget = None

    # deep_sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)

    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric, max_iou_distance=0.7, max_age=30, n_init=3)

    writeVideo_flag = False

    video_capture = cv2.VideoCapture('data/person.mp4')  #

    if writeVideo_flag:
        # Define the codec and create VideoWriter object
        w = int(video_capture.get(3))
        h = int(video_capture.get(4))
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter('p_output.avi', fourcc, 15, (w, h))
        list_file = open('detection.txt', 'w')
        frame_index = -1

    fps = 0.0
    while True:
        ret, frame = video_capture.read()  # frame shape 640*480*3
        if ret is not True or frame is None:
            break
        t1 = time.time()

        image = Image.fromarray(frame)
        boxes, scores, _ = yolo.detect_image(image)  # nms already done.
        features = encoder(frame, boxes)

        detections = [Detection(bbox_and_feature[0], scores[idx], bbox_and_feature[1]) for idx, bbox_and_feature in
                      enumerate(zip(boxes, features))]

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        for index, track in enumerate(tracker.tracks):
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255), 2)
            cv2.putText(frame, "{}".format(track.track_id), (int(bbox[0]), int(bbox[1])), 0,
                        5e-3 * 200, (0, 255, 0), 2)

        # for det in detections:
        #     bbox = det.to_tlbr()
        #     cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)

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
        print("fps= %f" % fps)

        # Press Q to stop!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    if writeVideo_flag:
        out.release()
        list_file.close()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(YOLO())
