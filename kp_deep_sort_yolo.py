#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

from timeit import time
import warnings
import cv2
from PIL import Image
from yolo import YOLO
import numpy as np
from deep_sort import nn_matching, generate_detections as gdet
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools.video_info import Video
from openpose.keypoints import TfPoseEstimator, get_graph_path, model_wh, get_keypoints, draw_humans
import tensorflow as tf

warnings.filterwarnings('ignore')

output_dir = 'output/'


def main(yolo, video_path=0, save_path=None):
    # Definition of the parameters
    max_cosine_distance = 0.5
    nn_budget = None
    # openpose
    w, h = model_wh('0x0')
    model = 'mobilenet_thin'
    config = tf.ConfigProto(device_count={'gpu': 0})
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    e = TfPoseEstimator(get_graph_path(model), target_size=(64, 64), tf_config=config)
    # deep_sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)

    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    video = Video(video_path)
    # video_capture = cv2.VideoCapture()  # 'data/person.mp4'
    w = video.w
    h = video.h
    tracker = Tracker(metric, img_shape=(w, h), max_eu_dis=0.1 * np.sqrt((w ** 2 + h ** 2)))
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(save_path, fourcc, 7, (w, h))
    list_file = open(output_dir + 'detection.txt', 'w')
    frame_index = -1
    fps = 0.0

    while True:
        ret, frame = video.video_capture.read()  # frame shape 640*480*3
        if ret is not True or frame is None:
            break
        t1 = time.time()

        image = Image.fromarray(frame)
        boxes, scores, _ = yolo.detect_image(image)
        # start = time.time()
        features = encoder(frame, boxes)  # 提取到每个框的特征
        # end = time.time()
        # print(end-start)
        detections = [Detection(bbox_and_feature[0], scores[idx], bbox_and_feature[1]) for idx, bbox_and_feature in
                      enumerate(zip(boxes, features))]  # 保存到一个类中

        # Call the tracker
        tracker.predict()
        tracker.update(detections, np.asarray(image))
        humans = get_keypoints(image, e)
        frame = draw_humans(image, humans, imgcopy=False)
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

        if save_path is not None:
            # Define the codec and create VideoWriter object

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
        # print("fps= %f" % fps)

        # Press Q to stop!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.video_capture.release()
    if save_path is not None:
        out.release()
        list_file.close()
    cv2.destroyAllWindows()


if __name__ == '__main__':  # 'car_add_center_match_85.avi'
    main(YOLO(iou_th=0.5, score_th=0.6), video_path='data/person.mp4', save_path=None)
