import cv2
from cv2 import VideoWriter_fourcc

video_capture = cv2.VideoCapture('../data/school.mp4')
w = int(video_capture.get(3))
h = int(video_capture.get(4))
w1 = int(w / 4)
h1 = int(h / 4)
out_root = '../data/school_d4.avi'
# Edit each frame's appearing time!
fps = 20
fourcc = VideoWriter_fourcc(*"MJPG")  # 支持jpg
videoWriter = cv2.VideoWriter(out_root, fourcc, fps, (w1, h1))

out = cv2.VideoWriter(out_root, fourcc, 15, (w1, h1))
while True:
    ret, frame = video_capture.read()
    if ret is not True or frame is None:
        break
    frame = cv2.resize(frame, (w1, h1))
    videoWriter.write(frame)

videoWriter.release()
out.release()
