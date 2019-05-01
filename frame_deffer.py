import cv2
from cv2 import VideoWriter_fourcc

video_capture = cv2.VideoCapture('./data/school.mp4')
w = int(video_capture.get(3))
h = int(video_capture.get(4))
out_root = './school_deffer.avi'
# Edit each frame's appearing time!
fps = 20
fourcc = VideoWriter_fourcc(*"MJPG")  # 支持jpg
videoWriter = cv2.VideoWriter(out_root, fourcc, fps, (w, h))
i = 0
while True:
    ret, now_frame = video_capture.read()
    i += 1
    if ret is not True or now_frame is None:
        break
    # now_frame = cv2.cvtColor(now_frame, cv2.COLOR_BGR2GRAY)
    if i == 1:
        before = now_frame
        continue

    deffer = now_frame - before
    before = now_frame
    cv2.imshow('', deffer)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    videoWriter.write(deffer)

videoWriter.release()
