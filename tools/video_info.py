import cv2


class Video(object):
    def __init__(self, video_path):
        self.path = video_path
        self.video_capture = cv2.VideoCapture(video_path)  # 'data/person.mp4'
        self.w = int(self.video_capture.get(3))
        self.h = int(self.video_capture.get(4))
