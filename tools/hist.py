# -*- encoding:utf-8 -*-
'''
python 绘制颜色直方图,cv2有hist和calcHist两种方法
'''
import cv2
# from matplotlib import pyplot as plt
import numpy as np


def hist(img, show=False):
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # bins->图像中分为多少格；range->图像中数字范围
    h = np.histogram(img.ravel(), 256, [0, 256])
    # if show:
    #     plt.show()
    return h[0] / img.size


# def calcHist(img, show=False):
#     color = ('b', 'g', 'r')
#     for i, col in enumerate(color):
#         histr = cv2.calcHist([img], [i], None, [256], [0, 256])
#         if show:
#             plt.plot(histr, color=col)
#     if show:
#         plt.xlim([0, 256])
#         plt.show()


def color_feature(img, times=3):
    def __color_feature(h):
        cf = []
        for i in range(int(len(h) / 2)):
            cf.append(h[2 * i] + h[2 * i + 1])
        return cf

    h = hist(img, False)
    for j in range(times):
        h = __color_feature(h)
    return h


if __name__ == '__main__':
    h = hist(cv2.imread(r'S:\PyCharmProject\yolo3\VOCdevkit\VOC2012\JPEGImages\2007_000033.jpg'), True)
    # calcHist(cv2.imread(r'S:\PyCharmProject\yolo3\VOCdevkit\VOC2012\JPEGImages\2007_000033.jpg'))
