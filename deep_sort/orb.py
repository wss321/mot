import cv2


# 自定义计算两个图片相似度函数
def img_similarity(img1, img2):
    """
    :param img1:
    :param img2:
    :return: 图片相似度
    """
    # 读取图片
    try:

        img1 = cv2.cvtColor(img1, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.cvtColor(img2, cv2.IMREAD_GRAYSCALE)
        # 初始化ORB检测器
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)

        # 提取并计算特征点
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)

        # knn筛选结果
        matches = bf.knnMatch(des1, trainDescriptors=des2, k=2)

        # 查看最大匹配点数目
        if len(matches) == 0:
            return 0.5
        good = [m[0] for m in matches if len(m) == 2 and m[0].distance < 0.75 * m[1].distance]
        if len(good) == 0:
            return 0.5
        similarity = len(good) / len(matches)
    except:
        print('error')
        return 0.5
    return similarity


if __name__ == '__main__':
    img1_path = cv2.imread(r'F:\img_spam\test\7ba.jpg', cv2.IMREAD_GRAYSCALE)
    img2_path = cv2.imread(r'F:\img_spam\test\ba.jpg', cv2.IMREAD_GRAYSCALE)
    similary = img_similarity(img1_path, img2_path)
