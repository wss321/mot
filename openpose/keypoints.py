from .tf_pose import common
from .tf_pose.estimator import TfPoseEstimator
from .tf_pose.networks import get_graph_path, model_wh


def get_keypoints(image, estimator, wh='0x0'):
    w, h = model_wh(wh)
    if image is None:
        raise ValueError('image is None')
    humans = estimator.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=4.0)
    return humans


def draw_humans(image, humans, imgcopy=False):
    return TfPoseEstimator.draw_humans(image, humans, imgcopy)


if __name__ == '__main__':
    import cv2

    image = common.read_imgfile(r'S:\PyCharmProject\SST\dataset\MOT16\test\MOT16-01\img1\000002.jpg', None, None)
    model = 'mobilenet_thin'
    e = TfPoseEstimator(get_graph_path(model), target_size=(432, 368))
    humans = get_keypoints(image, e)
    image = draw_humans(image, humans, imgcopy=False)
    try:
        import matplotlib

        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt

        plt.figure()
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.show()

    except Exception as e:
        cv2.imshow('result', image)
        cv2.waitKey()
