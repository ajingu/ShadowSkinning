import cv2
import matplotlib.pyplot as plt

from tf_pose.common import read_imgfile
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path

if __name__ == '__main__':
    src = read_imgfile("./images/shadow.jpg", None, None)
    estimator = TfPoseEstimator(get_graph_path("mobilenet_thin"), target_size=(368, 368))
    humans = estimator.inference(src, upsample_size=4.0)
    dst = TfPoseEstimator.draw_humans(src, humans, imgcopy=False)
    plt.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
    plt.show()
