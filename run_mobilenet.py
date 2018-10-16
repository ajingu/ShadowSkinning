import cv2
import tf_pose
from tf_pose.estimator import TfPoseEstimator
import matplotlib.pyplot as plt

image = tf_pose.common.read_imgfile("./images/shadow.jpg", None, None)
humans = tf_pose.infer(image="./images/shadow.jpg", model="mobilenet_thin", resize="368x368")

image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
