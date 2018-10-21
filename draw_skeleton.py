import cv2
import matplotlib.pyplot as plt

from tf_pose.common import read_imgfile

from lib.skeleton import SkeletonImplement

if __name__ == '__main__':
    src = read_imgfile("./images/shadow.jpg", None, None)
    dst = src.copy()

    skeletonImplement = SkeletonImplement()
    dst = skeletonImplement.draw_skeletons(dst)

    plt.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
    plt.show()
