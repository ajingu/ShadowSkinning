import sys

import cv2
import matplotlib.pyplot as plt

from tf_pose.common import read_imgfile

from lib.skeleton import SkeletonImplement

if __name__ == '__main__':
    src = read_imgfile("./images/shadow.jpg", None, None)
    dst = src.copy()

    skeletonImplement = SkeletonImplement()
    human = skeletonImplement.infer_skeleton(src)
    if human is None:
        print("Not a human has detected in the image.")
        sys.exit(0)

    dst = skeletonImplement.draw_skeleton(dst, human)

    plt.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
    plt.show()
