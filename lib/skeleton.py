from enum import Enum

import cv2

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path


class SkeletonPart(Enum):
    Nose = 0
    Neck = 1
    RShoulder = 2
    RElbow = 3
    RWrist = 4
    LShoulder = 5
    LElbow = 6
    LWrist = 7
    RHip = 8
    RKnee = 9
    RAnkle = 10
    LHip = 11
    LKnee = 12
    LAnkle = 13


SkeletonColors = [
    [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
    [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255]
]


class SkeletonImplement:
    def __init__(self, model="mobilenet_thin", target_size=(368, 368), tf_config=None):
        self.estimator = TfPoseEstimator(get_graph_path(model), target_size=target_size, tf_config=tf_config)

    def infer_skeleton(self, src):
        humans = self.estimator.inference(src, upsample_size=4.0)
        human = humans[0]
        for unused_index in [14, 15, 16, 17, 18]:
            if unused_index in human.body_parts.keys():
                del human.body_parts[unused_index]

        return human

    def draw_skeleton(self, img, human):
        return self.estimator.draw_humans(img, [human], imgcopy=False)


class SkeletonTest:
    def __init__(self, human, human_contour, src_shape):
        self.isAcquired = []  # Whether or not each joint is acquired
        self.probability = []  # Probability of each joint
        self.isWithinContour = []  # Whether or not each joint is within the contour

        src_height, src_width = src_shape[:2]

        body_part_positions = []

        for joint_index in range(SkeletonPart.LAnkle.value + 1):
            if joint_index in human.body_parts.keys():
                self.isAcquired.append(True)
                body_part = human.body_parts[joint_index]
                body_part_position = (int(body_part.x * src_width + 0.5), int(body_part.y * src_height + 0.5))
                body_part_positions.append(body_part_position)
                self.probability.append(body_part.score)
            else:
                self.isAcquired.append(False)
                body_part_positions.append(None)
                self.probability.append(0.0)

        self.isWithinContour = [body_part_position is not None
                                and cv2.pointPolygonTest(human_contour, body_part_position, False) > -1
                                for body_part_position in body_part_positions]

    # Change conditions freely
    def is_reliable(self):
        minimum_probability = 0.2

        all_joints_are_acquired = all(self.isAcquired)
        all_joints_are_reliable = all([p > minimum_probability for p in self.probability])
        all_joints_are_within_contour = all(self.isWithinContour)

        return all_joints_are_acquired and all_joints_are_reliable and all_joints_are_within_contour

    def report(self):
        print("Each joint is acquired:", self.isAcquired)
        print("Probability:", self.probability)
        print("Each joint is within the contour:", self.isWithinContour)
