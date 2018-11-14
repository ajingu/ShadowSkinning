from enum import Enum

import cv2
import numpy as np

from tf_pose.estimator import TfPoseEstimator, BodyPart
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
    REye = 14
    LEye = 15
    REar = 16
    LEar = 17


SkeletonColors = [
    [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
    [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255]
]

NUMBER_OF_BODY_PARTS = 14


class SkeletonImplement:
    def __init__(self, model="mobilenet_thin", target_size=(368, 368), tf_config=None, adjust_nose_position=False):
        self.estimator = TfPoseEstimator(get_graph_path(model), target_size=target_size, tf_config=tf_config)
        self.adjust_nose_position = adjust_nose_position

    def infer_skeleton(self, src):
        humans = self.estimator.inference(src, upsample_size=4.0)
        if len(humans) == 0:
            return None

        human = humans[0]

        # skip wrists under nose
        if 0 in human.body_parts and 4 in human.body_parts and 7 in human.body_parts:
            nose_y, right_wrist_y, left_wrist_y = human.body_parts[0].y, human.body_parts[4].y, human.body_parts[7].y
            if right_wrist_y > nose_y and left_wrist_y > nose_y:
                print("Wrists are under nose.")
                return None

        # adjust nose position
        if 16 in human.body_parts and 17 in human.body_parts:
            right_ear, left_ear = human.body_parts[16], human.body_parts[17]
            nose_x, nose_y = (right_ear.x + left_ear.x) / 2, (right_ear.y + left_ear.y) / 2
            if 0 in human.body_parts:
                human.body_parts[0].x = nose_x
                human.body_parts[0].y = nose_y
            else:
                human.body_parts[0] = BodyPart("0-0", 0, nose_x, nose_y, 0.5)

        # skip unused joints
        for unused_index in [14, 15, 16, 17, 18]:
            if unused_index in human.body_parts.keys():
                del human.body_parts[unused_index]

        if self.adjust_nose_position:
            nose, neck = human.body_parts[0], human.body_parts[1]
            if abs(nose.y - neck.y) / (abs(nose.x - neck.x) + 1e-4) < 5:
                human.body_parts[0].x = neck.x

        return human

    def draw_skeleton(self, img, human):
        return self.estimator.draw_humans(img, [human], imgcopy=False)


class SkeletonTest:
    def __init__(self, human, human_contour, src_shape):
        src_height, src_width = src_shape[:2]
        body_part_indices = list(human.body_parts.keys())

        self.isAcquired = np.zeros(NUMBER_OF_BODY_PARTS)
        self.isAcquired[body_part_indices] = 1

        self.isWithinContour = [cv2.pointPolygonTest(human_contour,
                                                     (int(human.body_parts[joint_index].x * src_width + 0.5),
                                                      int(human.body_parts[joint_index].y * src_height + 0.5)),
                                                     False) == 1
                                for joint_index in body_part_indices]

    # Change conditions freely
    def is_reliable(self):
        all_joints_are_acquired = all(self.isAcquired)
        all_joints_are_within_contour = all(self.isWithinContour)

        return all_joints_are_acquired and all_joints_are_within_contour

    def report(self):
        print("Each joint is acquired:", self.isAcquired)
        print("Each joint is within the contour:", self.isWithinContour)
