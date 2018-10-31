import cv2

from tf_pose.common import CocoPart
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path


class SkeletonImplement:
    def __init__(self):
        self.estimator = TfPoseEstimator(get_graph_path("mobilenet_thin"), target_size=(368, 368))

    def infer_skeleton(self, src):
        humans = self.estimator.inference(src, upsample_size=4.0)
        return humans[0]

    def draw_skeleton(self, img, human):
        return self.estimator.draw_humans(img, [human], imgcopy=False)


class SkeletonTest:
    def __init__(self, human, human_contour, src_shape):
        self.isAcquired = []  # Whether or not each joint is acquired
        self.probability = []  # Probability of each joint
        self.isWithinContour = []  # Whether or not each joint is within the contour

        src_height, src_width = src_shape[:2]

        body_part_positions = []

        for joint_index in range(CocoPart.Background.value):
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
    # [Example] Ignore REye, LEye, REar, LEar
    def is_reliable(self):
        minimum_probability = 0.2

        all_joints_are_acquired = all(self.isAcquired[:14])
        all_joints_are_reliable = all([p > minimum_probability for p in self.probability[:14]])
        all_joints_are_within_contour = all(self.isWithinContour[:14])

        return all_joints_are_acquired and all_joints_are_reliable and all_joints_are_within_contour

    def report(self):
        print("Each joint is acquired:", self.isAcquired)
        print("Probability:", self.probability)
        print("Each joint is within the contour:", self.isWithinContour)
