from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path


class SkeletonImplement:
    def __init__(self):
        self.estimator = TfPoseEstimator(get_graph_path("mobilenet_thin"), target_size=(368, 368))

    def infer_skeletons(self, src):
        return self.estimator.inference(src, upsample_size=4.0)

    def draw_skeletons(self, img):
        humans = self.infer_skeletons(img)
        return self.estimator.draw_humans(img, humans, imgcopy=False)
