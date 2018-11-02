import sys

from tf_pose.common import read_imgfile

from lib.contour import find_human_contour
from lib.skeleton import SkeletonImplement, SkeletonTest
from lib.shadow import Shadow
from lib.watch import watch_image_generation

TARGET_DIRECTORY_PATH = "./images"


def get_shadow_info(image_path, frame_index):
    src = read_imgfile(image_path, None, None)

    if src is None:
        print(image_path + " is not found.")
        return

    print("Frame Index:", frame_index)

    human_contour = find_human_contour(src)

    skeleton_implement = SkeletonImplement()
    human = skeleton_implement.infer_skeleton(src)
    human = skeleton_implement.remove_unused_joints(human)

    skeleton_test = SkeletonTest(human, human_contour, src.shape)
    skeleton_test.report()
    if not skeleton_test.is_reliable():
        print("This skeleton model is not reliable.")
        sys.exit(0)

    shadow = Shadow(src.shape, human, human_contour)

    print("The number of body parts:", len(shadow.body_part_positions))
    print("The number of contour vertices:", len(shadow.contour_vertex_positions))
    print("The number of triangle vertices:", len(shadow.triangle_vertex_indices))


if __name__ == '__main__':
    watch_image_generation(get_shadow_info, TARGET_DIRECTORY_PATH)
