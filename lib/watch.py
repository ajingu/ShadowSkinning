import os
import sys
import time

from tf_pose.common import read_imgfile
from watchdog.events import PatternMatchingEventHandler

from lib.contour import find_human_contour
from lib.skeleton import SkeletonTest
from lib.shadow import Shadow


class ImageGenerationEventHandler(PatternMatchingEventHandler):
    def __init__(self, patterns, skeleton_implement, oscClient):
        super(ImageGenerationEventHandler, self).__init__(patterns=patterns)
        self.skeleton_implement = skeleton_implement
        self.oscClient = oscClient

    def on_created(self, event):
        src_path = event.src_path.replace("\\", "/")
        print(src_path + " was created")
        frame_index = os.path.splitext(os.path.basename(src_path))[0].split("_")[-1]

        # Avoid the case of not finishing generating images
        time.sleep(0.1)

        src = read_imgfile(src_path, None, None)
        if src is None:
            print(src_path + " is not found.")
            return

        print("Frame Index:", frame_index)

        human_contour = find_human_contour(src)
        human = self.skeleton_implement.infer_skeleton(src)
        skeleton_test = SkeletonTest(human, human_contour, src.shape)
        if not skeleton_test.is_reliable():
            skeleton_test.report()
            print("This skeleton model is not reliable.")
            sys.exit(0)

        shadow = Shadow(src.shape, human, human_contour, arrangement_interval=10)
        print("The number of body parts:", len(shadow.body_part_positions))
        print("The number of vertices:", len(shadow.vertex_positions))
        print("The number of triangle vertices:", len(shadow.triangle_vertex_indices))
        self.oscClient.send(shadow, frame_index)

    def on_deleted(self, event):
        pass

    def on_modified(self, event):
        pass

    def on_moved(self, event):
        pass
