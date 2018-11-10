import os
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

        # firstmillis = int(round(time.perf_counter() * 1000))
        human_contour = find_human_contour(src)
        if human_contour is None:
            print("Not a human contour has detected in the image.")
            return
        # contourmillis = int(round(time.perf_counter() * 1000))
        # print("find_human_contour: {}ms".format(contourmillis - firstmillis))
        human = self.skeleton_implement.infer_skeleton(src)
        if human is None:
            print("Not a human has detected in the image.")
            return

        # skeletonmillis = int(round(time.perf_counter() * 1000))
        # print("infer_skeleton: {}ms".format(skeletonmillis-contourmillis))

        skeleton_test = SkeletonTest(human, human_contour, src.shape)
        if not skeleton_test.is_reliable():
            skeleton_test.report()
            print("This skeleton model is not reliable.")
            return
        # skeletontestmillis = int(round(time.perf_counter() * 1000))
        # print("skeleton_test: {}ms".format(skeletontestmillis - skeletonmillis))

        shadow = Shadow(src.shape, human, human_contour, arrangement_interval=10)
        # shadowmillis = int(round(time.perf_counter() * 1000))
        # print("shadow: {}ms".format(shadowmillis - skeletontestmillis))

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
