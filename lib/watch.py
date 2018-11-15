import os
import time
import traceback

from tf_pose.common import read_imgfile
from watchdog.events import PatternMatchingEventHandler

from lib.contour import find_human_contour
from lib.skeleton import SkeletonTest
from lib.shadow import Shadow


class ImageGenerationEventHandler(PatternMatchingEventHandler):
    def __init__(self, patterns, skeleton_implement, oscClient, arrangement_interval, binary_thresh,
                 maximum_inner_blob_area):
        super(ImageGenerationEventHandler, self).__init__(patterns=patterns)
        self.skeleton_implement = skeleton_implement
        self.oscClient = oscClient
        self.arrangement_interval = arrangement_interval
        self.binary_thresh = binary_thresh
        self.maximum_inner_blob_area = maximum_inner_blob_area

    def on_created(self, event):
        try:
            src_path = event.src_path.replace("\\", "/")
            print(src_path + " was created")
            frame_index = os.path.splitext(os.path.basename(src_path))[0].split("_")[-1]

            # Avoid the case of not finishing generating images
            time.sleep(0.01)

            src = read_imgfile(src_path, None, None)
            if src is None:
                print(src_path + " is not found.")
                return

            print("Frame Index:", frame_index)

            # rotation
            src = src.transpose(1, 0, 2)[::-1]

            # firstmillis = int(round(time.perf_counter() * 1000))
            human_contour = find_human_contour(src, self.binary_thresh, self.maximum_inner_blob_area)
            if human_contour is None:
                os.remove(src_path)
                print("Not a human contour has detected in the image.")
                return
            # contourmillis = int(round(time.perf_counter() * 1000))
            # print("find_human_contour: {}ms".format(contourmillis - firstmillis))
            human = self.skeleton_implement.infer_skeleton(src)
            if human is None:
                os.remove(src_path)
                print("Not a human has detected in the image.")
                return

            # skeletonmillis = int(round(time.perf_counter() * 1000))
            # print("infer_skeleton: {}ms".format(skeletonmillis-contourmillis))

            skeleton_test = SkeletonTest(human, human_contour, src.shape)
            if not skeleton_test.is_reliable():
                skeleton_test.report()
                os.remove(src_path)
                print("This skeleton model is not reliable.")
                return
            # skeletontestmillis = int(round(time.perf_counter() * 1000))
            # print("skeleton_test: {}ms".format(skeletontestmillis - skeletonmillis))

            shadow = Shadow(src.shape, human, human_contour, self.arrangement_interval)
            # shadowmillis = int(round(time.perf_counter() * 1000))
            # print("shadow: {}ms".format(shadowmillis - skeletontestmillis))

            print("The number of body parts:", len(shadow.body_part_positions))
            print("The number of vertices:", len(shadow.vertex_positions))
            print("The number of triangle vertices:", len(shadow.triangle_vertex_indices))
            self.oscClient.send(shadow, frame_index)

            polygons_image = src.copy()
            self.skeleton_implement.draw_skeleton(polygons_image, human)

            import cv2
            from lib.draw import draw_triangle

            for pt1_index, pt2_index, pt3_index in shadow.triangle_vertex_indices:
                draw_triangle(polygons_image,
                              shadow.vertex_positions[pt1_index],
                              shadow.vertex_positions[pt2_index],
                              shadow.vertex_positions[pt3_index])
        except Exception:
            traceback.print_exc()
            pass

    def on_deleted(self, event):
        pass

    def on_modified(self, event):
        pass

    def on_moved(self, event):
        pass
