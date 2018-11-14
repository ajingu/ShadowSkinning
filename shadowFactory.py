import sys
import time
import tensorflow as tf
from watchdog.observers import Observer

from lib.skeleton import SkeletonImplement
from lib.watch import ImageGenerationEventHandler
from lib.OSCclient import OSCclient

target_directory_path = "./images"

model = "cmu"
gpu_memory_fraction = 0.3
target_size = (512, 640)

ip = "127.0.0.1"
port = 5005
sleep_time = 0.1

arrangement_interval = 30

adjust_nose_position = False

binary_thresh = 240
maximum_inner_blob_area = 1000

if __name__ == '__main__':
    if len(sys.argv) > 0:
        target_directory_path = sys.argv[1]

    gpuConfig = tf.ConfigProto(
        gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction),
        device_count={'GPU': 1})

    skeleton_implement = SkeletonImplement(model, target_size, gpuConfig, adjust_nose_position)

    osc_client = OSCclient(ip, port, sleep_time)

    event_handler = ImageGenerationEventHandler(["*.jpg"], skeleton_implement, osc_client, arrangement_interval,
                                                binary_thresh, maximum_inner_blob_area)
    observer = Observer()
    observer.schedule(event_handler, target_directory_path)
    observer.start()
    print("watching image generation...")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()

    observer.join()
