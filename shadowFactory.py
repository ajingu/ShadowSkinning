import time
import tensorflow as tf
from watchdog.observers import Observer

from lib.skeleton import SkeletonImplement
from lib.watch import ImageGenerationEventHandler
from lib.OSCclient import OSCclient

TARGET_DIRECTORY_PATH = "./images"

model = "cmu"
gpu_memory_fraction = 0.5
target_size = (640, 512)

ip = "127.0.0.1"
port = 5005
sleep_time = 0.1

arrangement_interval = 10

if __name__ == '__main__':
    gpuConfig = tf.ConfigProto(
        gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction),
        device_count={'GPU': 1})

    skeleton_implement = SkeletonImplement(model, target_size, gpuConfig)

    osc_client = OSCclient(ip, port, sleep_time)

    event_handler = ImageGenerationEventHandler(["*.jpg"], skeleton_implement, osc_client, arrangement_interval)
    observer = Observer()
    observer.schedule(event_handler, TARGET_DIRECTORY_PATH)
    observer.start()
    print("watching image generation...")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()

    observer.join()
