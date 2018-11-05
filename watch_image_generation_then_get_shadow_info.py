import time

from watchdog.observers import Observer

from lib.skeleton import SkeletonImplement
from lib.watch import ImageGenerationEventHandler

TARGET_DIRECTORY_PATH = "./images"

if __name__ == '__main__':
    skeleton_implement = SkeletonImplement(target_size=(368, 368))

    event_handler = ImageGenerationEventHandler(["*.jpg"], skeleton_implement)
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
