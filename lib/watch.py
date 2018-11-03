import os
import time

from watchdog.events import PatternMatchingEventHandler
from watchdog.observers import Observer


class ImageGenerationEventHandler(PatternMatchingEventHandler):
    def __init__(self, patterns, callback):
        super(ImageGenerationEventHandler, self).__init__(patterns=patterns)
        self.callback = callback

    def on_created(self, event):
        src_path = event.src_path.replace("\\", "/")
        print(src_path + " was created")
        frame_index = os.path.splitext(os.path.basename(src_path))[0].split("_")[-1]

        # Avoid the case of not finishing generating images
        time.sleep(0.1)

        self.callback(src_path, frame_index)

    def on_deleted(self, event):
        pass

    def on_modified(self, event):
        pass

    def on_moved(self, event):
        pass


def watch_image_generation(callback, directory_path):
    event_handler = ImageGenerationEventHandler(["*.jpg"], callback)
    observer = Observer()
    observer.schedule(event_handler, directory_path)
    observer.start()
    print("watching image generation...")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()

    observer.join()
