import time

from watchdog.events import PatternMatchingEventHandler
from watchdog.observers import Observer


class ImageGenerationEventHandler(PatternMatchingEventHandler):
    def __init__(self, patterns, callback, directory_path, image_name):
        super(ImageGenerationEventHandler, self).__init__(patterns=patterns)
        self.callback = callback
        self.image_path = directory_path + "/" + image_name
        print(self.image_path)

    def on_created(self, event):
        print("image.jpg was created")
        self.callback(self.image_path)

    def on_deleted(self, event):
        pass

    def on_modified(self, event):
        print("image.jpg was modified")
        self.callback(self.image_path)

    def on_moved(self, event):
        print("image.jpg was moved")
        self.callback(self.image_path)

    # TODO: いらないやつをpassしてみる


def watch_image_generation(callback, directory_path, image_name):
    event_handler = ImageGenerationEventHandler(["*.jpg"], callback, directory_path, image_name)
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
