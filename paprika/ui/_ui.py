import sys

from PyQt5.QtCore import QObject
from PyQt5.QtWidgets import QApplication, QGridLayout

from paprika.cam import Camera
from paprika.ui._camera_thread import FrozenCameraThread, RunningCameraThread
from paprika.ui._config import *
from paprika.ui._helper import *


class UserInterface(QObject):
    """
    UI suitable for 4 FHD screens in portrait mode.
    """

    def __init__(self, camera: Camera):
        super().__init__()
        self.app = QApplication(sys.argv)
        self.camera = camera
        self.running_camera_thread = RunningCameraThread(self, self.camera)
        self.frozen_camera_thread = FrozenCameraThread(
            self, self.camera, frozen_camera_refresh_seconds
        )

        self.running_camera_label = QLabel()
        self.frozen_camera_label = QLabel()

        self.screen_widgets = get_full_screen_widgets(self.app)
        self.init_screen_camera_feed()
        self.init_screen_lower_features()
        self.init_screen_higher_features()
        self.init_screen_predictions()

    def init_screen_camera_feed(self):
        # add the two camera feeds to a layout
        screen_widget = self.screen_widgets[screen_nr_camera_feed]
        layout = QGridLayout(screen_widget)
        frozen_camera_layout = image_with_explanation(
            self.frozen_camera_label,
            large_font_size,
            frozen_camera_german_text,
            frozen_camera_english_text,
        )
        running_camera_layout = image_with_explanation(
            self.running_camera_label,
            large_font_size,
            running_camera_german_text,
            running_camera_english_text,
        )
        layout.addLayout(frozen_camera_layout, 0, 0, Qt.AlignCenter)
        layout.addLayout(running_camera_layout, 1, 0, Qt.AlignCenter)

        # run the two camera feed threads
        self.running_camera_thread.new_capture_signal.connect(
            self.on_new_running_capture
        )
        self.running_camera_thread.run()
        self.frozen_camera_thread.new_capture_signal.connect(self.on_new_frozen_capture)
        self.frozen_camera_thread.run()

    def init_screen_lower_features(self):
        pass

    def init_screen_higher_features(self):
        pass

    def init_screen_predictions(self):
        pass

    def on_new_running_capture(self, image: np.ndarray):
        pixmap = camera_image_to_pixmap(image)
        pixmap = resize_pixmap(pixmap, camera_capture_size)
        self.running_camera_label.setPixmap(pixmap)

    def on_new_frozen_capture(self, image: np.ndarray):
        pixmap = camera_image_to_pixmap(image)
        pixmap = resize_pixmap(pixmap, camera_capture_size)
        self.frozen_camera_label.setPixmap(pixmap)

    def run(self):
        sys.exit(self.app.exec_())
