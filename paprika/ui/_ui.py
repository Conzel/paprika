import random
import sys

from PyQt5.QtCore import QObject
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import QApplication, QGridLayout, QShortcut

from paprika.cam import Camera
from paprika.ml import DummyAnalysis, Inceptionv1Analysis
from paprika.ui._ui_thread import *
from paprika.ui._config import *
from paprika.ui._helper import *


class UserInterface(QObject):
    """
    UI suitable for 4 FHD screens in portrait mode.
    """

    def __init__(self, camera: Camera, analysis_class: NeuralNetworkAnalysis):
        super().__init__()
        self.app = QApplication(sys.argv)

        self.analysis_class = analysis_class
        self.camera = camera

        # set up the two threads that capture camera images and connect to their signal
        self.running_camera_thread = QThread()
        self.running_camera_worker = RunningCameraWorker(self.camera)
        self.running_camera_worker.moveToThread(self.running_camera_thread)
        self.running_camera_worker.new_capture_signal.connect(
            self.on_new_running_capture
        )
        self.running_camera_thread.started.connect(self.running_camera_worker.start)
        self.frozen_camera_thread = QThread()
        self.frozen_camera_worker = FrozenCameraWorker(
            self.camera, frozen_camera_refresh_seconds
        )
        self.frozen_camera_worker.moveToThread(self.frozen_camera_thread)
        self.frozen_camera_worker.new_capture_signal.connect(self.on_new_frozen_capture)
        self.frozen_camera_thread.started.connect(self.frozen_camera_worker.start)

        # create labels for the two cameras
        self.running_camera_label = QLabel()
        self.frozen_camera_label = QLabel()

        # create labels for the filter visualisations
        self.filter_image_labels = {}
        self.filter_text_labels = {}
        for layer in selected_layers:
            self.filter_image_labels[layer] = []
            self.filter_text_labels[layer] = []
            for _ in range(filter_column_length * filter_row_length):
                self.filter_image_labels[layer].append(QLabel())
                self.filter_text_labels[layer].append(QLabel())

        # set up the layout on all of the 4 screens
        self.screen_widgets = get_full_screen_widgets(self.app)
        self.init_screen_camera_feed()
        self.init_screen_lower_filters()
        self.init_screen_higher_filters()
        self.init_screen_predictions()

        # add Ctrl+Q shortcut for quitting the app
        for screen_widget in self.screen_widgets:
            quit_shortcut = QShortcut(QKeySequence("Ctrl+Q"), screen_widget)
            quit_shortcut.activated.connect(self.app.quit)

        # run the two camera feed threads
        self.running_camera_thread.start()
        self.frozen_camera_thread.start()

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

    def init_screen_lower_filters(self):
        # add the filter visualisations to a layout
        screen_widget = self.screen_widgets[screen_nr_lower_filters]
        layout = QGridLayout(screen_widget)

        for i in range(layers_per_screen):
            layer = selected_layers[i]
            layer_layout = image_and_text_grid(
                self.filter_image_labels[layer],
                self.filter_text_labels[layer],
                small_font_size,
            )
            layout.addLayout(layer_layout, 0, i, Qt.AlignCenter)

    def init_screen_higher_filters(self):
        # add the filter visualisations to a layout
        screen_widget = self.screen_widgets[screen_nr_higher_filters]
        layout = QGridLayout(screen_widget)

        for i in range(layers_per_screen):
            layer = selected_layers[i + layers_per_screen]
            layer_layout = image_and_text_grid(
                self.filter_image_labels[layer],
                self.filter_text_labels[layer],
                small_font_size,
            )
            layout.addLayout(layer_layout, 0, i, Qt.AlignCenter)

    def init_screen_predictions(self):
        pass

    def on_new_running_capture(self, image: np.ndarray):
        """
        Slot for when a new running camera capture is obtained.
        It displays the new image.
        """
        pixmap = camera_image_to_pixmap(image)
        pixmap = resized_pixmap(pixmap, camera_capture_size)
        self.running_camera_label.setPixmap(pixmap)

    def on_new_frozen_capture(self, image: np.ndarray):
        """
        Slot for when a new frozen camera capture is obtained.
        It starts a thread for computing the new ML analysis.
        """
        self.analysis_thread = QThread()
        self.analysis_worker = AnalysisWorker(self.analysis_class, image)
        self.analysis_worker.moveToThread(self.analysis_thread)
        self.analysis_worker.new_analysis_signal.connect(self.on_new_analysis)
        self.analysis_thread.started.connect(self.analysis_worker.start)

        # signals and slots for deleting the worker and the thread
        self.analysis_worker.new_analysis_signal.connect(self.analysis_thread.quit)
        self.analysis_worker.new_analysis_signal.connect(
            self.analysis_worker.deleteLater
        )
        self.analysis_worker.new_analysis_signal.connect(
            self.analysis_thread.deleteLater
        )

        self.analysis_thread.start()

    def on_new_analysis(self, analysis_dto: AnalysisResultsDTO):
        """
        Slot for when a new ML analysis is obtained.
        It changes the elements obtained from the analysis as well as the frozen camera image.
        """
        # set camera feed to new capture
        image = analysis_dto.original_image
        pixmap = camera_image_to_pixmap(image)
        pixmap = resized_pixmap(pixmap, camera_capture_size)
        self.frozen_camera_label.setPixmap(pixmap)

        # update the filter images in each layer
        layer_filters = analysis_dto.layer_filters
        for layer in selected_layers:
            activated_filters = layer_filters[layer]
            for i in range(filter_column_length * filter_row_length):
                image_path, filter_id, filter_activation = activated_filters[i]
                pixmap = QPixmap(image_path)
                self.filter_image_labels[layer][i].setPixmap(
                    resized_pixmap(pixmap, filter_size)
                )
                self.filter_text_labels[layer][i].setText(
                    f"Filter {filter_id}  -  {round(filter_activation, 1)}%"
                )

    def run(self):
        self.app.exec_()
