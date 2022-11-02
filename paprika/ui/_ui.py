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
        self.running_camera_thread = QThread(parent=self)
        self.running_camera_worker = RunningCameraWorker(self.camera)
        self.running_camera_worker.moveToThread(self.running_camera_thread)
        self.running_camera_worker.new_capture_signal.connect(
            self.on_new_running_capture
        )
        self.running_camera_thread.started.connect(self.running_camera_worker.start)

        self.analysis_thread = QThread(parent=self)
        self.analysis_worker = AnalysisWorker(
            self.camera, analysis_refresh_seconds, self.analysis_class
        )
        self.analysis_worker.moveToThread(self.analysis_thread)
        self.analysis_worker.new_analysis_signal.connect(self.on_new_analysis)
        self.analysis_thread.started.connect(self.analysis_worker.start)

        # create labels for the two cameras
        self.running_camera_label = QLabel()
        self.frozen_camera_label = QLabel()

        # create labels for the filter visualisations
        self.filter_image_labels = {}
        self.filter_text_labels = {}
        self.frames = {}
        for layer in selected_layers:
            self.frames[layer] = QFrame()
            self.filter_image_labels[layer] = []
            self.filter_text_labels[layer] = []
            for _ in range(filter_column_length * filter_row_length):
                self.filter_image_labels[layer].append(QLabel())
                self.filter_text_labels[layer].append(QLabel())

        # create labels for the saliency map
        self.saliency_image_label = QLabel()
        self.saliency_english_label = QLabel()
        self.saliency_german_label = QLabel()

        # create labels for the predictions
        self.prediction_score_labels = []
        self.prediction_german_labels = []
        self.prediction_english_labels = []
        self.prediction_image_labels = []
        for _ in range(nr_predictions):
            self.prediction_score_labels.append(QLabel())
            self.prediction_german_labels.append(QLabel())
            self.prediction_english_labels.append(QLabel())
            self.prediction_image_labels.append(
                [QLabel() for _ in range(nr_imagenet_images)]
            )

        # set up the layout on the 4 screens
        self.screen_widgets = get_full_screen_widgets(self.app)
        if screen_nr_camera_feed is not None:
            self.init_screen_camera_feed()
        if screen_nr_lower_filters is not None:
            self.init_screen_lower_filters()
        if screen_nr_higher_filters is not None:
            self.init_screen_higher_filters()
        if screen_nr_predictions is not None:
            self.init_screen_predictions()

        # add Ctrl+Q shortcut for quitting the app
        for screen_widget in self.screen_widgets:
            quit_shortcut = QShortcut(QKeySequence("Ctrl+Q"), screen_widget)
            quit_shortcut.activated.connect(self.app.quit)

        # run the two camera feed threads
        self.running_camera_thread.start()
        self.analysis_thread.start()

    def init_screen_camera_feed(self):
        # add the two camera feeds to a layout
        screen_widget = self.screen_widgets[0]
        layout = QGridLayout(screen_widget)
        frozen_camera_layout = image_with_explanation(
            self.frozen_camera_label,
            large_font_size,
            frozen_camera_german_text,
            frozen_camera_english_text,
            QLabel(),
            QLabel(),
        )
        running_camera_layout = image_with_explanation(
            self.running_camera_label,
            large_font_size,
            running_camera_german_text,
            running_camera_english_text,
            QLabel(),
            QLabel(),
        )
        layout.addLayout(frozen_camera_layout, 0, 0, Qt.AlignCenter)
        layout.addLayout(running_camera_layout, 1, 0, Qt.AlignCenter)

    def init_screen_lower_filters(self):
        # add the filter visualisations to a layout
        screen_widget = self.screen_widgets[1]
        layout = QGridLayout(screen_widget)

        for i in range(layers_per_screen):
            layer = selected_layers[i]
            frame = self.frames[layer]
            image_and_text_grid(
                self.filter_image_labels[layer],
                self.filter_text_labels[layer],
                small_font_size,
                frame,
            )
            if i == 0:
                visible_arrows = visible_arrows_from_camera
            else:
                visible_arrows = visible_arrows_between_filters
            layout.addLayout(
                arrow_column_layout(visible_arrows),
                0,
                layout.columnCount(),
                Qt.AlignCenter,
            )
            layout.addWidget(frame, 0, layout.columnCount(), Qt.AlignCenter)
        layout.addLayout(
            arrow_column_layout(visible_arrows_between_filters),
            0,
            layout.columnCount(),
            Qt.AlignCenter,
        )

    def init_screen_higher_filters(self):
        # add the filter visualisations to a layout
        screen_widget = self.screen_widgets[2]
        layout = QGridLayout(screen_widget)

        for i in range(layers_per_screen):
            layer = selected_layers[i + layers_per_screen]
            frame = self.frames[layer]
            image_and_text_grid(
                self.filter_image_labels[layer],
                self.filter_text_labels[layer],
                small_font_size,
                frame,
            )
            layout.addLayout(
                arrow_column_layout(visible_arrows_between_filters),
                0,
                layout.columnCount(),
                Qt.AlignCenter,
            )
            layout.addWidget(frame, 0, layout.columnCount(), Qt.AlignCenter)
        layout.addLayout(
            arrow_column_layout(visible_arrows_to_predictions),
            0,
            layout.columnCount(),
            Qt.AlignCenter,
        )

    def init_screen_predictions(self):
        # add the saliency map and predictions to a layout
        screen_widget = self.screen_widgets[3]
        layout = QGridLayout(screen_widget)

        saliency_layout = image_with_explanation(
            self.saliency_image_label,
            large_font_size,
            saliency_map_german_text("X"),
            saliency_map_english_text("Y"),
            self.saliency_german_label,
            self.saliency_english_label,
        )
        predictions_layout = score_text_image_grid(
            self.prediction_score_labels,
            self.prediction_german_labels,
            self.prediction_english_labels,
            self.prediction_image_labels,
            large_font_size,
            huge_font_size,
            medium_font_size,
            large_font_size,
        )

        layout.addLayout(saliency_layout, 0, 0, Qt.AlignCenter)
        layout.addLayout(predictions_layout, 1, 0, Qt.AlignVCenter)

    def on_new_running_capture(self, image: np.ndarray):
        """
        Slot for when a new running camera capture is obtained.
        It displays the new image.
        """
        pixmap = camera_image_to_pixmap(image)
        pixmap = resized_pixmap(pixmap, camera_capture_size)
        self.running_camera_label.setPixmap(pixmap)

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

        # update the saliency map
        saliency_image = analysis_dto.saliency_map
        pixmap = image_to_pixmap(saliency_image)
        pixmap = resized_pixmap(pixmap, camera_capture_size)
        self.saliency_image_label.setPixmap(pixmap)

        # update the predictions
        class_predictions = analysis_dto.class_predictions
        for i in range(nr_predictions):
            prediction = class_predictions[i]
            self.prediction_score_labels[i].setText(f"{round(prediction.score, 1)}%")
            english_text, german_text = prediction.label, prediction.label
            self.prediction_german_labels[i].setText(german_text)
            self.prediction_english_labels[i].setText(english_text)
            for j in range(nr_imagenet_images):
                imagenet_image = prediction.similar_images[j]
                pixmap = QPixmap(imagenet_image)
                if i != 0:
                    image_size = imagenet_small_size
                else:
                    image_size = imagenet_large_size
                self.prediction_image_labels[i][j].setPixmap(
                    resized_pixmap(pixmap, image_size)
                )

    def run(self):
        self.app.exec_()
