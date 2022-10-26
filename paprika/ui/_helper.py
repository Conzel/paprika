"""
Contains helper functions for the UI.
"""
from typing import List
from PIL import Image
from skimage.transform import resize
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QGuiApplication, QPixmap, QImage, QFont
from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout, QHBoxLayout, QGridLayout, QFrame

from paprika.ui._config import *


def get_full_screen_widgets(app: QGuiApplication):
    """
    For each connected display, returns a widget that is shown in full screen.
    """
    screens = app.screens()
    widgets = []
    used_screen_nrs = []
    for screen_nr in [screen_nr_camera_feed, screen_nr_lower_filters, screen_nr_higher_filters, screen_nr_predictions]:
        screen = screens[screen_nr]
        widget = QWidget()
        screen_geometry = screen.geometry()
        widget.move(screen_geometry.left(), screen_geometry.top())
        widget.showFullScreen()
        widget.setWindowTitle(f"{screen_nr}.{used_screen_nrs.count(screen_nr)}")
        used_screen_nrs.append(screen_nr)
        widgets.append(widget)
    return widgets


def middle_cropped_image(image: np.ndarray) -> np.ndarray:
    """
    Returns a middle crop of the image where height = width.
    Input image has shape (height, width, channel) and width > height.
    """
    height, width, channel = image.shape
    new_image = image[:, (width - height) // 2 : (width + height) // 2, :]
    return new_image


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """
    Converts an image in BGR format to RGB format.
    """
    return image[:, :, [2, 1, 0]]


def image_for_analysis_pil_image(image: np.ndarray) -> Image.Image:
    """
    Makes image suitable for ML analysis.
    It converts from BGR format to RGB, crops in the middle,
    resizes to 224*224 and converts to PIL Image.
    """
    analysis_image = bgr_to_rgb(image)
    analysis_image = middle_cropped_image(analysis_image)
    analysis_image = resize(analysis_image, (224, 224), anti_aliasing=True)
    analysis_image = Image.fromarray(analysis_image, mode="RGB")
    return analysis_image


def image_for_analysis(image: np.ndarray) -> np.ndarray:
    """
    Makes image suitable for ML analysis.
    It converts from BGR format to RGB, crops in the middle,
    resizes to 224*224 and converts to np.uint8.
    RGB values are in [0, 255].

    image: in BGR format, values in [0, 255]
    """
    analysis_image = bgr_to_rgb(image)
    analysis_image = middle_cropped_image(analysis_image)
    analysis_image = analysis_image / 255
    analysis_image = resize(analysis_image, (224, 224), anti_aliasing=True)
    analysis_image = analysis_image * 255
    analysis_image = np.require(analysis_image, np.uint8, "C")
    return analysis_image


def image_to_pixmap(image: np.ndarray) -> QPixmap:
    """
    Returns image as QPixmap.

    image: in RGB format, values in [0, 255]
    """
    height, width, channel = image.shape
    bytes_per_line = 3 * width
    image = np.require(image, np.uint8, "C")
    image = QImage(image, width, height, bytes_per_line, QImage.Format_RGB888)
    return QPixmap(image)


def camera_image_to_pixmap(captured_image: np.ndarray) -> QPixmap:
    """
    Returns captured_image as QPixmap in RGB format and cropped in the middle.

    captured_image: in BGR format, values in [0, 255]
    """
    cropped_image = middle_cropped_image(captured_image)
    cropped_image = bgr_to_rgb(cropped_image)
    return image_to_pixmap(cropped_image)


def resized_pixmap(pixmap: QPixmap, size: int) -> QPixmap:
    """
    Returns the pixmap rescaled to height size.
    """
    return pixmap.scaledToHeight(size)


def image_with_explanation(
    image_label: QLabel,
    font_size: int,
    german_text: str,
    english_text: str,
    german_text_label: QLabel,
    english_text_label: QLabel,
) -> QVBoxLayout:
    """
    Returns a QVBoxLayout in which there are three aligned QLabels containing the image,
    the German text and the English text.
    """
    layout = QVBoxLayout()
    layout.addWidget(image_label)

    german_text_label.setText(german_text)
    german_text_label.setFont(QFont(german_font, font_size))
    german_text_label.setStyleSheet(f"color: {german_colour}")
    german_text_label.setAlignment(Qt.AlignCenter)

    english_text_label.setText(english_text)
    english_text_label.setFont(QFont(english_font, large_font_size))
    english_text_label.setStyleSheet(f"color: {english_colour}")
    english_text_label.setAlignment(Qt.AlignCenter)

    layout.addWidget(german_text_label)
    layout.addWidget(english_text_label)
    return layout


def image_and_text_grid(
    image_labels: List[QLabel], text_labels: List[QLabel], font_size: int, frame: QFrame):
    """
    Creates a QGridLayout with filter_column_length columns and filter_row_length rows.
    It organises the visualisation images in this grid. Below each image it adds a text label.
    The QGridLayout is created with frame as its parent.
    """
    grid_layout = QGridLayout(frame)
    grid_layout.setContentsMargins(*frame_margin_layer)
    frame.setFrameShape(QFrame.Box)
    frame.setStyleSheet(f"color: {german_colour}")
    frame.setLineWidth(frame_width)

    i = 0
    for column in range(filter_column_length):
        for row in range(filter_row_length):
            v_layout = QVBoxLayout()
            image_label = image_labels[i]
            text_label = text_labels[i]
            text_label.setFont(QFont(german_font, font_size))
            text_label.setStyleSheet(f"color: {german_colour}")
            text_label.setAlignment(Qt.AlignCenter)
            v_layout.addWidget(image_label)
            v_layout.addWidget(text_label)
            v_layout.addSpacing(vertical_spacing_filters)
            i += 1
            grid_layout.addLayout(v_layout, column, row)
    grid_layout.setHorizontalSpacing(horizontal_spacing_filters)
    return grid_layout


def score_text_image_grid(
    score_labels,
    german_labels,
    english_labels,
    image_labels,
    font_size_score,
    larger_font_size_score,
    font_size_label,
    larger_font_size_label,
) -> QVBoxLayout:
    """
    Returns a QVBoxLayout with nr_prediction rows.
    Each row contains the score for the label, its name in German and in English
    and nr_imagenet_images with the same label.
    """
    v_layout = QVBoxLayout()
    for i_pred in range(nr_predictions):
        h_layout = QHBoxLayout()
        h_layout.addSpacing(predictions_edge_spacing)
        # add the prediction score to the row
        h_layout.addWidget(score_labels[i_pred], stretch=1)
        # add the German and English labels to row
        label_v_layout = QVBoxLayout()
        label_v_layout.addSpacing(predictions_labels_spacing)
        label_v_layout.addWidget(german_labels[i_pred])
        label_v_layout.addWidget(english_labels[i_pred])
        label_v_layout.addSpacing(predictions_labels_spacing)
        h_layout.addLayout(label_v_layout, stretch=3)
        # add the images to the row
        image_h_layout = QHBoxLayout()
        for i_img in range(nr_imagenet_images):
            image_h_layout.addWidget(image_labels[i_pred][i_img])
            if i_img != nr_imagenet_images - 1:
                image_h_layout.addStretch()
        h_layout.addLayout(image_h_layout, stretch=7)
        h_layout.addSpacing(predictions_edge_spacing)
        # add the row to the column
        v_layout.addLayout(h_layout)
        v_layout.addSpacing(predictions_bottom_spacing)
        # set fonts with larger font size for the first prediction
        if i_pred != 0:
            score_labels[i_pred].setFont(QFont(german_font, font_size_score))
            german_labels[i_pred].setFont(QFont(german_font, font_size_label))
            english_labels[i_pred].setFont(QFont(english_font, font_size_label))
        else:
            score_labels[i_pred].setFont(QFont(german_font, larger_font_size_score))
            german_labels[i_pred].setFont(QFont(german_font, larger_font_size_label))
            english_labels[i_pred].setFont(QFont(english_font, larger_font_size_label))
        score_labels[i_pred].setStyleSheet(f"color: {german_colour}")
        german_labels[i_pred].setStyleSheet(f"color: {german_colour}")
        english_labels[i_pred].setStyleSheet(f"color: {english_colour}")
    return v_layout
