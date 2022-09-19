"""
Contains helper functions for the UI.
"""
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QGuiApplication, QPixmap, QImage, QFont
from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout

from paprika.ui._config import *


def get_full_screen_widgets(app: QGuiApplication):
    """
    For each connected display, returns a widget that is shown in full screen.
    """
    screens = app.screens()
    widgets = []
    for screen in screens:
        widget = QWidget()
        screen_geometry = screen.geometry()
        widget.move(screen_geometry.left(), screen_geometry.top())
        widget.showFullScreen()
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


def camera_image_to_pixmap(captured_image: np.ndarray) -> QPixmap:
    """
    Returns captured_image as QPixmap in RGB format and cropped in the middle.
    captured_image: in BGR format
    """
    cropped_image = middle_cropped_image(captured_image)
    height, width, channel = cropped_image.shape
    bytes_per_line = 3 * width
    cropped_image = np.require(cropped_image, np.uint8, "C")
    image = QImage(
        cropped_image, width, height, bytes_per_line, QImage.Format_RGB888
    ).rgbSwapped()
    return QPixmap(image)


def resize_pixmap(pixmap: QPixmap, size: int) -> QPixmap:
    """
    Returns the pixmap rescaled to size x size dimension.
    """
    return pixmap.scaled(size, size)


def image_with_explanation(
    image_label: QLabel, font_size: int, german_text: str, english_text: str
) -> QVBoxLayout:
    """
    Returns a QVBoxLayout in which there are three aligned QLabels containing the image,
    the German text and the English text.
    """
    layout = QVBoxLayout()
    layout.addWidget(image_label)

    german_text_label = QLabel()
    german_text_label.setText(german_text)
    german_text_label.setFont(QFont(german_font, font_size))
    german_text_label.setStyleSheet(f"color: {german_colour}; text-align: center;")
    german_text_label.setAlignment(Qt.AlignCenter)

    english_text_label = QLabel()
    english_text_label.setText(english_text)
    english_text_label.setFont(QFont(english_font, large_font_size))
    english_text_label.setStyleSheet(f"color: {english_colour}; text-align: center;")
    english_text_label.setAlignment(Qt.AlignCenter)

    layout.addWidget(german_text_label)
    layout.addWidget(english_text_label)
    return layout
