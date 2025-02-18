"""
Contains helper functions for the UI.
"""
from typing import List
from PIL import Image
from skimage.transform import resize
import numpy as np
from PyQt5.QtCore import (
    QRect,
    Qt,
    QPropertyAnimation,
    QParallelAnimationGroup,
    QSequentialAnimationGroup,
)
from PyQt5.QtGui import QGuiApplication, QPixmap, QImage, QFont, QFontDatabase
from PyQt5.QtWidgets import (
    QWidget,
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QFrame,
    QGraphicsOpacityEffect,
)

from paprika.ui._config import *


def add_myriad_pro_fonts():
    """
    Adds the Myriad Pro Regular and Light fonts to the font database.
    """
    font_database = QFontDatabase()
    font_database.addApplicationFont("../paprika/ui/MyriadPro/MyriadPro-Light.otf")
    font_database.addApplicationFont("../paprika/ui/MyriadPro/MyriadPro-Regular.otf")


def get_full_screen_widgets(app: QGuiApplication):
    """
    For each connected display, returns a widget that is shown in full screen.
    """
    screens = app.screens()
    widgets = []
    used_screen_nrs = []
    for screen_nr in [
        screen_nr_camera_feed,
        screen_nr_lower_filters,
        screen_nr_higher_filters,
        screen_nr_predictions,
    ]:
        screen = screens[screen_nr]
        widget = QWidget()
        screen_geometry = screen.geometry()
        widget.move(screen_geometry.left(), screen_geometry.top())
        widget.showFullScreen()
        widget.setWindowTitle(f"{screen_nr}.{used_screen_nrs.count(screen_nr)}")
        widget.setStyleSheet(f"background-color: {background_colour}")
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
    It converts from BGR format to RGB, crops in the middle
    and converts to PIL Image.
    """
    analysis_image = bgr_to_rgb(image)
    analysis_image = middle_cropped_image(analysis_image)
    # analysis_image = resize(analysis_image, (224, 224), anti_aliasing=True)
    analysis_image = Image.fromarray(analysis_image, mode="RGB")
    return analysis_image


def image_for_analysis(image: np.ndarray) -> np.ndarray:
    """
    Makes image suitable for ML analysis.
    It converts from BGR format to RGB, crops in the middle
    and converts to np.uint8.
    RGB values are in [0, 255].

    image: in BGR format, values in [0, 255]
    """
    analysis_image = bgr_to_rgb(image)
    analysis_image = middle_cropped_image(analysis_image)
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


def resized_pixmap_by_height(pixmap: QPixmap, height: int) -> QPixmap:
    """
    Returns the pixmap rescaled to given height.
    """
    return pixmap.scaledToHeight(height)


def resized_pixmap_by_width(pixmap: QPixmap, width: int) -> QPixmap:
    """
    Returns the pixmap rescaled to given width.
    """
    return pixmap.scaledToWidth(width)


def cropped_vertical_pixmap(pixmap: QPixmap, new_height: int) -> QPixmap:
    """
    Returns the cropped pixmap with the new height.
    The image is cropped in the middle.
    """
    top = (pixmap.height() - new_height) // 2
    rect = QRect(0, top, pixmap.width(), new_height)
    return pixmap.copy(rect)


def cropped_horizontal_pixmap(pixmap: QPixmap, new_width: int) -> QPixmap:
    """
    Returns the cropped pixmap with the new width.
    The image is cropped in the middle.
    """
    left = (pixmap.width() - new_width) // 2
    rect = QRect(left, 0, new_width, pixmap.height())
    return pixmap.copy(rect)


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
    layout.addSpacing(camera_capture_spacing)
    image_label.setAlignment(Qt.AlignCenter)

    german_text_label.setText(german_text)
    german_text_label.setFont(QFont(german_font, font_size))
    german_text_label.setStyleSheet(f"color: {german_colour}")
    german_text_label.setAlignment(Qt.AlignCenter)

    english_text_label.setText(english_text)
    english_text_label.setFont(QFont(english_font, large_font_size))
    english_text_label.setStyleSheet(f"color: {english_colour}")
    english_text_label.setAlignment(Qt.AlignCenter)

    layout.addWidget(german_text_label)
    layout.addSpacing(camera_capture_labels_spacing)
    layout.addWidget(english_text_label)
    return layout


def arrow_column_layout(visible_arrows) -> (QVBoxLayout, List[QLabel]):
    """
    Returns QVBoxLayout with nr_arrows of arrows, where only the arrows at the indices given by the list
    visible_arrows are coloured gray and all the other ones have the background colour (thus are not visible).
    Also returns the list of arrows as QLabel list.
    """
    layout = QVBoxLayout()
    arrows = []
    for i in range(nr_arrows):
        # arrow_label = QLabel("➞")
        arrow_label = QLabel("➔")
        arrows.append(arrow_label)
        arrow_label.setFont(QFont(english_font, large_font_size))
        if i in visible_arrows:
            arrow_label.setStyleSheet(f"color: {arrow_and_box_colour}")
        else:
            arrow_label.setStyleSheet(f"color: {background_colour}")
        arrow_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(arrow_label)
        if i != nr_arrows - 1:
            layout.addSpacing(arrow_spacing)
    return layout, arrows


def three_dots_label():
    label = QLabel("• • •")
    label.setFont(QFont(german_font, large_font_size))
    label.setStyleSheet(f"color: {arrow_and_box_colour}")
    return label


def image_and_text_grid(
    image_labels: List[QLabel], font_size: int, frame: QFrame
):
    """
    Creates a QGridLayout with filter_column_length columns and filter_row_length rows.
    It organises the visualisation images in this grid.
    The QGridLayout is created with frame as its parent.
    """
    grid_layout = QGridLayout(frame)
    grid_layout.setContentsMargins(*frame_margin_layer)
    frame.setFrameShape(QFrame.Box)
    frame.setStyleSheet(f"color: {arrow_and_box_colour}")
    frame.setLineWidth(frame_width)

    i = 0
    for column in range(filter_column_length):
        for row in range(filter_row_length):
            v_layout = QVBoxLayout()
            image_label = image_labels[i]
            v_layout.addWidget(image_label)
            v_layout.addSpacing(vertical_spacing_filters)
            i += 1
            grid_layout.addLayout(v_layout, column, row)
    grid_layout.addWidget(
        three_dots_label(),
        grid_layout.rowCount(),
        grid_layout.columnCount() - 1,
        Qt.AlignCenter,
    )
    grid_layout.setHorizontalSpacing(horizontal_spacing_filters)
    return grid_layout


def score_text_image_grid(
    score_labels,
    german_labels,
    english_labels,
    image_labels,
    font_size_score,
    font_size_label,
) -> QVBoxLayout:
    """
    Returns a QVBoxLayout with nr_prediction rows.
    Each row contains the score for the label, its name in German and in English
    and nr_imagenet_images with the same label.
    """
    v_layout = QVBoxLayout()
    for i_pred in range(nr_predictions):
        h_layout_frame = QFrame()
        h_layout_frame.setFixedWidth(predictions_row_width)
        h_layout = QHBoxLayout(h_layout_frame)
        h_layout.addSpacing(predictions_edge_spacing[0])
        # add the prediction score to the row
        score_labels[i_pred].setFixedWidth(score_width)
        h_layout.addWidget(score_labels[i_pred])
        # add the German and English labels to row
        label_v_layout = QVBoxLayout()
        label_v_layout_widget = QWidget()
        label_v_layout_widget.setLayout(label_v_layout)
        label_v_layout_widget.setFixedWidth(label_width)
        label_v_layout.addSpacing(predictions_labels_spacing)
        label_v_layout.addWidget(german_labels[i_pred])
        label_v_layout.addWidget(english_labels[i_pred])
        label_v_layout.addSpacing(predictions_labels_spacing)
        h_layout.addWidget(label_v_layout_widget)
        # add the images to the row
        image_h_layout = QHBoxLayout()
        image_h_layout_widget = QWidget()
        image_h_layout_widget.setLayout(image_h_layout)
        image_h_layout_widget.setFixedWidth(images_width)
        for i_img in range(nr_imagenet_images):
            image_h_layout.addWidget(image_labels[i_pred][i_img])
            if i_img != nr_imagenet_images - 1:
                image_h_layout.addStretch()
        image_h_layout.setContentsMargins(
            0, predictions_bottom_top_margin, 0, predictions_bottom_top_margin
        )
        image_h_layout.setSpacing(0)
        h_layout.addWidget(image_h_layout_widget)
        h_layout.addSpacing(predictions_edge_spacing[1])
        h_layout.setContentsMargins(0, 0, 0, 0)
        # add the row to the column
        v_layout.addWidget(h_layout_frame)
        if i_pred != nr_predictions - 1:
            v_layout.addSpacing(predictions_bottom_spacing)
        # set fonts
        score_labels[i_pred].setFont(QFont(german_font, font_size_score))
        german_labels[i_pred].setFont(QFont(german_font, font_size_label))
        english_labels[i_pred].setFont(QFont(english_font, font_size_label))
        score_labels[i_pred].setStyleSheet(f"color: {german_colour}")
        german_labels[i_pred].setStyleSheet(f"color: {german_colour}")
        english_labels[i_pred].setStyleSheet(f"color: {english_colour}")
        if i_pred == 0:
            h_layout_frame.setStyleSheet(
                f"background-color: {top_prediction_background_colour}"
            )
    return v_layout


def add_opacity_animations(labels: List[QLabel], animation: QSequentialAnimationGroup):
    """
    Adds animations to animation such that all labels fade in, stay opaque and fade out
    at the same time.
    animation_milliseconds gives the length of the three phases
    """
    fade_in_animation_group = QParallelAnimationGroup()
    fade_out_animation_group = QParallelAnimationGroup()
    opaque_animation_group = QParallelAnimationGroup()
    for label in labels:
        effect = QGraphicsOpacityEffect()
        label.setGraphicsEffect(effect)
        # arrows fade in
        fade_in_animation = QPropertyAnimation(effect, b"opacity")
        fade_in_animation.setDuration(animation_milliseconds[0])
        fade_in_animation.setStartValue(0)
        fade_in_animation.setEndValue(1)
        fade_in_animation_group.addAnimation(fade_in_animation)
        # arrows stay opaque
        opaque_animation = QPropertyAnimation(effect, b"opacity")
        opaque_animation.setDuration(animation_milliseconds[1])
        opaque_animation.setStartValue(1)
        opaque_animation.setEndValue(1)
        opaque_animation_group.addAnimation(opaque_animation)
        # arrows fade out
        fade_out_animation = QPropertyAnimation(effect, b"opacity")
        fade_out_animation.setDuration(animation_milliseconds[2])
        fade_out_animation.setStartValue(1)
        fade_out_animation.setEndValue(0)
        fade_out_animation_group.addAnimation(fade_out_animation)
    animation.addAnimation(fade_in_animation_group)
    animation.addAnimation(opaque_animation_group)
    animation.addAnimation(fade_out_animation_group)
