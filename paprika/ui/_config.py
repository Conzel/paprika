"""
Contains constants for the UI.
"""

# image and spacing sizes
camera_capture_size = 750
filter_size = 370
imagenet_small_size = 110
imagenet_large_size = 130
vertical_spacing_filters = 30
frame_margin_layer = (
    20,
    20,
    20,
    20,
)  # left, top, right, bottom margins between layer content and frame
frame_width = 4
horizontal_spacing_filters = 15
predictions_labels_spacing = (
    20  # increasing this decreases the space between the German and the English label
)
predictions_edge_spacing = (
    20  # spacing on the left and right sides of one line of predictions with images
)
predictions_bottom_spacing = 15  # spacing under the predictions
nr_arrows = 9
arrow_spacing = 119
visible_arrows_from_camera = [2]
visible_arrows_to_predictions = [5]
visible_arrows_between_filters = range(nr_arrows)

analysis_refresh_seconds = 2

huge_font_size = 24
large_font_size = 20
medium_font_size = 16
small_font_size = 12

german_font = "Yu Gothic UI Semibold"
english_font = "Yu Gothic UI Semilight"

german_colour = "#4b4b4b"
english_colour = "#212121"
background_colour = "#f0f0f0"

# screen number for each action
# set each of them to 0, 1, 2 or 3 (multiple can have the same value)
screen_nr_camera_feed = 1
screen_nr_lower_filters = 1
screen_nr_higher_filters = 1
screen_nr_predictions = 1

selected_layers = ["mixed3b", "mixed4b", "mixed4e", "mixed5b"]  # all layers to be shown

layers_per_screen = 2  # number of layers shown in one screen
filter_column_length = 4  # number of filters shown in one column
filter_row_length = 1  # number of filters shown in one row

nr_predictions = 5  # the number of top predictions to show
nr_imagenet_images = 0  # the number of imagenet images to show per prediction

# text for screen with camera feed
frozen_camera_german_text = "Analysiertes Bild"
frozen_camera_english_text = "Analysed image"
running_camera_german_text = "Kamerabild"
running_camera_english_text = "Camera image"


# text for screens with lower and higher filters

# text for screen with predictions
def saliency_map_german_text(prediction: str):
    return f"Wichtige Teile des Bildes für die Klassifizierung"


def saliency_map_english_text(prediction: str):
    return f"Important parts of the image for classification"
