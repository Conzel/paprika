"""
Contains constants for the UI.
"""

# image and spacing sizes
camera_capture_size = 750
camera_capture_spacing = (
    10
)  # spacing between the camera capture images and the texts underneath
camera_capture_labels_spacing = (
    8
)  # spacing between the texts underneath the camera captures
filter_size = 390
similar_image_height = 160
similar_image_vertical_width = 120  # the width that a vertical image is cropped to
similar_images_width_sum = 560  # the max width sum of the similar images in one line
vertical_spacing_filters = 20
frame_margin_layer = (
    15,
    15,
    15,
    15,
)  # left, top, right, bottom margins between layer content and frame
frame_width = 3
horizontal_spacing_filters = 15
predictions_labels_spacing = (
    18  # increasing this decreases the space between the German and the English label
)
predictions_edge_spacing = [
    6,
    4,
]  # spacing on the left and right sides of one line of predictions with images
predictions_bottom_top_margin = 10
predictions_bottom_spacing = 3  # spacing under the predictions
variable_images_width = (
    True
)  # if the labels are shorter, more space is accorded to the images
predictions_row_width = 1056  # width of the rows containing score, label and images
score_width = 110
label_width = 315
images_width = 585

# arrow configs
nr_arrows = 10
arrow_spacing = 143
visible_arrows_from_camera = [3]
visible_arrows_to_predictions = [5]
visible_arrows_between_filters = range(nr_arrows)
animation_milliseconds = (250, 500, 600)  # fade in, opaque, fade out

analysis_refresh_seconds = 2
two_camera_images = False

large_font_size = 25
medium_font_size = 23
small_font_size = 16

german_font = "Myriad Pro"
english_font = "Myriad Pro Light"

# german_colour = "#4b4b4b"
# english_colour = "#212121"
# background_colour = "#f0f0f0"
# top_prediction_background_colour = "#e2e2e2"

# violet: #2f009d  lighter version: #f8f5ff

german_colour = "#000000"
english_colour = "#212121"
arrow_and_box_colour = "#212121"
background_colour = "#ffffff"
top_prediction_background_colour = "#e2e2e2"

# screen number for each action
# set each of them to 0, 1, 2 or 3 (multiple can have the same value)
screen_nr_camera_feed = 0
screen_nr_lower_filters = 3
screen_nr_higher_filters = 2
screen_nr_predictions = 1

selected_layers = ["mixed4a", "mixed4b", "mixed4e", "mixed5a"]  # all layers to be shown

layers_per_screen = 2  # number of layers shown in one screen
filter_column_length = 4  # number of filters shown in one column
filter_row_length = 1  # number of filters shown in one row

nr_predictions = 5  # the number of top predictions to show
nr_imagenet_images = 3  # the number of imagenet images to show per prediction

# text for screen with camera feed
frozen_camera_german_text = "Analysiertes Bild"
frozen_camera_english_text = "Analysed image"
running_camera_german_text = "Kamerabild"
running_camera_english_text = "Camera image"

# text for screen with predictions
saliency_map_german_text = "Wichtige Teile des Bildes für die Klassifizierung"
saliency_map_english_text = "Important parts of the image for classification"
