"""
Contains constants for the UI.
"""

# image and spacing sizes
camera_capture_size = 700
filter_size = 218
vertical_spacing_filters = 20
horizontal_spacing_filters = 15

frozen_camera_refresh_seconds = 3

large_font_size = 20
medium_font_size = 16
small_font_size = 12

german_font = "Yu Gothic UI Semibold"
english_font = "Yu Gothic UI Semilight"

german_colour = "#4b4b4b"
english_colour = "#212121"

# screen number for each action
screen_nr_camera_feed = 1
screen_nr_lower_filters = 0
screen_nr_higher_filters = 2
screen_nr_predictions = 3

selected_layers = ["mixed3b", "mixed4b", "mixed4e", "mixed5b"]  # all layers to be shown

layers_per_screen = 2  # number of layers shown in one screen
filter_column_length = 6  # number of filters shown in one column
filter_row_length = 2  # number of filters shown in one row

# text for screen with camera feed
frozen_camera_german_text = "Das Bild, das gerade analysiert wird"
frozen_camera_english_text = "The image that is currently being analysed"
running_camera_german_text = "Der aktuelle Kamera-Feed"
running_camera_english_text = "The current camera feed"

# text for screens with lower and higher filters
