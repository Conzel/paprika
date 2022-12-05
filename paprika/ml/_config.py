"""
Contains constants for the ML analysis.
"""

imagenet_relative_path = "../imagenet/"
visualisations_relative_path = "visualisations/400/"
grouped_classes_relative_path = "grouped_classes.csv"
filter_numbers_per_layer = {
    "mixed3a": 256,
    "mixed3b": 480,
    "mixed4a": 508,
    "mixed4b": 512,
    "mixed4c": 512,
    "mixed4d": 528,
    "mixed4e": 832,
    "mixed5a": 832,
    "mixed5b": 1024,
}
filter_strings_to_numbers = {
    "mixed3a": 70,
    "mixed3b": 78,
    "mixed4a": 87,
    "mixed4b": 95,
    "mixed4c": 103,
    "mixed4d": 111,
    "mixed4e": 119,
    "mixed5a": 128,
    "mixed5b": 136,
}
device_name = "cuda"
