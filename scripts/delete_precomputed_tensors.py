"""
if precomputed tensors and (dictionaries) exist, they are deleted (so that the precomputations script can be run with different settings)

"""
import os
from os import path
import json
import torch

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


if __name__ == "__main__":

    source_folder = "../imagenet/"
    # choose layer
    layer_string = "mixed5b"
    layer_number = filter_strings_to_numbers[layer_string]

    all_classes = os.listdir(f"{source_folder}")
    for class_id in all_classes:
        if path.exists(f"{source_folder}{class_id}/{class_id}_activation_tensor.pt"):
            os.remove(f"{source_folder}{class_id}/{class_id}_activation_tensor.pt")
        if path.exists(f"{source_folder}{class_id}/{class_id}_dictionary.json"):
            os.remove(f"{source_folder}{class_id}/{class_id}_dictionary.json")
