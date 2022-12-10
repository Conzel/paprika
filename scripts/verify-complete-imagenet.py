"""
Verify that all ImageNet folders have at least 732 images.
"""

import os

from paprika.ml._label_converter import construct_subclass_group_dict

def count_images_in_folder(path):
    files = os.listdir(path)
    files = [file for file in files if ".JPEG" in file]
    return len(files)

subclass_group_dict = construct_subclass_group_dict()
assert(len(subclass_group_dict.values()) == 1000)
for subclass in subclass_group_dict.values():
    class_id = subclass.id
    folder_path = f"../imagenet/{class_id}"
    nr_images = count_images_in_folder(folder_path)
    if nr_images < 732:
        print(f"{nr_images}    {class_id}")
