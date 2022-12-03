from typing import NamedTuple

import pandas

from paprika.ml._config import grouped_classes_relative_path


# immutable struct
class Subclass(NamedTuple):
    id: str
    german_group: str
    english_group: str


class GroupLabelId(NamedTuple):
    german_group: str
    english_group: str


class GroupScore:
    def __init__(self, id: str, max_score: float):
        self.id = id
        self.max_score = max_score


def construct_subclass_group_dict():
    """
    Returns a dictionary mapping the positions 1, 2, ..., 1000 to Subclass structs containing
    id (e.g. n02119789), german_group (e.g. Fuchs) and english_group (e.g. Fox).
    """
    classes_df = pandas.read_csv(grouped_classes_relative_path, delimiter="|")
    subclass_group_dict = {}
    for _, row in classes_df.iterrows():
        subclass_group_dict[row["position"]] = Subclass(
            row["id"], row["german_group"], row["english_group"]
        )
    return subclass_group_dict


def construct_group_score_dict(subclass_group_dict):
    """
    Returns a dictionary based on subclass_group_dict. It maps each GroupLabelId struct
    to the first id of the group and a max_score initialised to 0.
    """
    group_score_dict = {}
    for position, subclass in subclass_group_dict.items():
        if (subclass.german_group, subclass.english_group) not in group_score_dict:
            group_score_dict[
                GroupLabelId(subclass.german_group, subclass.english_group)
            ] = GroupScore(subclass.id, 0.0)
    return group_score_dict


def labelConverter():
    labelDict = {}
    lines = []
    with open("old_labels.txt") as file:
        # read all lines into the lines-array
        lines = file.readlines()

    # remove leading and trailing whitespaces
    lines = [line.strip() for line in lines]

    for idx, line in enumerate(lines):
        # save number of imagenet class
        class_number = line[:9]
        # remove the first 10 letters/numbers (n02119789 1 kit fox -> 1 kit fox)
        temp = line[10:]
        # Split String at every spacecharacter to extract the numbers at the beginning
        temp = temp.split(" ")
        lines[idx] = (temp, class_number)

    # Insert into dictionary
    for line in lines:
        num = int(line[0][0])
        # due to splitting at space -> red hat will be ['red', 'hat'] which will need to be put together
        label = " ".join(line[0][1:])
        labelDict[num] = (label, line[1])

    # For checking if every index is in the dict
    lastIndex = 0
    for label in labelDict:
        if label != lastIndex + 1:
            print("Probleme mit Index: " + str(label))
            print("   " + str(label) + " " + str(lastIndex))
        # letzten vorhandenen Index aktualisieren
        lastIndex = label

    return labelDict
