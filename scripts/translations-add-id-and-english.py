import csv
import time
from PIL import Image


def process_one_class(id1, id2, german_group_label):
    # id1, id2, german_group_label, english_group_label, english_labels_list
    print(id1)
    # show one image of class
    img = 1
    shown_image = False
    while not shown_image:
        try:
            image = Image.open(
                f"D:\imagenet\ILSVRC\Data\CLS-LOC\\train\\{id1}\\{id1}_{img}.jpeg"
            )
            image.show()
            shown_image = True
        except FileNotFoundError:
            img += 1
    # show all subclasses in group
    subclasses = []
    for translation in translations:
        if translation[1] == german_group_label:
            subclasses.append(translation[0])
    print(subclasses)
    # show list of English labels
    for labels_list_line in labels_list_lines:
        if id1 in labels_list_line:
            english_labels_list = labels_list_line
    print(english_labels_list[10:])

    english_group_label = input("english group label: ")
    grouped_classes_writer.writerow(
        [id1, id2, german_group_label, english_group_label, english_labels_list[10:]]
    )


ids = range(900, 1000)

with open("grouped_classes.csv", "a", newline="", encoding="utf-8") as result_file:
    grouped_classes_writer = csv.writer(result_file, delimiter="|")

    with open("old_labels.txt") as old_labels_file:
        old_labels_lines = old_labels_file.read().splitlines()

        with open("translations.csv", newline="", encoding="utf-8") as translations_csv:
            translations_reader = csv.reader(translations_csv, delimiter=",")
            translations = [row for row in translations_reader][1:]

            with open("..\paprika\ml\LOC_synset_mapping.txt") as labels_list_file:
                labels_list_lines = labels_list_file.read().splitlines()

                for i in ids:
                    translation = translations[i]
                    old_labels_line = old_labels_lines[i]
                    print()
                    print(translation)
                    print(old_labels_line)
                    process_one_class(
                        old_labels_line.split(" ")[0],
                        old_labels_line.split(" ")[1],
                        translation[1],
                    )
