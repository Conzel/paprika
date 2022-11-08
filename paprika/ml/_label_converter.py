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
        lines[idx] = temp

    # Insert into dictionary
    for line in lines:
        num = int(line[0])
        # due to splitting at space -> red hat will be ['red', 'hat'] which will need to be put together
        label = " ".join(line[1:])
        labelDict[num] = (label,class_number)

    # For checking if every index is in the dict
    lastIndex = 0
    for label in labelDict:
        if label != lastIndex + 1:
            print("Probleme mit Index: " + str(label))
            print("   " + str(label) + " " + str(lastIndex))
        # letzten vorhandenen Index aktualisieren
        lastIndex = label

    return labelDict


if __name__ == "__main__":
    labelConverter()
