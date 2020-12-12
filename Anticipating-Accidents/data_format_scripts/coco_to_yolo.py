import json
from collections import OrderedDict
from tqdm import tqdm

INSTANCES_PATH = "custom_annotations_train.json"
NAMES_PATH = "custom_annotations.names"
OUTPUT_FILE_PATH = "custom_annotations_train.txt"

coco = json.load(open(INSTANCES_PATH))
images = coco["images"]
annotations = coco["annotations"]
categories = coco["categories"]
replaced_name = {}

class_to_id = {}
id_to_class = {}
with open(NAMES_PATH, "r") as fd:
    index = 0
    for class_name in fd:
        class_name = class_name.strip()
        if len(class_name) != 0:
            id_to_class[index] = class_name
            class_to_id[class_name] = index
            index += 1

dataset = {}

for annotation in tqdm(annotations, desc="Parsing"):
    image_id = annotation["image_id"]
    category_id = annotation["category_id"]

    # Find image
    file_name = None
    image_height = 0
    image_width = 0
    for image in images:
        if image["id"] == image_id:
            file_name = image["file_name"]
            image_height = image["height"]
            image_width = image["width"]
            break

    if file_name is None:
        continue

    # Find class id
    class_id = None
    for category in categories:
        if category["id"] == category_id:
            category_name = category["name"]
            if category_name in replaced_name:
                category_name = replaced_name[category_name]

            class_id = class_to_id.get(category_name)
            break

    if class_id is None:
        continue

    # Calculate x,y,w,h
    x_center = annotation["bbox"][0] + annotation["bbox"][2] / 2
    x_center /= image_width
    y_center = annotation["bbox"][1] + annotation["bbox"][3] / 2
    y_center /= image_height
    width = annotation["bbox"][2] / image_width
    height = annotation["bbox"][3] / image_height

    if dataset.get(image_id):
        dataset[image_id][1].append(
            [class_id, x_center, y_center, width, height]
        )
    else:
        dataset[image_id] = [
            file_name,
            [[class_id, x_center, y_center, width, height]],
        ]

dataset = OrderedDict(sorted(dataset.items()))

with open(OUTPUT_FILE_PATH, "w") as fd:
    for image_id, bboxes in tqdm(dataset.items(), desc="Saving"):
        data = bboxes[0]
        for bbox in bboxes[1]:
            data += " "
            data += "{:d},".format(bbox[0])
            data += "{:8.6f},".format(bbox[1])
            data += "{:8.6f},".format(bbox[2])
            data += "{:8.6f},".format(bbox[3])
            data += "{:8.6f}".format(bbox[4])

        data += "\n"
        fd.write(data)
