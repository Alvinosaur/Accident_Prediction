import cv2
import numpy as np
import tensorflow as tf
from yolov4.tf import YOLOv4

yolo = YOLOv4()
yolo.classes = "custom_annotations.names"
yolo.input_size = (640, 480)
yolo.batch_size = 2
# 000000000009.jpg 45,0.479492,0.688771,0.955609,0.595500 45,0.736516,0.247188,0.498875,0.476417 50,0.637063,0.732938,0.494125,0.510583
dataset = yolo.load_dataset(
    "custom_annotations_train.txt", image_path_prefix="")

for i, (images, gt) in enumerate(dataset):
    for j in range(len(images)):
        _candidates = []
        for candidate in gt:
            grid_size = candidate.shape[1:3]
            _candidates.append(
                tf.reshape(
                    candidate[j], shape=(
                        1, grid_size[0] * grid_size[1] * 3, -1)
                )
            )
        candidates = np.concatenate(_candidates, axis=1)

        frame = images[j, ...] * 255
        frame = frame.astype(np.uint8)

        pred_bboxes = yolo.candidates_to_pred_bboxes(candidates[0])
        pred_bboxes = yolo.fit_pred_bboxes_to_original(
            pred_bboxes, frame.shape)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        image = yolo.draw_bboxes(frame, pred_bboxes)
        cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("result", image)
        cv2.waitKey(0)
        # while cv2.waitKey(10) & 0xFF != ord("q"):
        #     pass
    if i == 10:
        break

cv2.destroyWindow("result")
