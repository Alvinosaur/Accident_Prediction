
"""
Feature Extractor: 
Input:
Output: 4096 x 1 feature
every frame

Train last 3 layers

Datasets to train on:
KITTI


Actual data shape: 
slice in to replace the VGG feature with our own features

don't actually need the IDT feature, only need VGG feature

We can do other modifications: try concatenating our own extracted features with
PCA

Train actual accident predictor for 40 epochs


Things to figure out:


Metrics to figure out:


Dynamic-Spatial-Attention RNN. This is our proposed method. Our method has

three variants (see Sec. 3.2): (1) no full-frame features, only attention on object can-
didates (D); (2) weighted-summing full-frame feature with object-specific features

(F+D-sum); (3) concatenating full-frame features with object features
(F+D-con.).


Run YOLO, Fast-RCNN, Faster-RCNN on dataset and compare their speed and
accuracy. Independent of accident prediction. Show some training curves, show
some pictures.

Feed these into the  accident prediction as features and compare accident
prediction performance

Tasks
- correct labels to data and images
- how to pass in variable sized objects and extract features for them?
- where do we extract the backbone features?


use detectron2 to get bounding boxes
and  pass entire image or cropped obejct into whatever feature extractor:
ResNet50, VGG, MobileNet 

transforms.resize up to 224 for H and W
https://pytorch.org/hub/pytorch_vision_resnet/
https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5


"""


"""
Resources: 
Visualize  YOLO architecture:
https://netron.app/?url=https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolo.cfg

YOLO Colab Notebook guiding how to train pretrained YOLO:
https://colab.research.google.com/drive/1_GdoqCJWXsChrOiY8sZMr_zbr_fH-0Fg?usp=sharing#scrollTo=qaONUTI2l4Sf



"""
import re


def parse_file(fname):
    with open(fname, "r") as f:
        text = f.read()
        lines = text.split("\n")[:-1]
        # 000001	1	car	1020	375	1278	558	0
        frame_infos = [[] for i in range(100)]

        for l in lines:
            frame_num, obj_id, obj_name, left, top, right, bottom, _ = l.split(
                "\t")
            frame_num = int(frame_num) - 1
            obj_id = int(obj_id)
            left = int(left)
            top = int(top)
            right = int(right)
            bottom = int(bottom)

            frame_infos[frame_num].append(
                (obj_name, obj_id, left, top, right, bottom))

    return frame_infos
