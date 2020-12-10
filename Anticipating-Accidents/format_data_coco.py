# https://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch
# Output:
# {
#     "info": {...},
#     "licenses": [...],
#     "images": [...],
#     "annotations": [...],
#     "categories": [...], <-- Not in Captions annotations
#     "segment_info": [...] <-- Only in Panoptic annotations
# }

import os
import json
import cv2
from tqdm import tqdm


def save_frames_from_video(dst_folder, vid_path, base_image_id,
                           save=True, default_vid_len=100):
    # Read the video from specified path
    cam = cv2.VideoCapture(vid_path)

    # frame
    image_id = base_image_id
    shape = None

    while(True):

        # reading from frame
        ret, frame = cam.read()
        if shape is None:
            height, width, channels = frame.shape
            shape = {"height": height, "width": width}

        if not save:
            image_id += default_vid_len
            break

        if ret:
            # if video is still left continue creating images
            name = os.path.join(dst_folder, str(image_id) + '.jpg')

            # writing the extracted images
            cv2.imwrite(name, frame)

            # increasing counter so that it will
            # show how many frames are created
            image_id += 1
        else:
            break

    # Release all space and windows once done
    cam.release()
    cv2.destroyAllWindows()

    return image_id, shape


def parse_annotation_file(fname):
    with open(fname, "r") as f:
        text = f.read()
        lines = text.split("\n")[:-1]
        # 000002	6	car	579	330	661	401	0
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
            width = right - left
            height = bottom - top

            frame_infos[frame_num].append(
                (obj_name, obj_id, left, top, width, height))

    return frame_infos


def parse_videos(output, vid_folder, max_num, save_images=True):
    image_id = 1
    vid_names = sorted(os.listdir(vid_folder))[:max_num]
    for vi, vname in tqdm(enumerate(vid_names)):

        # print("EARLY STOPPING!!")
        # if vi == 2:
        #     break

        extension = os.path.splitext(vname)[1]
        if extension != ".mp4":
            continue

        # save images from video
        vid_path = os.path.join(vid_folder, vname)
        print("Parsing: %s..." % vid_path)
        print("Saving images to: %s..." % dst_folder)

        new_image_id, im_shape = save_frames_from_video(
            dst_folder, vid_path, image_id, save=save_images)

        # create json output
        # image_id += 100
        # im_shape = {"height": 720, "width": 1280}
        for im_id in range(image_id, new_image_id):
            output["images"].append(
                {"file_name": os.path.join(dst_folder, str(im_id) + ".jpg"),
                 "id": im_id,
                 "height": im_shape["height"],
                 "width": im_shape["width"]}
            )

        image_id = new_image_id


def parse_annotations(output, annotation_folder, cur_annotation_fnames,
                      category_to_id, max_num):
    image_id = 1
    annotation_id = 1
    cur_annotation_fnames = cur_annotation_fnames[:max_num]
    for fname_i, fname in tqdm(enumerate(cur_annotation_fnames)):
        path = os.path.join(annotation_folder, fname)

        # if fname_i == 2:
        #     break

        # parse annotation labels
        per_frame_labels = parse_annotation_file(path)
        for fi, frame_labels in enumerate(per_frame_labels):
            for label in frame_labels:
                (obj_name, _, left, top, width, height) = label
                if obj_name not in category_to_id:
                    obj_id = len(output["categories"]) + 1
                    output["categories"].append(
                        {"supercategory": obj_name,
                            "id": obj_id, "name": obj_name}
                    )
                    # assign id of object to number of objects so far
                    category_to_id[obj_name] = obj_id

                output["annotations"].append(
                    {
                        "image_id": image_id + fi,
                        "bbox": [left, top, width, height],
                        "category_id": category_to_id[obj_name],
                        "id": annotation_id,  # unique to every annotation
                        "area": width * height,
                        "segmentation": [],
                        "iscrowd": False,
                    },
                )
                annotation_id += 1

        assert(len(per_frame_labels) == 100)
        image_id += len(per_frame_labels)


if __name__ == "__main__":
    output_names_path = "custom_annotations.names"
    annotation_folder = "/Users/Alvin/Documents/School/class_assignments/18794/project/Accident_Prediction/Anticipating-Accidents/dataset/annotation"
    annotation_fnames = sorted(os.listdir(annotation_folder))

    base_vid_folder = "/Users/Alvin/Documents/School/class_assignments/18794/project/Accident_Prediction/Anticipating-Accidents/dataset/videos/"
    # ORDER MATTERS! DON't SWITCH TRAINING AND TESTING
    vid_folders = [base_vid_folder + "training/positive",
                   base_vid_folder + "testing/positive"]
    vid_bounds = [(0, 455), (455, len(annotation_fnames))]
    category_to_id = dict()
    train_output_categories = None

    # for dataset in [train, test]
    for di, vid_folder in enumerate(vid_folders):
        set_type = "train" if di == 0 else "test"
        output_json_path = f"custom_annotations_{set_type}.json"
        output = {
            "info": {
                "description": f"Accident Prediction Custom Dataset {set_type}"
            },
            "licenses": {},
            "images": [],
            "categories": [],
            "annotations": []
        }
        if train_output_categories is not None:
            output["categories"] = train_output_categories

            # make data folder
        dst_folder = f"data/images/{set_type}"
        try:
            # creating a folder named data
            if not os.path.exists(dst_folder):
                os.makedirs(dst_folder)

        # if not created then raise error
        except OSError:
            print('Error: Creating directory of data')

        save_images = True
        if di == 0:
            max_num = 100
        else:
            max_num = 50
        parse_videos(output, vid_folder,
                     save_images=save_images, max_num=max_num)

        start, end = vid_bounds[di]
        cur_annotation_fnames = annotation_fnames[start:end]
        parse_annotations(output, annotation_folder,
                          cur_annotation_fnames, category_to_id, max_num=max_num)

        train_output_categories = output["categories"]

        with open(output_json_path, 'w') as json_file:
            json.dump(output, json_file)

    with open(output_names_path, "w") as f:
        class_names = "\n".join(category_to_id.keys())
        f.write(class_names)
