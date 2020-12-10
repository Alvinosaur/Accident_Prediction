# Importing all necessary libraries
import cv2
import os

from parse_data import parse_file


def save_frames_from_video(path):
    # Read the video from specified path
    cam = cv2.VideoCapture(path)

    try:

        # creating a folder named data
        if not os.path.exists('data'):
            os.makedirs('data')

    # if not created then raise error
    except OSError:
        print('Error: Creating directory of data')

    # frame
    currentframe = 1

    while(True):

        # reading from frame
        ret, frame = cam.read()

        if ret:
            # if video is still left continue creating images
            name = './data/frame' + str(currentframe) + '.jpg'
            print('Creating...' + name)

            # writing the extracted images
            cv2.imwrite(name, frame)

            # increasing counter so that it will
            # show how many frames are created
            currentframe += 1
        else:
            break

    # Release all space and windows once done
    cam.release()
    cv2.destroyAllWindows()


def show_detections(im_path, detections):
    """[summary]

    Args:
        im_path ([type]): [description]
        start_point ([type]): represents the top left corner of rectangle
        end_point ([type]): represents the bottom right corner of rectangle
    """
    window_name = 'Image'

    # Blue color in BGR
    color = (255, 0, 0)

    # Line thickness of 2 px
    thickness = 2

    # Using cv2.rectangle() method
    # Draw a rectangle with blue line borders of thickness of 2 px
    image = cv2.imread(im_path)

    for det in detections:
        obj_name, obj_id, left, top, right, bottom = det
        image = cv2.rectangle(image, (left, top),
                              (right, bottom), color, thickness)

    # Displaying the image
    cv2.imshow(window_name, image)
    cv2.waitKey(0)


# save_frames_from_video(
#     "/Users/Alvin/Documents/School/class_assignments/18794/project/Accident_Prediction/Anticipating-Accidents/dataset/videos/training/positive/000001.mp4")
frame_infos = parse_file(
    "/Users/Alvin/Documents/School/class_assignments/18794/project/Accident_Prediction/Anticipating-Accidents/dataset/annotation/000001.txt")
for frame_id in range(1, 100):
    detections = frame_infos[frame_id - 1]
    show_detections(im_path="data/frame%d.jpg" %
                    frame_id, detections=detections)
