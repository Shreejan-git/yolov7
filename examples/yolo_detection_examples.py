import logging
import os.path
from typing import Union
import cv2
import numpy as np
from detect import YoloV7


def post_process(result, show_drawn_bbox=False) -> Union[np.ndarray, str, bool]:
    detected_objects_data = result[0]
    total_object_detected = result[1]
    original_image = result[2]

    if total_object_detected:
        for detected_object_data in detected_objects_data:
            left, top, right, bottom = detected_object_data[0]
            cv2.rectangle(original_image, (left, top), (right, bottom), (255, 52, 23), 3)

            if show_drawn_bbox:
                cv2.namedWindow("img", cv2.WINDOW_NORMAL)
                cv2.imshow('img', original_image)
                cv2.waitKey(0)
        return True
    else:
        return False


class ObjectDetection:
    def __init__(self):
        self.yolo_instance = YoloV7(conf_thres=0.6)

    def detect_objects(self, img_source, show_drawn_bbox=False):

        acceptable_image_format = ("jpg", "png", "jpeg")
        if os.path.isdir(img_source):  # directory of images
            all_images = os.listdir(img_source)
            for image in all_images:
                if image.split(".")[-1].lower() in acceptable_image_format:
                    img_path: str = os.path.join(img_source, image)
                    result = self.yolo_instance.detect(source=img_path)

                    result = post_process(result=result, show_drawn_bbox=show_drawn_bbox)
                    return result

                else:
                    return "Could not proceed the given image format."
        elif os.path.isfile(img_source):
            if img_source.split(".")[-1] in acceptable_image_format:
                try:
                    result = self.yolo_instance.detect(source=img_source)
                    result = post_process(result=result, show_drawn_bbox=show_drawn_bbox)
                    if result:
                        return {"leaf_found_flag": "Found",
                                "no_of_leaf_detected": 1}
                    else:
                        return {"leaf_found_flag": "not_found",
                                "no_of_leaf_detected": 1}

                except Exception as e:
                    logging.info("Error in predicting in yolo", e)
                    return None

        else:  # for video through web or mp4
            # define a video capture object
            vid = cv2.VideoCapture(1)

            while True:

                # Capture the video frame
                # by frame
                ret, frame = vid.read()

                if ret:
                    # result = self.yolo_instance.detect(source=frame)
                    # self.post_process(result=result, show_drawn_bbox=show_drawn_bbox)

                    # Display the resulting frame
                    cv2.imshow('frame', frame)

                # the 'q' button is set as the
                # quitting button you may use any
                # desired button of your choice
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # After the loop release the cap object
            vid.release()
            # Destroy all the windows
            cv2.destroyAllWindows()


if __name__ == "__main__":
    img_dir = '/home/vertexaiml/Downloads/plant_detection_project/final_cropped_images/Potato early blight'
    img_path = '/home/vertexaiml/Downloads/MicrosoftTeams-image (2).png'

    a = ObjectDetection()
    a.detect_objects(img_source=img_path, show_drawn_bbox=True)
    # a.detect_objects(img_source=img_dir, show_drawn_bbox=True)
