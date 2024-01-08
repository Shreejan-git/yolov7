# import the opencv library
import cv2
from detect import YoloV7

if __name__ == "__main__":
    result = YoloV7(conf_thres=0.6)
    # define a video capture object
    vid = cv2.VideoCapture(0)

    while True:

        # Capture the video frame
        # by frame
        ret, frame = vid.read()
        if ret:
            result.detect(source=frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()
