from detect import YoloV7

if __name__ == "__main__":
    result = YoloV7(conf_thres=0.6)
    # result.detect('/home/vertexaiml/Downloads/plant_detection_project/PlantDoc-Dataset/test/Apple leaf/20180511_090912-14gtw8a-e1526047952754.jpg', view_img=False, webcam=False)
    result.detect('/home/vertexaiml/Downloads/plant_detection_project/PlantDoc-Dataset/test/Apple leaf', view_img=False, webcam=False)
