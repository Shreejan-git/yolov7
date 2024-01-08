from detect import YoloV7

if __name__ == "__main__":
    result = YoloV7(conf_thres=0.6)
    result.detect('/home/vertexaiml/Desktop/potato_bactrial_spot.jpg')
