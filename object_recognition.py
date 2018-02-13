import numpy as np
import argparse
import cv2

def parsing_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True,
        help="path to input image")
    ap.add_argument("-p", "--prototxt", required=True,
        help="path to Caffe 'deploy' prototxt file")
    ap.add_argument("-m", "--model", required=True,
        help="path to Caffe pre-trained model")
    return ap

class Object_Detector:

    # This network was trained on the COCO dataset and fine-tuned on PASCAL VOC
    # It was trained on 20 different weights which are here labeled
    labels = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]

    # generate random colors for the box labels on the image
    label_colors = np.random.uniform(0, 255, size = (len(labels), 3))

    def __init__(self):
        self.network = None
        self.img = None
        self.blob = None
        self.h = 0
        self.w = 0
        self.objects = None
    
    def load_model(self, args):
        # load the pretrained MobileNet SSD  model from the disk
        self.network = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

    def load_image(self, args):
        # load the image from the disk and convert it to a blob
        # resize it to a 300x300 format and set the scalefactor to a predetermined value
        # last parameter represents the mean value
        self.img = cv2.imread(args["image"])
        self.blob = cv2.dnn.blobFromImage(cv2.resize(self.img, (300, 300)), 0.007843, (300, 300), 127.5)
        # save the height and width information about iamge
        (self.h, self.w) = self.img.shape[:2]

    def forward_propagation_blob(self):
        # in this step, the blob is passed through the network 
        # using forward propagation
        # the results are detected objects on the image
        self.network.setInput(self.blob)
        self.objects = self.network.forward()

    def object_labeling(self):
        for i in np.arange(0, self.objects.shape[2]):
            # for correct object recognition, it is required
            # to detect the corresponding class labels
            class_index = int(self.objects[0, 0, i, 1])
            border_box = self.objects[0, 0, i, 3:7] * np.array([self.w, self.h, self.w, self.h])
            (coord1, coord2, coord3, coord4) = border_box.astype("int")
            rectangle_top = (coord1, coord2)
            rectangle_down = (coord3, coord4)
            # displaying the prediction
            cv2.rectangle(self.img,  rectangle_top, rectangle_down, self.label_colors[class_index], 2)
            # calculate where to display object information
            y = 0
            if rectangle_top[1] - 15 > 15:
                y = rectangle_top[1] - 15
            else:
                y = rectangle_top[1] + 15
            label_information = "{}: {:.2f}%".format(self.labels[class_index], self.objects[0, 0, i, 2] * 100)
            cv2.putText(self.img, label_information, (rectangle_top[0], y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.label_colors[class_index], 2)

    def image_display(self, title=""):
        #display the image
        cv2.imshow(title, self.img)
        cv2.waitKey(0)

def main():
    ap = parsing_arguments()
    args = vars(ap.parse_args())
    detector_instance = Object_Detector()
    detector_instance.load_model(args)
    detector_instance.load_image(args)
    detector_instance.forward_propagation_blob()
    detector_instance.object_labeling()
    detector_instance.image_display("Output")

main()