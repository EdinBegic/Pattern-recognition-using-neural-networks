from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2

def parsing_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True,
        help="path to trained model model")
    ap.add_argument("-i", "--image", required=True,
        help="path to input image")
    return ap

def image_preprocessing(args):
    img = cv2.imread(args["image"])
    orig = img.copy()
    img = cv2.resize(img, (28, 28))
    img = img.astype("float") / 255.0
    img = img_to_array(img)
    img = np.expand_dims(img, axis = 0)
    return orig, img

def create_prediction(args, orig, img):
    # first load the model
    model = load_model(args["model"])
    # image classification
    (cats, dogs) = model.predict(img)[0]
    # setting the labels
    conffidence = 0
    label = ""
    birds = 0
    if dogs > cats and dogs > birds:
        label = "Dog"
        conffidence = dogs
    elif cats > dogs and cats > birds:
        label = "Cat"
        conffidence = cats
   # elif birds > dogs and birds > cats:
   #     label = "Bird"
   #     conffidence = birds
    label = "{}: {:.2f}%".format(label, conffidence * 100)
    output = imutils.resize(orig, width = 400)
    cv2.putText(output, label, (10, 25), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 
        0.7, (0, 255, 0), 2)
    cv2.imshow("Output", output)
    cv2.waitKey(0)

def main():
    ap = parsing_arguments()
    args = vars(ap.parse_args())
    orig, img = image_preprocessing(args)
    create_prediction(args, orig, img)

main()