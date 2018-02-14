import matplotlib
matplotlib.use("Agg")
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from le_net_model import LeNetModel
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import cv2
import os

class TrainedModel:

    # initialize the number of epochs to train for, initial learning rate,
    # and batch size
    def __init__(self, epochs, init_lr, batch_size):
        self.epochs = epochs
        self.init_lr = init_lr
        self.batch_size = batch_size
        self.data = []
        self.labels = []
        # data augmentation enables generating additional
        # training data by randomly transforming the input images
        self.img_augmentation = ImageDataGenerator(rotation_range=30, width_shift_range=0.1, 
            height_shift_range=0.1, shear_range=0.2, zoom_range=0.2, 
            horizontal_flip=True, fill_mode='nearest')
    
    def image_preprocessing(self, i_path):
        img = cv2.imread(i_path)
        img = cv2.resize(img, (28, 28))
        img = img_to_array(img)
        return img

    def data_label_initialization(self, args):
        image_paths = sorted(list(paths.list_images(args["dataset"])))
        # random shuffling the images
        random.seed(42)
        random.shuffle(image_paths)
        for i_path in image_paths:
            self.data.append(self.image_preprocessing(i_path))
            label = i_path.split(os.path.sep)[-2]
            if label == "dogs":
                label = 1
            elif label == "cats":
                label = 0
            elif label == "birds":
                label = 2
            self.labels.append(label)
    
    def split_train_test_data(self,args):
        # scaling the pixel intesities to [0, 1] range
        self.data = np.array(self.data, dtype = "float") / 255.0
        self.labels = np.array(self.labels)
        (train_X, test_X, train_Y, test_Y) = train_test_split(self.data, self.labels, test_size = 0.25, random_state=42)
        # integer to vector conversion
        train_Y = to_categorical(train_Y, num_classes=int(args["classes"]))
        test_Y = to_categorical(test_Y, num_classes= int(args["classes"]))
        return train_X, test_X, train_Y, test_Y
    
    def train_model(self, args, train_X, test_X, train_Y, test_Y):
        # First initialize the LeNetModel from the script le_net_model
        model = LeNetModel.build(width = 28, height = 28, depth = 3, classes = int(args["classes"]))
        opt = Adam(lr = self.init_lr, decay= self.init_lr / self.epochs)
        model.compile(loss = "binary_crossentropy", optimizer=opt, metrics=["accuracy"])

        #Training the built model
        H = model.fit_generator(self.img_augmentation.flow(train_X, train_Y, batch_size=self.batch_size),
            validation_data=(test_X, test_Y), steps_per_epoch=len(train_X),
            epochs=self.epochs, verbose=1)
        model.save(args["model"])

def parsing_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True,
        help="path to input dataset")
    ap.add_argument("-m", "--model", required=True,
        help="path to output model")
    ap.add_argument("-c", "--classes", required=True,
        help="number of classes to recognize")
    return ap

def main():
    ap = parsing_arguments()
    args = vars(ap.parse_args())
    trained_model = TrainedModel(1, 1e-3, 32)
    trained_model.data_label_initialization(args)
    tr_X, te_X, tr_Y, te_Y = trained_model.split_train_test_data(args)
    trained_model.train_model(args, tr_X, te_X, tr_Y, te_Y)

main()