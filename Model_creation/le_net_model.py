# import the necessary packages
from keras.models import Sequential
from keras.layers.convolutional import Conv2D # Class responsible for the convolution
from keras.layers.convolutional import MaxPooling2D # Max-pooling operations
from keras.layers.core import Activation # Applying a particular activation function
from keras.layers.core import Flatten # Flattening the network topology
from keras.layers.core import Dense
from keras import backend as K

class LeNetModel:
	@staticmethod
	def build(width, height, depth, classes): #building the architecture of the model
		# for correct model initialization three paramethers are required 
		# the width of the images
		# the height of the images
		# the number of channels (number of colors) on the images represented 
		# with the depth parameter
		# the total number of classes which will be recognized 

		# Initialize the model with a linear stack of layers
		model = Sequential()
		# inputShape will use channels last ordering
		inputShape = (height, width, depth)

		# adding layers to the model
		# first set of CONV => RELU => POOL layers
		# the CONV layer will learn 20 convolution filters 
		model.add(Conv2D(20, (5, 5), padding="same",
			input_shape=inputShape))
		# Applying a ReLU activation function
		model.add(Activation("relu"))
		# 2x2 max-pooling
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
		# second set of CONV => RELU => POOL layers
		# common practice to increase number of filters as the network architecture grows
		model.add(Conv2D(50, (5, 5), padding="same"))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

		# flattening out the volume into a set of fully-connected layers
		# first (and only) set of FC => RELU layers
		model.add(Flatten()) # flattening the max-pooling2D layer into a single vector
		model.add(Dense(500)) # layer consists of 500 nodes
		model.add(Activation("relu"))
 
		# softmax classifier
		model.add(Dense(classes))
		model.add(Activation("softmax"))
 
		# return the constructed network architecture
		return model
