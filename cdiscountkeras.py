import numpy as np
from keras import optimizers
import keras
from keras.models import model_from_json
import pandas as pd
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.convolutional import ZeroPadding2D
from keras.utils import np_utils
import os.path as path
from keras.applications.vgg16 import VGG16
import logging
from keras.preprocessing.image import ImageDataGenerator
modelfile='model.json'
weightsfiles='weights.h5'



def loadModel():
	#define model. it's simple VGG-16 with modified full connectted layer.
	my_model=None
	if path.exists(modelfile):
		with open(modelfile) as json_file:
			loaded_model_json = json_file.read()
			my_model = model_from_json(loaded_model_json)
			my_model.load_weights(weightsfiles)
	else:
		model_vgg16_conv = VGG16(weights='imagenet', include_top=False)
		#model_vgg16_conv.summary()
		input = Input(shape=(90,90,3),name = 'input')
		output_vgg16_conv = model_vgg16_conv(input)
		x = Flatten(name='flatten')(output_vgg16_conv)
		x = Dense(4096, activation='relu', name='fc1')(x)
		x = Dense(4096, activation='relu', name='fc2')(x)
		x = Dense(5271, activation='softmax', name='predictions')(x)
		my_model = Model(input=input, output=x)
		my_model.summary()

	return my_model


def saveModel(my_model):
	model_json = my_model.to_json()
	with open(modelfile, "w") as json_file:
		json_file.write(model_json)
	my_model.save_weights(weightsfiles)
	#print("Saved model to disk")
	
	

