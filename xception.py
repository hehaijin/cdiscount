import numpy as np
from keras import optimizers
import keras
from keras.models import model_from_json
import pandas as pd
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input,GlobalAveragePooling2D
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.convolutional import ZeroPadding2D
from keras.utils import np_utils
import os.path as path
from keras.applications.xception import Xception
import logging
from keras.preprocessing.image import ImageDataGenerator

modelfile='xception.json'
weightsfiles='xceptionweights.h5'



def loadModel():
	my_model=None
	if path.exists(modelfile):
		with open(modelfile) as json_file:
			loaded_model_json = json_file.read()
			my_model = model_from_json(loaded_model_json)
			my_model.load_weights(weightsfiles)
	else: 
		input = Input(shape=(90,90,3),name = 'input')
		print(type(input))
		model0 = Xception(include_top=False, weights='imagenet',
							input_tensor=None, input_shape=(90, 90, 3))
		print(type(model0))
		for lay in model0.layers:
			lay.trainable = False
			
		x = model0.output
		ys=model0.outputs
		print(type(x))
		for y in ys: 
			print(type(y))
		
		x = GlobalAveragePooling2D(name='avg_pool')(x)

		x = Dropout(0.2)(x)
		x = Dense(5271, activation='softmax', name='predictions')(x)
		my_model = Model(model0.input, output=x)
		#my_model.summary()
	return my_model



def saveModel(my_model):
	model_json = my_model.to_json()
	with open(modelfile, "w") as json_file:
		json_file.write(model_json)
	my_model.save_weights(weightsfiles)
	#print("Saved model to disk")
	
m=loadModel()
saveModel(m)
