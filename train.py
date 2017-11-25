from keras.preprocessing.image import ImageDataGenerator
import dataset
import cdiscountkeras
from keras import optimizers
from keras.utils import np_utils


nb_epoch=1

datagen = ImageDataGenerator(
		featurewise_center=True, # set input mean to 0 over the dataset
		samplewise_center=False, # set each sample mean to 0
		featurewise_std_normalization=True, # divide inputs by std of the dataset
		samplewise_std_normalization=False, # divide each input by its std
		zca_whitening=False, # apply ZCA whitening
		rotation_range=20, # randomly rotate images in the range (degrees, 0 to 180)
		width_shift_range=0.2, # randomly shift images horizontally (fraction of total width)
		height_shift_range=0.2, # randomly shift images vertically (fraction of total height)
		horizontal_flip=True, # randomly flip images
		vertical_flip=False) # randomly flip images


bg=dataset.BatchGenerator(20)
X,Y=bg.__next__()

datagen.fit(X)
my_model=cdiscountkeras.loadModel()

sgd = optimizers.SGD(lr=0.0001)
my_model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

bg2=dataset.BatchGenerator(50)              
for e in range(nb_epoch):
	print("big epoch %d" % e)
	for X_train, Y_train in bg2: 
		Y_train= np_utils.to_categorical(Y_train,5271)
		my_model.fit_generator(datagen.flow(X_train, Y_train),steps_per_epoch=10, epochs=1,verbose=1)
     
 
 
cdiscountkeras.saveModel()            
              
              



