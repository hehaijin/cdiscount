from keras.preprocessing.image import ImageDataGenerator
import dataset
import cdiscountkeras
from keras import optimizers
from keras.utils import np_utils



nb_epoch=1


bg=dataset.BatchGenerator(20)
X,Y=bg.__next__()

datagen.fit(X)
my_model=cdiscountkeras.loadModel()

sgd = optimizers.SGD(lr=0.0001)
my_model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

bg2=dataset.BatchGenerator(5000)              
for e in range(nb_epoch):
	print("big epoch %d" % e)
	for X_train, Y_train in bg2:
		Y_train= np_utils.to_categorical(Y_train,5271)
		my_model.fit(X_train, Y_train,batch_size=32, epochs=1,verbose=1)
     
 
 
cdiscountkeras.saveModel(my_model)            
              
