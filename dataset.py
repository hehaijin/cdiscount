import os
import os.path as path
import numpy as np
import pandas as pd
import bson
import logging
import random
from itertools import takewhile
import io
from skimage.data import imread
from keras.preprocessing.image import ImageDataGenerator



sampledir='sampledir'
dataroot='data'
trainfile='train.bson'

def categorydict():
	df=pd.read_csv(path.join(dataroot, 'category_names.csv'))
	d=dict()
	
	l=df.shape[0]
	#print(l)
	for i in range(l):
		r=df.iloc[i]['category_id']
		r=str(r)
		#print(r)
		d[r]=i
	return d

def getTotalTrainImageCount():
	count=0
	for sample in bson.decode_file_iter(open(path.join(dataroot,trainfile), 'rb')):
		imgs=sample['imgs']
		count=count+len(imgs)
	return count

def getTotalTrainProductCount():
	count=0
	for sample in bson.decode_file_iter(open(path.join(dataroot,trainfile), 'rb')):
		count=count+1
	return count



#a generator that generates indefinately
def BatchGenerator(batch_size):
	categories=categorydict()
	count=0
	batchX=[]
	batchY=[]
	for sample in bson.decode_file_iter(open(path.join(dataroot,trainfile), 'rb')):
		imgs=sample['imgs']
		c=sample['category_id']
		c=str(c)
		#print(type(c))
		cid=categories[c]
		for i in range(len(imgs)):
			im=imgs[i]['picture']
			im=imread(io.BytesIO(im))
			batchX.append(im)
			batchY.append(cid)
			count=count+1
			#print(count)
			if count< batch_size:
				pass
			else:
				yield np.asarray(batchX),np.asarray(batchY)
				count=0
				batchX=[]
				batchY=[]
			


        
bg=BatchGenerator(3)
X,Y=next(bg)
#datagen.fit(X)

	
	

