import os
import os.path as path
import numpy as np
import pandas as pd
import bson
import logging
import random
from itertools import takewhile
import io
import cv2
from skimage.data import imread
from keras.preprocessing.image import ImageDataGenerator



sampledir='sampledir'
dataroot='data'
trainfile='train.bson'
testfile='test.bson'

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



#generator that loop through train file to output data in batches.
#here the batch_size is conting pictures
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
			im=cv2.resize(im,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_AREA)
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
			

#here the batchY is category_id
#here the batch is counting sample
def TestBatchGenerator(batch_size=1):
	categories=categorydict()
	count=0
	batchX=[]

	for sample in bson.decode_file_iter(open(path.join(dataroot,testfile), 'rb')):
		imgs=sample['imgs']
		c=sample['_id']
		for i in range(len(imgs)):
			im=imgs[i]['picture']
			im=imread(io.BytesIO(im))
			im=cv2.resize(im,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_AREA)
			batchX.append(im)
	
		batchY=c
		count=count+1
		#print(count)
		if count< batch_size:
			pass
		else:
			yield np.asarray(batchX),np.asarray(batchY)
			count=0
			batchX=[]
			
	



	
	

