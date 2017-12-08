from cdiscountkeras import loadModel
import numpy as np
from dataset import categorydict,TestBatchGenerator
import pandas as pd
		
def getKey(dictionary,value):
	for k,v in dictionary.items():
		if v==value:
			return k
	return -1

def reverseDict(d):
	newd={}
	for k,v in d.items():
		newd[v]=k
	return newd
	


my_model=loadModel()


#need to compile model?
result=[]

cat=categorydict()
newd=reverseDict(cat)


count=0
for pics, productid in TestBatchGenerator():
	count=count+1
	if count%1000==0:
		print(count)
	r=my_model.predict(pics)
	r1=np.argmax(r,axis=1)
	b=np.bincount(r1)
	r2=np.argmax(b)
	r3=newd[r2]
	if len(pics)==1 or  b[r2]>=2:
		#majority voting
		result.append([productid,r3])
	else: 
		#get the biggest number of prediction
		r4=[]
		for i in r1:
			r4.append(r[i])
		k=argmax(r4)
		print("number of pics "+ str(len(pics)))
		print(k)
		print(r1[k])
		result.append([productid,r1[k]])
			
result=np.asarray(result)
submission=pd.DataFrame({'_id':result[:,0],'category_id':result[:,1]})
submission.to_csv("submission.csv",index=False)








