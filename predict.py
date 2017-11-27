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
		newd[v]=key
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
	if count==10000:
		break
	r=my_model.predict(pics)
	r=np.argmax(r,axis=1)
	r=np.argmax(np.bincount(r))
	r=newd[r]
	result.append([productid,r])
	
	
result=np.asarray(result)
submission=pd.DataFrame({'_id':result[:,0],'category_id':result[:,1]})
submission.to_csv("submission.csv",index=False)








