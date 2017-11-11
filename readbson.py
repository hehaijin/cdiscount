import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output
import io
import bson                       # this is installed with the pymongo package
import matplotlib.pyplot as plt
from skimage.data import imread   # or, whatever image library you prefer
import multiprocessing as mp      # will come in handy due to the size of the data

data = bson.decode_file_iter(open('C:\\Users\\Haijin\\datascience\\cdiscount\\input\\train_example.bson', 'rb'))

datalist = []

for c, d in enumerate(data):
    product_id = d['_id']
    category_id = d['category_id'] # This won't be in Test data
    count=0
    pictures=[]
    for e, pic in enumerate(d['imgs']):
        count=count+1
        picture = imread(io.BytesIO(pic['picture']))
        plt.imshow(picture)
        plt.show()
        pictures.append(picture)
    datalist.append((product_id,category_id,count,pictures))
        # do something with the picture, etc
        
df = pd.DataFrame(datalist, columns=['product_id','category_id','pic_count','pics'])
 
df.head(1)
 
df.agg('sum')
df.agg('mean') 