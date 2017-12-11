import dataset
import os.path as path
import cv2

def main():
	savefolder='clusterfolder'
	categories=['1000016400','1000016402','1000016404','1000021520','1000016406','1000016398','1000016408','1000021522','1000016416','1000016410','1000016412','1000016414']

	bg=dataset.ClusterImageGenerator(5000)      

	filecount=0
	for pic,c in bg:
		N=len(c):
		for j in N:	
			filecount=filecount+1
			if filecount
			try:
				i=categories.index(c[j])
				cv2.imwrite(path.join(savefolder, str(i)+"-"+count(filecount)+'.jpg') ,pic[j])
			except ValueError:
				pass	
	
