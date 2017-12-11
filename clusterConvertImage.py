#import dataset
import os.path as path
#import cv2

def main():
	savefolder='clusterfolder'
	categories=['1000016400','1000016402','1000016404','1000021520','1000016406','1000016398','1000016408','1000021522','1000016416','1000016410','1000016412','1000016414']
	print(len(categories))
	bg=dataset.ClusterImageGenerator(5000)      
	
	totalfilecount=0
	fc=0
	for pic,c in bg:
		N=len(c)
		print("batch length "+str(N))
		for j in range(N):	
			totalfilecount=totalfilecount+1
			if totalfilecount%1000==0:
				print(totalfilecount)
			if c[j] in categories:
				fc=fc+1
				print("found file "+str(fc))
				i=categories.index(c[j])
				cv2.imwrite(path.join(savefolder, str(i)+"-"+str(fc)+'.jpg') ,pic[j])
			
	



if __name__=="__main__":
	main()




