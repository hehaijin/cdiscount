import csv

level1={}
i=0
with open('category_names.csv',newline='') as csvfile:
		trainreader = csv.reader(csvfile)
		for row in trainreader:
			if level1.get(row[2])==None:
				i=i+1
				level1[row[2]]=i
				print(row[1])
				print(i)
			
			
