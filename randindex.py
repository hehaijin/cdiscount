
def calculateRandIndex(original,cluster):
    N1=len(original)
    N2=len(cluster)
    if N1 != N2:
        raise ValueEroor("input length not the same")
    count=0
    for i in range(N2):
        for j in range(i+1,N2):
            if cluster[i]==cluster[j] and original[i]==original[j]:
                count=count+1
            elif cluster[i]!=cluster[j] and original[i]!=original[j]:
                count=count+1
    totalcount=N1*(N1-1)/2
    return count/totalcount
                

original=[]
cluster=[]
for i in range(len(in_data)):
    row=in_data[i]
    cluster.append(row.y[0])
    original.append(int(row.metas[0].split('-')[0]))
rdindex=calculateRandIndex(original,cluster)
print(rdindex)
out_data=rdindex
