import matplotlib.pyplot as plt



accuracies=[0.45447,0.48142,0.51501,0.55375,0.58946,0.53646]
epochs=[]
for i in range(6):
	epochs.append(i+1)

plt.plot(epochs,accuracies)
plt.show()







