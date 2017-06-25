#this script clearly shows the exponential growth in using relu
import numpy as np
import matplotlib.pyplot as plt
layers = 4
hidden_size  =100
input_size = 2
inputs = np.random.rand(1,input_size)
maximums = []
for i in range(layers):
	w = np.random.rand(np.shape(inputs)[1], hidden_size)*2 -1 # weights are from a normal distribution [-1, 1]
	l = np.matmul(inputs, w)
	l = np.maximum(l, 0, l) #relu
	inputs = l;
	maximums.append(np.max(l))
plt.plot(maximums)
plt.show()