import torch
import torch.autograd as grad
import numpy as np
from Networks import IRNN
import torch.optim as optim
import matplotlib.pyplot as plt
periods = 2
memory = 200
arc = [1,32,32]
nstates = 128
net = IRNN(arc,memory, 0.3)
def one_hot(x, n):
	size = np.size(x)
	arr = np.zeros([size, n])
	for i in range(size):
		arr[i][x[i]] = 1
	return arr

w = grad.Variable((torch.rand(arc[-1], nstates) * 2 -1), requires_grad = True)
b = grad.Variable(torch.zeros(nstates), requires_grad = True)
def forward(x):
	out = net.forward_pass(x)
	out = torch.mm(out, w)
	out = out + b.expand_as(out)
	out = torch.exp(out)
	out = out/torch.sum(out,1).expand_as(out)
	return out

periods = 10
nsamples = 10000;
a = np.sin(np.linspace(0,periods*np.pi*2, nsamples))
a = ((a-np.min(a))/(np.max(a)-np.min(a)) * (nstates-1)).astype(int)

x_data = []
y_data = one_hot(a[:-memory], nstates)
for i in range(nsamples-memory):
	x_data.append(a[i: i+memory])

x_data = np.array(x_data)/(nstates-1)
x_data = np.reshape(x_data, [nsamples-memory, memory, 1])
x_data = grad.Variable(torch.from_numpy(x_data).float())
y_data = grad.Variable(torch.from_numpy(y_data).float())
epochs = 10
minibatches = 10
minibatch_size = int((nsamples-memory)/minibatches)
optimizer = optim.Adam([w,b] + net.params(), 0.01)
for i in range(epochs):
	for m in range(minibatches):
		optimizer.zero_grad()
		x_batch = x_data[m*minibatch_size: (m+1)*minibatch_size]
		y_batch = y_data[m*minibatch_size: (m+1)*minibatch_size]
		out = forward(x_batch)
		cost = torch.mean(-y_batch*torch.log(out) - (1-y_batch)*torch.log(1-out))
		cost.backward()
		optimizer.step()
		print(cost)





