import torch
import torch.autograd as grad
import numpy as np
import matplotlib.pyplot as plt
class IRNN:
	def __init__(self, arc, memory, alpha = 1):
		#arc is an array which descibes the width of each recurrent layer 
		#ex: [10,300,300,300] 300,300,300 are all recurrent layers

		#alpha is a constant which scales the identity matrix at initialization 

		self.h_weights = [] 
		self.c_weights = []
		self.h_bias = []
		self.c_bias = []
		self.memory = memory
		self.arc = arc;
		for i in range(1, len(arc)): #hidden layers are arc[1]->arc[-1]
			self.h_weights.append(
				grad.Variable((torch.rand(arc[i-1], arc[i])*2 -1), 
					requires_grad = True))
			self.c_weights.append(
				grad.Variable((torch.eye(arc[i], arc[i])*alpha)
					, requires_grad = True))
			self.h_bias.append(grad.Variable(torch.zeros(1,arc[i]), requires_grad = True))
			self.c_bias.append(grad.Variable(torch.zeros(1,arc[i]), requires_grad = True))
	def soft_maximum(self, x):
		return torch.log(torch.sum(torch.exp(x), 1))
	def forward_pass(self, x):
		#input should be [batch, memory, arc[0]]
		batch_size = x.size()[0]
		x = torch.transpose(x, 0,1)# -> [memory, batch, arc[0]]
		a = x #input of each layer
		for i in range(1, len(self.arc)):
			init_state = grad.Variable(torch.zeros(batch_size, self.arc[i]))
			states = [init_state]
			for t in range(self.memory):
				hx = torch.mm(a[t], self.h_weights[i-1]) #linear transform with inputs
				hx = hx + self.h_bias[i-1].expand_as(hx)
				cx = torch.mm(states[t], self.c_weights[i-1]) #linear transform with states and identity matrix
				cx = cx + self.c_bias[i-1].expand_as(cx)

				state = (cx + hx)
				state = state.clamp(min =0) #relu
				state = state/self.soft_maximum(state)[0].expand_as(state)
				
				states.append(state)
			a = states[1:] #not include the initial state
		return a[-1]
	def params(self):
		return self.h_weights + self.c_weights + self.h_bias + self.c_bias

class NumpyIRNN:
	def __init__(self, arc, memory, alpha):
		self.h_weights = [] 
		self.c_weights = []
		self.h_bias = []
		self.c_bias = []
		self.memory = memory
		self.arc = arc;
		for i in range(1, len(arc)): #hidden layers are arc[1]->arc[-1]

			self.h_weights.append(np.random.rand(arc[i-1], arc[i])*2 -1)
				
			self.c_weights.append(np.eye(arc[i], arc[i]) * alpha)

			self.h_bias.append(np.zeros([1,arc[i]]))
			self.c_bias.append(np.zeros([1,arc[i]]))

	def forward_pass(self, x):
		#input should be [batch, memory, arc[0]]
		batch_size = np.shape(x)[0]
		x = np.transpose(x, [1,0,2])# -> [memory, batch, arc[0]]
		a = x #input of each layer
		g = []
		for i in range(1, len(self.arc)):
			init_state = np.zeros([batch_size, self.arc[i]])
			states = [init_state]
			for t in range(self.memory):

				hx = np.matmul(a[t], self.h_weights[i-1]) #linear transform with inputs
				hx = hx + self.h_bias[i-1]
				cx = np.matmul(states[t], self.c_weights[i-1]) #linear transform with states and identity matrix
				cx = cx + self.c_bias[i-1]
				state = (cx + hx)
				state = np.maximum(state,0,state) #relu
				state = state/np.max(state)
				states.append(state)
				g.append(state[-1][-1])
				#print(state)
				#print(torch.exp(state))
			#print('layer: ' + str(i) + '  max final value    ' + str(torch.max(states[-1])))
			a = states[1:] #not include the initial state
		plt.plot(g)
		plt.show()
		return a[-1]