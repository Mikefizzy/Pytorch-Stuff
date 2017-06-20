import torch
import torch.autograd as grad
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
				grad.Variable(torch.rand(arc[i-1], arc[i]).cuda(), 
					requires_grad = True))
			self.c_weights.append(
				grad.Variable((torch.eye(arc[i], arc[i])*alpha).cuda()
					, requires_grad = True))
			self.h_bias.append(grad.Variable(torch.zeros(1,arc[i]).cuda(), requires_grad = True))
			self.c_bias.append(grad.Variable(torch.zeros(1,arc[i]).cuda(), requires_grad = True))

	def forward_pass(self, x):
		#input should be [batch, memory, arc[0]] and should be already in cuda memory
		batch_size = x.size()[0]
		x = torch.transpose(x, 0,1)# -> [memory, batch, arc[0]]
		a = x #input of each layer

		for i in range(1, len(self.arc)):
			init_state = grad.Variable(torch.zeros(batch_size, self.arc[i]).cuda())
			states = [init_state]
			for t in range(self.memory):
				hx = torch.mm(a[t], self.h_weights[i-1]) #linear transform with inputs
				hx = hx + self.h_bias[i-1].expand_as(hx)

				cx = torch.mm(states[t], self.c_weights[i-1]) #linear transform with states and identity matrix
				cx = cx + self.c_bias[i-1].expand_as(cx)

				state = (cx + hx)
				state = state.clamp(0) #relu
				states.append(state)
			a = states
		return a[-1];

	def params(self):
		return self.h_weights + self.c_weights + self.h_bias + self.c_bias

