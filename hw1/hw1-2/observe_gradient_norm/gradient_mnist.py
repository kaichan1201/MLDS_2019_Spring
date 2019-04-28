import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BATCH_SIZE = 128
KERNEL_SIZE = 3
EPOCH = 100
gradient_list = []
loss_list = []
NUM_WORKER = 1

data_train = datasets.MNIST(root = 'MNIST_train.npy', transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.5],[0.5])]), train = True, download = True)
data_test = datasets.MNIST(root = 'MNIST_test.npy', transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.5],[0.5])]), train = False, download = True)

data_loader_train = Data.DataLoader(dataset = data_train, batch_size = BATCH_SIZE, shuffle = True, num_workers = NUM_WORKER)
data_loader_test = Data.DataLoader(dataset = data_test, batch_size = BATCH_SIZE, shuffle = True, num_workers = NUM_WORKER)

class Net(torch.nn.Module):
	def __init__(self):
		super(Net,self).__init__()
		self.conv = torch.nn.Sequential(nn.Conv2d(1,64,kernel_size = KERNEL_SIZE, stride = 1, padding = 1),
										nn.ReLU(),
										nn.Conv2d(64,128,kernel_size = KERNEL_SIZE, stride = 1, padding = 1),
										nn.ReLU(),
										nn.MaxPool2d(stride=2, kernel_size = KERNEL_SIZE-1))
		self.dense = torch.nn.Sequential(nn.Linear(14*14*128, 1024),
										nn.ReLU(),
										nn.Dropout(p=0.5),
										nn.Linear(1024,10))
	def forward(self,x):
		x = self.conv(x)
		x = x.view(-1, 14*14*128)
		x = self.dense(x)
		return x

class CNN(nn.Module):
	def __init__(self):
		super(CNN, self).__init__()
		self.conv1 = nn.Sequential(
			nn.Conv2d(
				in_channels =1,
				out_channels = 16,
				kernel_size=5,
				stride=1,
				padding=2),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2)
		)
		self.conv2 = nn.Sequential(
			nn.Conv2d(16,32,5,1,2),
			nn.ReLU(),
			nn.MaxPool2d(2)
		)
		self.out = nn.Linear(32*7*7,10)

	def forward(self, x):
		x = self.conv1(x)
		x = self.conv2(x)
		x = x.view(x.size(0),-1)
		output = self.out(x)
		return output
		
net = CNN()
loss_func = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(),lr = 0.2, momentum = 0.9)

print(net)

def grad_norm(p):
    grad_sum = 0.0
    for element in net.parameters():
        grad = (element.grad.cpu().data.numpy()**p).sum()
        grad_sum += grad
    return grad_sum ** (1/p)

print (len(data_loader_train))
for epoch in range(EPOCH):
	epoch_grad = 0
	epoch_loss = 0
	'''for i, x in enumerate(data_loader_train):
		print (x)'''
	for step, (batch_x,batch_y) in enumerate(data_loader_train):
		batch_x,batch_y = Variable(batch_x), Variable(batch_y)
		optimizer.zero_grad()

		prediction = net(batch_x)

		loss = loss_func(prediction, batch_y)
		epoch_loss += loss.data.numpy()

		loss.backward()

		optimizer.step()
		epoch_grad += grad_norm(2)
		print ('epoch: %d, loss: %.7f, gradient: %.7f' %(epoch, loss, grad_norm(2)))
		gradient_list.append(grad_norm(2))
		loss_list.append(loss.data.numpy())
	'''print ('epoch: %d, loss: %.7f, gradient: %.7f' %(epoch, epoch_loss/len(data_loader_train),epoch_grad/len(data_loader_train) ))
	gradient_list.append(epoch_grad/len(data_loader_train))
	loss_list.append(epoch_loss/len(data_loader_train))'''

print(sum (p.numel() for p in net.parameters()))
print ('TRAINING FINISHING.')

torch.save(net,'MNIST.save')
print('save model.')


gradient = np.array(gradient_list)
x = np.linspace(0,len(data_loader_train),len(data_loader_train))
x_ep = np.linspace(0,EPOCH,EPOCH)
plt.subplot(211)
plt.plot(x,gradient_list,label = 'Gradient Norm')
plt.legend(loc = 'upper right')
#plt.show()
np.save('gradientNorm_MNIST.npy', gradient)
loss = np.array(loss_list)
plt.subplot(212)
plt.plot(x,loss_list, label = 'Loss', color = 'red')
plt.legend(loc = 'upper right')
plt.show()
np.save('loss_MNIST.npy',loss)
print('save loss.')

pd.DataFrame(gradient_list).to_csv('gradientNorm_MNIST.csv')
pd.DataFrame(loss_list).to_csv('loss_MNIST.csv')




