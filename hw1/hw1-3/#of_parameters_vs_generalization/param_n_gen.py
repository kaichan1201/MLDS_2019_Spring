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
import argparse

BATCH_SIZE = 1024
TEST_BATCH_SIZE = 1000
KERNAL_SIZE = 3
EPOCH = 10
LEARNING_RATE = 0.01
loss_list_train = []
loss_list_test = []
accuracy_list_train = []
accuracy_list_test = []
NUM_WORKER = 1


params = []
Record_Train_Loss = []
Record_Train_Acc = []
Record_Test_Loss = []
Record_Test_Acc = []


class CNN(nn.Module):
	f1 = 0
	f2 = 0
	def __init__(self,filter_n1):
		super(CNN,self).__init__()
		self.f1 = filter_n1
		#self.f2 = filter_n2
		self.conv1 = nn.Conv2d(1,self.f1,5,1)
		#self.conv2 = nn.Conv2d(self.f1,self.f2,5,1)
		self.fc1 = nn.Linear(12*12*self.f1,500)
		self.fc2 = nn.Linear(500,10)

	def forward(self,x):
		x = F.relu(self.conv1(x))
		x = F.max_pool2d(x,2,2)
		#x = F.relu(self.conv2(x))
		#x = F.max_pool2d(x,2,2)
		x = x.view(-1,12*12*self.f1)
		x = F.selu(self.fc1(x))
		x = self.fc2(x)
		return F.log_softmax(x,dim=1)

def train(args, model, device, train_loader, optimizer, epoch,train_loss):
	model.train()
	train_loss = 0
	for batch_idx, (data, target) in enumerate(train_loader):
		data, target = data.to(device), target.to(device)
		optimizer.zero_grad()
		output = model(data)
		loss = F.nll_loss(output, target)
		train_loss += loss
		#print (loss)
		loss.backward()
		optimizer.step()
		pred = output.argmax(dim=1,keepdim=True)
		#if batch_idx % args.log_interval == 0:
		#	print('Train Epoch:{} [{}/{} ({:.0f}%)]\tLoss: {:.7f}'.format(epoch, batch_idx*len(data),len(train_loader.dataset),100.*batch_idx/len(train_loader),loss.item()))
	train_loss /= len(train_loader.dataset)
	print ('\n Train set: Average loss: {:.5f}'.format(train_loss))

def test(args, model, device, test_loader):
	model.eval()
	test_loss = 0
	test_acc = 0
	with torch.no_grad(): #no traced by autograd
		for data, target in test_loader:
			data, target = data.to(device), target.to(device) #move/cast the params and buffers
			output = model(data)
			test_loss += F.nll_loss(output, target, reduction='sum').item() #sum up batch loss
			pred = output.argmax(dim=1, keepdim=True)
			test_acc += pred.eq(target.view_as(pred)).sum().item() #compute element-wise equality

	test_loss /= len(test_loader.dataset)
	print('\nTest set: Average loss: {:.5f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss,test_acc,len(test_loader.dataset),100.*test_acc/len(test_loader.dataset)))
	test_acc /= len(test_loader.dataset)
	return test_loss, test_acc

def main():
	parser = argparse.ArgumentParser(description = 'Number of parameters v.s. Generalization')
	parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, metavar='N',help='input batch size for training')
	parser.add_argument('--test-batch-size',type=int, default=TEST_BATCH_SIZE,metavar='N',help='input batch size for testing')
	parser.add_argument('--epochs',type=int,default=EPOCH,metavar='N',help='number of epochs to train')
	parser.add_argument('--lr',type=float,default=LEARNING_RATE,metavar='LR',help='learning rate')
	parser.add_argument('--momentum',type=float,default=0.5,metavar='M',help='SGD momentum')
	parser.add_argument('--no-cuda',action='store_true',default=False,help='disables CUDA training')
	parser.add_argument('--seed',type=int,default=1,metavar='S',help='random seed')
	parser.add_argument('--log-interval',type=int,default=50,metavar='N',help='how many batches to wait before logging trainig status')
	parser.add_argument('--save-model',action='store_true',default=True,help='For Saving the current Model')

	args = parser.parse_args()
	use_cuda = not args.no_cuda and torch.cuda.is_available()

	torch.manual_seed(args.seed)

	device = torch.device("cuda" if use_cuda else "cpu")

	kwargs = {'num_workers': NUM_WORKER, 'pin_memory':True} if use_cuda else {}
	
	data_train = datasets.MNIST(root='MNIST_train.npy',transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081,))]),train=True,download=True)
	data_test = datasets.MNIST(root='MNIST_test.npy',transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081,))]),train=False,download=True)

	data_loader_train = Data.DataLoader(dataset=data_train,batch_size=args.batch_size,shuffle=True,**kwargs)
	data_loader_test = Data.DataLoader(dataset=data_test,batch_size=args.batch_size,shuffle=True,**kwargs)

	filter_1 = np.linspace(5,500,20) 
	#filter_2 = np.linspace(50,150,5)

	#for filter_n2 in filter_2:
	for filter_n1 in filter_1:
		model = CNN(int(filter_n1)).to(device)
		optimizer = optim.SGD(model.parameters(),lr=args.lr,momentum=args.momentum)

		train_loss, train_acc, test_loss, test_acc, train_val_loss= 0.,0.,0.,0.,0.
		for epoch in range(1, args.epochs+1):
			train(args, model, device, data_loader_train, optimizer, epoch, train_loss)

		train_val_loss, train_acc = test(args,model,device,data_loader_train)
		test_loss, test_acc = test(args, model, device, data_loader_test)
		if(args.save_model):
			torch.save(model.state_dict(),"param_n_gen.pt")

		model_param = sum(p.numel() for p in model.parameters())
		print (model_param)
		params.append(model_param)
		Record_Train_Loss.append(train_val_loss)
		Record_Train_Acc.append(train_acc)
		Record_Test_Loss.append(test_loss)
		Record_Test_Acc.append(test_acc)
	print (Record_Test_Loss)
	print (Record_Train_Acc)

	#plot
	plt.subplot(211)
	plt.xlabel ('number of parameters', fontsize = 16)
	plt.ylabel ('accuracy', fontsize = 16)
	line1 = plt.scatter(np.array(params),np.array(Record_Train_Acc),label = 'Train Accuracy', color = 'blue')
	line2 = plt.scatter(np.array(params),np.array(Record_Test_Acc),label = 'Test Accuracy', color = 'red')
	plt.legend(handles = [line1,line2], loc='lower right')
	plt.subplot(212)
	plt.xlabel ('number of parameters', fontsize = 16)
	plt.ylabel ('loss', fontsize = 16)
	line3 = plt.scatter(np.array(params),np.array(Record_Train_Loss),label = 'Train Loss', color = 'blue')
	line4 = plt.scatter(np.array(params),np.array(Record_Test_Loss),label = 'Test Loss', color = 'red')
	plt.legend(handles = [line3,line4], loc='upper right')
	plt.show()


if __name__ == '__main__':
	main()



















