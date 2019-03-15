import torch
import numpy as np
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.utils.data as Data
import random

BATCH_SIZE = 1024
NUM_WORKER = 0
epochs = 60
epochs_for_grad = 1000
rec_num = 0

x_train = torch.unsqueeze(torch.linspace(0.0001, 1.0, 5000),dim = 1)
y_train = torch.from_numpy(np.sinc(5*x_train))

x_train,y_train = Variable(x_train),Variable(y_train)

#plt.scatter(x_train.data.numpy(), y_train.data.numpy())
#plt.show()

class Net(torch.nn.Module):
	def __init__(self, n_feature, n_output):
		super(Net, self).__init__()

		self.hidden1 = torch.nn.Linear(n_feature, 5)
		self.hidden2 = torch.nn.Linear(5, 10)
		self.hidden3 = torch.nn.Linear(10, 5)
		self.predict = torch.nn.Linear(5, n_output)

	def forward(self, x):
		x = F.relu(self.hidden1(x))
		x = F.relu(self.hidden2(x))
		x = F.relu(self.hidden3(x))
		x = self.predict(x)
		return x


torch_dataset = Data.TensorDataset(x_train,y_train)
loader = Data.DataLoader(
            dataset = torch_dataset,
            batch_size = BATCH_SIZE,
            shuffle = True,
            num_workers = NUM_WORKER,
            )
print (loader)
#print (net)

def grad_norm(p):
    grad_sum = 0.0
    for element in net.parameters():
        grad = (element.grad.cpu().data.numpy()**p).sum()
        grad_sum += grad
    return grad_sum ** (1/p)

loss_list = []



while rec_num < 100 :
    net = Net(1,1)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.05)
    optimizer_g = torch.optim.Adam(net.parameters(), lr=0.0001)
    loss_func = torch.nn.MSELoss()
    for epoch in range(epochs):
        for (batch_x, batch_y) in loader: #把資料變成 （1,(x,y))
            #print('Epoch: ',epoch, '|Step: ', step, '|batch_x: ', batch_x.detach().numpy(), '|batch_y: ', batch_y.detach().numpy())
            batch_x, batch_y = Variable(batch_x), Variable(batch_y)
            
            prediction = net(batch_x)

            loss = loss_func(prediction, batch_y)
            # print("loss =") 
            # print(loss.data.numpy())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print("gradient_norm = ") 

    for epoch in range(epochs_for_grad):
        grad_0 = 0  # 0 gradient
        for (batch_x, batch_y) in loader: #把資料變成 （1,(x,y))
            #print('Epoch: ',epoch, '|Step: ', step, '|batch_x: ', batch_x.detach().numpy(), '|batch_y: ', batch_y.detach().numpy())
            batch_x, batch_y = Variable(batch_x), Variable(batch_y)
            
            prediction = net(batch_x)

            loss = loss_func(prediction, batch_y)
            # print("loss =") 
            # print(loss.data.numpy())
            optimizer_g.zero_grad()
            loss.backward()
            optimizer_g.step()
            # print("gradient_norm = ") 
            if(grad_norm(2) < 5e-3) : 
                print("grad_norm nearly 0 !!")
                torch.save(net.state_dict(), './net_params_{}.pkl'.format(rec_num))
                grad_0 = 1
                rec_num += 1
                loss_list.append(loss)
                break
        if grad_0 == 1 : break

loss = np.array(loss_list)
np.save('loss.npy', loss)






