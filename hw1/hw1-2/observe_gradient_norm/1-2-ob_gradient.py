import torch
import numpy as np
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.utils.data as Data

BATCH_SIZE = 128
NUM_WORKER = 1
epochs = 20000
gradient_list = []
loss_list = []

x_train = torch.unsqueeze(torch.linspace(0.000001,1,10000),dim = 1)
y_train = torch.from_numpy(np.sinc(5*x_train))

x_train,y_train = Variable(x_train),Variable(y_train)

#plt.scatter(x_train.data.numpy(), y_train.data.numpy())
#plt.show()

class Net(torch.nn.Module): 
    def __init__(self, n_features, n_hidden, n_output): #搭圖
        super(Net,self).__init__()
        self.hidden_1 = torch.nn.Linear(n_features, n_hidden)
        self.hidden_2 = torch.nn.Linear(n_hidden, n_hidden)
        self.hidden_3 = torch.nn.Linear(n_hidden, n_hidden)
        self.hidden_4 = torch.nn.Linear(n_hidden, n_hidden)
        self.hidden_5 = torch.nn.Linear(n_hidden, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)
    def forward(self, x): #f前向傳遞
        x = F.relu(self.hidden_1(x))
        x = F.relu(self.hidden_2(x))
        x = F.relu(self.hidden_3(x))
        x = F.relu(self.hidden_4(x))
        x = F.relu(self.hidden_5(x))
        x = self.predict(x)
        return x

net = Net(1,10,1) 
torch_dataset = Data.TensorDataset(x_train,y_train)
loader = Data.DataLoader(
            dataset = torch_dataset,
            batch_size = BATCH_SIZE,
            shuffle = True,
            num_workers = NUM_WORKER,
            )
print (loader)
print (net)

def grad_norm(p):
    grad_sum = 0.0
    for element in net.parameters():
        grad = (element.grad.cpu().data.numpy()**p).sum()
        grad_sum += grad
    return grad_sum ** (1/p)

'''plt.ion()
plt.show()'''

optimizer = torch.optim.SGD(net.parameters(), lr=0.5)
loss_func = torch.nn.MSELoss()

for epoch in range(epochs):
    epoch_grad = 0
    epoch_loss = 0
    #print(len(loader))
    for step, (batch_x, batch_y) in enumerate(loader): #把資料變成 （1,(x,y))
        #print('Epoch: ',epoch, '|Step: ', step, '|batch_x: ', batch_x.detach().numpy(), '|batch_y: ', batch_y.detach().numpy())
        batch_x, batch_y = Variable(batch_x), Variable(batch_y)
        
        prediction = net(batch_x)

        loss = loss_func(prediction, batch_y)
        epoch_loss += loss.data.numpy()

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()
        epoch_grad += grad_norm(2)

        '''print ('epoch: %d, loss: %.7f, gradient: %.7f' %(epoch, loss.data.numpy(),grad_norm(2)) )
        gradient_list.append(grad_norm(2))
        loss_list.append(loss.data.numpy())'''

    print ('epoch: %d, loss: %.7f , gradient: %.7f' %(epoch, epoch_loss/len(loader), epoch_grad/len(loader)))
    gradient_list.append(epoch_grad / len(loader))
    loss_list.append(epoch_loss / len(loader))

    '''if (epoch%5 ==0):
        plt.cla()
        plt.scatter(batch_x.data.numpy(),batch_y.data.numpy())
        plt.plot(batch_x.numpy(), prediction.data.numpy(),'r-',lw=5)
        #plt.text(0.5,0,'Loss=%.4f' % loss.data[0],fontdict={'size':20,'color':'red'})
        plt.pause(0.1)'''
'''plt.ioff()
plt.show()'''

print(sum (p.numel() for p in net.parameters()))
print ('TRAINING FINISHING.')

torch.save(net,'sinc1.save')
print('save model.')


gradient = np.array(gradient_list)
x = np.linspace(0,epochs,epochs)
plt.subplot(211)
plt.plot(x,gradient_list,label = 'Gradient')
plt.legend(loc = 'best')
np.save('gradientNorm.npy', gradient)
loss = np.array(loss_list)
plt.subplot(212)
plt.plot(x,loss_list, label = 'Loss', color = 'red')
plt.legend(loc = 'best')
plt.show()
np.save('loss.npy',loss)
print('save loss.')

#plot loss and gradient








