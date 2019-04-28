import torch
import numpy as np
import pandas as pd
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.utils.data as Data
import random

BATCH_SIZE = 1024
NUM_WORKER = 0
epochs = 10
epochs_for_grad = 1000

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

training_loss = np.load('loss.npy')
loss_func = torch.nn.MSELoss()
min_ratio = []
  
for model in range(100):

    net = Net(1,1) 
    optimal_loss = training_loss[model]

    sample_num = 5000
    bigger_num = 0
    for sample in range(sample_num):
        net.load_state_dict(torch.load('./net_params_{}.pkl'.format(model)))

        for param in net.parameters():
            noise = (random.random() - 0.5) * 2
            param.data.add_(noise)
        pred = net(x_train)
        loss = loss_func(pred, y_train)
        loss = loss / 5000
        if loss > optimal_loss:
            bigger_num += 1
        if sample % (sample_num / 100) == 0:
            print ('Minimum_ratio: %06d / %06d' %(bigger_num, sample), end='\r')

    min_ratio.append(bigger_num / sample_num)
    print('\nModel[%03d] Minimum_ratio: %.6f, loss: %.6f' %(model, bigger_num / sample_num, training_loss[model]))
min_ratio = np.array(min_ratio)
np.save('minimum_ratio.npy',min_ratio)


