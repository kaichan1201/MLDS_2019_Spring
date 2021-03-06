import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
import random

torch.manual_seed(random.random() * 1000000)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden1 = nn.Linear(28 * 28, 55)
        self.hidden2 = nn.Linear(55, 150)
        self.predict = nn.Linear(150, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.selu(self.hidden1(x))
        x = F.selu(self.hidden2(x))
        x = self.predict(x)
        return F.log_softmax(x, dim = 1)

def train(epoch, train_loader, model):
    sens = 0
    loss = 0
    acc = 0
    optimizer = optim.Adam(model.parameters(), lr = 0.01)
    model.train() #all modules are initialized to train mode
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data, requires_grad = True), Variable(target)
        optimizer.zero_grad() #set all gradients to zero
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward() #backpropogation
        optimizer.step() #update
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]    Loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()), end = '\r')
        if epoch == 200:
            sens_batch = torch.norm(data.grad.data.view(len(data), -1), 2, 1).mean()
            sens += sens_batch
    if epoch == 200:
        loss, acc = test(train_loader, model, training_test = True)
        sens = sens/len(train_loader.dataset)
    return loss, acc, sens

def test(test_loader, model, training_test = False):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = Variable(data), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, reduction = 'sum').item() #sum up batch loss
        pred = output.data.max(1, keepdim = True)[1] #get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
    acc = 100. * correct / len(test_loader.dataset)
    test_loss /= len(test_loader.dataset)
    if (training_test):
        print('Training set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            test_loss, correct, len(test_loader.dataset),
            acc))
    else:
        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            test_loss, correct, len(test_loader.dataset),
            acc))
    return test_loss, acc

sensitivity_list = []
train_loss_list = []
test_loss_list = []
train_acc_list = []
test_acc_list = []

for i in range(3, 16):
    model = Net()
    batch = 2 ** i
    print('Begin training on batch size = {}'.format(batch))
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train = True, download = True,
                        transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307, ), (0.3081, ))
                        ])),
        batch_size = batch, shuffle = True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train = False,
                        transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307, ), (0.3081, ))
                        ])),
        batch_size = 1024, shuffle = True)
    trn_loss = 0
    trn_acc = 0
    s = 0
    test_los = 0
    test_acc = 0
    for epoch in range(1, 200 + 1):
        x, y, z = train(epoch, train_loader, model)
    w, k = test(test_loader, model)
    print('Sensitivity = {}'.format(z))
    sensitivity_list.append(z)
    train_loss_list.append(x)
    test_loss_list.append(w)
    train_acc_list.append(y)
    test_acc_list.append(k)
sensitivity_list = np.array(sensitivity_list)
np.save('sensitivity_list.npy', sensitivity_list)
train_loss_list = np.array(train_loss_list)
np.save('train_loss_list.npy', train_loss_list)
test_loss_list = np.array(test_loss_list)
np.save('test_loss_list.npy', test_loss_list)
train_acc_list = np.array(train_acc_list)
np.save('train_acc_list.npy', train_acc_list)
test_acc_list = np.array(test_acc_list)
np.save('test_acc_list.npy', test_acc_list)
