import numpy as np
import argparse
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adam, SGD
from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import load_model
import keras.backend as K
import matplotlib.pyplot as plt


def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    d_len = 10000
    x_train = x_train[0:d_len]
    y_train = y_train[0:d_len]
    x_train = x_train.reshape(d_len, 28*28)
    x_test = x_test.reshape(x_test.shape[0], 28*28)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    # convert class vectors to binary classes
    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)
    x_train = x_train
    x_test = x_test

    x_train = x_train/255
    x_test = x_test /255
    return (x_train, y_train), (x_test, y_test)

(x_train, y_train), (x_test, y_test) = load_data()

#parse argument
parser = argparse.ArgumentParser()
parser.add_argument("path1")
parser.add_argument("path2")
parser.add_argument("-t","--title")
args = parser.parse_args()

model1 = load_model(args.path1)
model2 = load_model(args.path2)
new_model = load_model(args.path1)
weights_1 = model1.get_weights()
weights_2 = model2.get_weights()
test_losses, test_acc = [], []
train_losses, train_acc = [], []
alpha = np.arange(-1,2,0.01)
#print(alpha)

for a in alpha:
	new_weights = np.multiply(weights_1,(1-a))+np.multiply(weights_2,a)
	new_model.set_weights(new_weights)
	test_scores = new_model.evaluate(x_test,y_test)
	test_losses.append(test_scores[0])
	test_acc.append(test_scores[1])
	train_scores = new_model.evaluate(x_train,y_train)
	train_losses.append(train_scores[0])
	train_acc.append(train_scores[1])

test_losses = np.log(test_losses)
train_losses = np.log(train_losses)
#print (K.eval(model1.optimizer.lr))
fig, ax1 = plt.subplots()
ax1.plot(alpha,test_losses,'b-')
ax1.plot(alpha,train_losses,'b--')
ax1.set_xlabel('alpha')
ax1.set_ylabel('Cross Entropy',color='b')
ax1.tick_params('y',colors='b')

ax2 = ax1.twinx()
l1 = ax2.plot(alpha,test_acc,'r-')
l2 = ax2.plot(alpha,train_acc,'r--')
ax2.set_ylabel('Accuracy',color='r')
ax2.tick_params('y',colors='r')

plt.legend((l1[0],l2[0]),("test","train"),loc=1)

fig.tight_layout()
if args.title:
	plt.title(args.title)
	plt.savefig(args.title)
plt.show()
