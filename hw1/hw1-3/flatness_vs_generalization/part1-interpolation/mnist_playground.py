import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adam, SGD
from keras.utils import np_utils
from keras.datasets import mnist

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

model = Sequential()
model.add(Dense(input_dim = 28*28, units = 55, activation = 'sigmoid'))
model.add(Dense(units = 150, activation = 'sigmoid'))
model.add(Dense(units = 10, activation = 'softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer = Adam(lr=1e-2), metrics = ['accuracy'])

model.fit(x_train, y_train, batch_size = 64, epochs = 5)
score = model.evaluate(x_test, y_test)
print ('\nTest Loss = ', score[0])
print ('\nTest Acc = ', score[1])

#model.summary()
#model.save('mymnist_batch64.h5')
