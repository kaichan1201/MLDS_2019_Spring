import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Input, Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import Adam, SGD
from keras.utils import np_utils
from keras.datasets import mnist
from keras.callbacks import CSVLogger



def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    d_len = 5000
    d_len_val = 1500
    d_len_rand = 300
    x_train = x_train[0:d_len]
    y_train = y_train[0:d_len]
    x_test_same = x_test[d_len_rand:d_len_val]
    x_test_shuffle = x_test[0:d_len_rand]
    np.random.shuffle(x_test_shuffle)
    x_test = np.concatenate((x_test_shuffle, x_test_same))
    y_test = y_test[0:d_len_val]
    x_train = x_train.reshape(d_len, 28,28,1)
    x_test = x_test.reshape(x_test.shape[0], 28,28,1)
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
model.add( Convolution2D(32,3,3, input_shape = (28,28,1)) )
model.add(MaxPooling2D( (2,2) ))
# model.add( Convolution2D(45,3,3) )
# model.add(MaxPooling2D( (2,2) ))
# model.add( Convolution2D(43,3,3) )
# model.add(MaxPooling2D( (2,2) ))
model.add(Flatten())
model.add(Dense(units = 32, activation = 'relu'))
model.add(Dense(units = 10, activation = 'softmax'))
model.summary()
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
csv_logger = CSVLogger('mnist_20.csv')
model.fit(x_train, y_train, callbacks=[csv_logger] ,batch_size = 50, epochs = 60, validation_data=(x_test, y_test))
# score = model.evaluate(x_train, y_train)
# print ('\nTest Acc = ', score[1])