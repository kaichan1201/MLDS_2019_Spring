import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Input, Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import Adam, SGD
from keras.utils import np_utils
from keras.datasets import cifar10
from keras.callbacks import CSVLogger



def load_data():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    d_len = 1000
    x_train = x_train[0:d_len]
    y_train = y_train[0:d_len]
    x_train = x_train.reshape(d_len, 32,32, 3)
    x_test = x_test.reshape(x_test.shape[0], 32, 32, 3)
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
model.add( Convolution2D(43,3,3, input_shape = (32,32,3)) )
model.add(MaxPooling2D( (2,2) ))
model.add( Convolution2D(43,3,3) )
model.add(MaxPooling2D( (2,2) ))
model.add( Convolution2D(43,3,3) )
model.add(MaxPooling2D( (2,2) ))
model.add(Flatten())
model.add(Dense(units = 10, activation = 'softmax'))
model.summary()
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
csv_logger = CSVLogger('cifar10_1.csv')
model.fit(x_train, y_train, callbacks=[csv_logger] ,batch_size = 50, epochs = 60)
model.save('cifar10_3.h5')