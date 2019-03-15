import numpy as np
import os
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adam, SGD
from keras.utils import np_utils
from keras.datasets import mnist
from keras.callbacks import ModelCheckpoint

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

try:
    os.remove('loss_history.txt')
except OSError:
    pass

for i in range(8):
    print("%d time training" % i )
    model = Sequential()
    model.add(Dense(input_dim = 28*28, units = 55, activation = 'relu'))
    model.add(Dense(units = 150, activation = 'relu'))
    model.add(Dense(units = 10, activation = 'softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

    filepath="weights/%d_weights-record-{epoch:02d}.hdf5" % (i)
    checkpoint = ModelCheckpoint(filepath, verbose=1, period=3)
    hist = model.fit(x_train, y_train, batch_size = 100, epochs = 15,callbacks=[checkpoint])
    score = model.evaluate(x_test, y_test)
    print ('\nTest Acc = ', score[1])

    #save loss history
    loss_history_3 = [v for (i,v) in enumerate(hist.history['loss']) if (i+1)%3==0]
    with open('loss_history.txt',mode='a') as myfile:
        myfile.write('\n'.join(str(line) for line in loss_history_3))
        myfile.write('\n')

model.summary()
model.save('mymnist.h5')
