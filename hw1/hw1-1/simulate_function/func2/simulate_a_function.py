import sys
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

if __name__ == "__main__":
    seed = 666
    np.random.seed(seed)

    batch_size = 128
    epochs = 20000

    x_data = np.linspace(0.0001, 1.0, num = 10000)
    y_data = np.sinc(5 * x_data) * np.sign(np.sin(10 * x_data))

    model0 = Sequential()
    model0.add(Dense(5, input_dim = 1, kernel_initializer = 'normal', activation = 'selu'))
    model0.add(Dense(10, kernel_initializer = 'normal', activation = 'selu'))
    model0.add(Dense(10, kernel_initializer = 'normal', activation = 'selu'))
    model0.add(Dense(10, kernel_initializer = 'normal', activation = 'selu'))
    model0.add(Dense(10, kernel_initializer = 'normal', activation = 'selu'))
    model0.add(Dense(10, kernel_initializer = 'normal', activation = 'selu'))
    model0.add(Dense(5, kernel_initializer = 'normal', activation = 'selu'))
    model0.add(Dense(1, kernel_initializer = 'normal'))
    print(model0.summary())

    model0.compile(loss = 'mse', optimizer = 'adam')
    fitHistory0 = model0.fit(x_data, y_data, epochs = epochs, batch_size = batch_size, shuffle = True)
    loss_history0 = fitHistory0.history['loss']
    model0.save('model0.h5')
    
    model1 = Sequential()
    model1.add(Dense(10, input_dim = 1, kernel_initializer = 'normal', activation = 'selu'))
    model1.add(Dense(18, kernel_initializer = 'normal', activation = 'selu'))
    model1.add(Dense(15, kernel_initializer = 'normal', activation = 'selu'))
    model1.add(Dense(4, kernel_initializer = 'normal', activation = 'selu'))
    model1.add(Dense(1, kernel_initializer = 'normal'))
    print(model1.summary())

    model1.compile(loss = 'mse', optimizer = 'adam')
    fitHistory1 = model1.fit(x_data, y_data, epochs = epochs, batch_size = batch_size, shuffle = True)
    loss_history1 = fitHistory1.history['loss']
    model1.save('model1.h5')

    model2 = Sequential()
    model2.add(Dense(190, input_dim = 1, kernel_initializer = 'normal', activation = 'selu'))
    model2.add(Dense(1, kernel_initializer = 'normal'))
    print(model2.summary())

    model2.compile(loss = 'mse', optimizer = 'adam')
    fitHistory2 = model2.fit(x_data, y_data, epochs = epochs, batch_size = batch_size, shuffle = True)
    loss_history2 = fitHistory2.history['loss']
    model2.save('model2.h5')

    plt.plot(loss_history0)
    plt.plot(loss_history1)
    plt.plot(loss_history2)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.yscale('log')
    plt.xlabel('epoch num')
    plt.legend(['model0_loss', 'model1_loss', 'model2_loss'], loc = 'lower left')
    plt.savefig('model_loss.png')
    plt.clf()

    x = np.linspace(0.001, 1.0, num = 100)
    y = np.sinc(5 * x) * np.sign(np.sin(10 * x))
    y_predict0 = np.squeeze(model0.predict(x))
    y_predict1 = np.squeeze(model1.predict(x))
    y_predict2 = np.squeeze(model2.predict(x))

    plt.plot(x, y, color = 'black')
    plt.plot(x, y_predict0, color = 'red')
    plt.plot(x, y_predict1, color = 'green')
    plt.plot(x, y_predict2, color = 'blue')
    plt.title('function')
    plt.ylabel('Y')
    plt.xlabel('X')
    plt.legend(['sinc(5x)*sign(sin(10x))', 'model0', 'model1', 'model2'], loc = 'upper right')
    plt.savefig('prediction.png')
