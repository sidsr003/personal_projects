import numpy as np
from networks import Network
from layers import Layer
from dense_layers import DenseLayer
from activation_layers import ActivationLayer
from losses import MSE, MSEDerivative
from activation_functions import tanh, tanhDerivative
from preprocessing import toGreyscale

from keras.datasets import mnist
from keras.utils import np_utils

#Loading the mnist database from server
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#x_train, x_test of shape (60000, 28, 28), (10000, 28, 28) and y_train, y_test of shape (60000,), (10000,)
#Now we reshape and normalise:

x_train = x_train.reshape(x_train.shape[0], 784, 1)
x_train = x_train.astype('float32')
x_train /= 255


y_train = np_utils.to_categorical(y_train)  #Adds an extra dimension by converting the number 3 for example to [0, 0, 0, 1, 0, 0, 0, 0, 0, 0] Shape change from (60000,) to 60000, 10)
y_train = y_train.reshape(y_train.shape[0], y_train.shape[1], 1)

                         
x_test = x_test.reshape(x_test.shape[0], 784, 1)
x_test = x_test.astype('float32')
x_test /= 255

y_test = np_utils.to_categorical(y_test)
y_test = y_test.reshape(y_test.shape[0], y_test.shape[1], 1)


#Now we establish the network

net = Network()
net.addLayer(DenseLayer(784, 100))
net.addLayer(ActivationLayer(tanh, tanhDerivative))
net.addLayer(DenseLayer(100, 50))
net.addLayer(ActivationLayer(tanh, tanhDerivative))
net.addLayer(DenseLayer(50, 10))
net.addLayer(ActivationLayer(tanh, tanhDerivative))

#We shall train on 1000 samples only
net.useLoss(MSE, MSEDerivative)
net.fit(x_train[5000:10000], y_train[5000:10000], epochs = 20, learningRate = 0.1)

#Now we shall load a single handwritten digit to test the model on. It must be a 1 x 784 x 1 array.
testData = toGreyscale()
output = np.array(net.predict(testData))
print(output)
print("The number input is a: ", np.argmax(output[0]))


        





























