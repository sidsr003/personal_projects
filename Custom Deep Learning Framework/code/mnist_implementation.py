import numpy as np
from layers import Dense
from activations import Sigmoid, Relu, Tanh, Softmax
from utils import one_hot_encode
from losses import MSE, Cross_Entropy
from network import Sequential
from tensorflow import keras
from initializers import Xavier, Uniform, Constant
from optimizers import Vanilla_Gradient_Descent, RMS_Prop

(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
X_train = X_train.reshape((X_train.shape[0], -1))/255
X_test = X_test.reshape((X_test.shape[0], -1))/255

y_train = one_hot_encode(y_train)
y_test = one_hot_encode(y_test)

net = Sequential([
    Dense(784, 64, activation=Tanh, kernel_initializer=Xavier()),
    Dense(64, 10, activation=Softmax, kernel_initializer=Uniform(-0.5, 0.5))
])

net.summary()
net.compile(Cross_Entropy(), RMS_Prop(1e-3))
net.fit(X_train[:60000], y_train[:60000], 32, 5)
print(net.validate(X_test, y_test)['accuracy'])

