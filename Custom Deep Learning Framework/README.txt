A from-scratch, custom API suite for deep learning

MNIST:
784 -- > 64 Relu --> 10 Sigmoid, MSE, lr=1.0, 40 epochs, acc=97.47%
784 -- > 64 Relu --> 10 None, MSE, lr=0.01, 100 epochs, acc=96.56%
784 -- > 64 Relu --> 10 Softmax, Cross_Entropy, lr=1, 5 epochs, acc=95.25%
784 -- > 64 (Xavier Inititalization) Relu --> 10 (Uniform Initialization(-0.5, 0.5)) Softmax, Cross_Entropy, Vanilla_Gradient_Descent, lr=0.01, 10 epochs, acc=95.59%
784 -- > 64 (Xavier Inititalization) Relu --> 10 (Uniform Initialization(-0.5, 0.5)) Softmax, Cross_Entropy, RMS_Prop, lr=0.001, 5 epochs, acc=96.69%