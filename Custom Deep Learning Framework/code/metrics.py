import numpy as np

def accuracy(y_predicted, y):
    # expects one hot encoded y_predicted of shape (num_examples, num_classes)
    # expects y of shape (num_examples,) 
    predicted_classes = np.argmax(y_predicted, axis=1)
    y = np.argmax(y, axis=1)
    true_classes = y
    total = predicted_classes.shape[0]
    correct = np.sum(predicted_classes==true_classes)
    accuracy = correct/total
    return accuracy