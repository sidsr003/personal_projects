import numpy as np

def one_hot_encode(x):
    num_classes = len(np.unique(x))
    one_hot = np.zeros((x.shape[0], num_classes))
    indices = np.concatenate((np.arange(x.shape[0]).reshape(-1,1), x.reshape(-1, 1)), axis=1)
    one_hot[indices[:, 0], indices[:, 1]] = 1
    return one_hot