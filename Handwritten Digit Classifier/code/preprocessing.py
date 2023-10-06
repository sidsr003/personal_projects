from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def toGreyscale():
    img = Image.open('New Digits\img1.png')
    imgData = np.array(list(img.getdata()))
    newData = np.ndarray(shape=(1, 784, 1), dtype = 'int32')
    for i in range(784):
        newData[0, i,0] = np.mean(imgData[i])
    return (255-newData)























