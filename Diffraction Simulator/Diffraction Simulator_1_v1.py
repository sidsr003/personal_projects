from PIL import Image
import numpy as np
import math
import cmath
import time

L = 650 # Wavelength in nm
D = 1 # Slit-Screen distance
xInPixelSize = 5*10**(-7)# Dimension of a single pixel in real-world
yInPixelSize = 5*10**(-7)
xScreenSize = 1 # In metres
yScreenSize = 1 #In metres
outputWidth = 50 # x Screen size in pixels
outputHeight = 50 # y Screen size In pixels
outPath = 'imgOut.bmp'



# Functions defined below:

def dist(p, q):
    return math.sqrt(((p[0]-q[0])**2+(p[1]-q[1])**2+(p[2]-q[2])**2))
def calcExponential(p, q):
    distance = dist(p, q)
    j = 0+1j
    arg = (distance*k)*j
    return cmath.exp(arg)/distance

def computeIntensityAt(p, q):
    total = 0
    for i in range(0, inputWidth):
        for j in range(0, inputHeight):
            if (imgSrc.getpixel((i, j)) == 0):
                total += calcExponential(((i-inputWidth/2)*xInPixelSize, (j-inputHeight/2)*yInPixelSize, 0), ((p-outputWidth/2)*xOutPixelSize, (q-outputHeight/2)*yOutPixelSize, D))
    return abs(total)**2
def brightnessRemap(pixelVal): #Takes value between 0 and 1
    return pixelVal**0.25 #Using a concave function to improve visibilty
def produceImg():
    global L, D, imgSrc, inputWidth, inputHeight, xInPixelSize, yInPixelSize, xOutPixelSize, yOutPixelSize, k
    imgSrc = Image.open('imgSrc.bmp', 'r')
    inputWidth, inputHeight = imgSrc.size
    L = L*10**(-9) # Convert to metres
    k = 2*math.pi #(no lambda because we are using "lambda units" throughout)
    D = D/L # In "lambda units"
    xInPixelSize = xInPixelSize/L # In "lambda units"
    yInPixelSize = yInPixelSize/L # In "lambda units"
    xOutPixelSize = xScreenSize/outputWidth # x-Dimension of a single pixel in metric
    yOutPixelSize = yScreenSize/outputHeight # y-Dimension of a single pixel in metric
    xOutPixelSize = xOutPixelSize/L # In "lambda units"
    yOutPixelSize = yOutPixelSize/L # In "lambda units"
    
    imgOut = Image.new('RGB', (outputWidth, outputHeight)) # Create output image object
    imgPixels = np.zeros((outputWidth, outputHeight, 3)) # Create array to hold pixel values
    
    for i in range(0, outputWidth):
        for j in range(0, outputHeight):
            pixelVal = computeIntensityAt(i, j)
            imgPixels[i][j][0] = pixelVal
            print('Computed {0}\r'.format(i*outputHeight+j+1))

    print()
            
    strengthParameter = np.amax(imgPixels)
    if strengthParameter!=0:
        for i in range(0, outputWidth):
             for j in range(0, outputHeight):
                 pixelVal = brightnessRemap(imgPixels[i][j][0]/strengthParameter)
                 imgOut.putpixel((i, j), (int(255*pixelVal), 0, 0))
                 print('Normalised {0}\r'.format(i*outputHeight+j+1))
        print()

    imgOut.save(outPath)
    imgOut.show()

produceImg()
