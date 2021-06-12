# Spatial Frequency (SF) ratio model


import psychopy.visual as visual
import psychopy.event as event
from random import random, choice
import numpy as np
from math import pi, cos, sin, exp 
import matplotlib.pyplot as plt
from matplotlib.image import imread
from matplotlib.pyplot import imshow


# laplacain of gaussian function
def laplacianGaussian(N, sigma):
    #fiter size is N
    log = np.zeros((N, N))
    
    #center origin to (r0,c0)
    #r0 = (N+1)/2 
    #c0 = (N+1)/2
    
    for r in range(-(N-1)/2,(N-1)/2):
        for c in range(-(N-1)/2,(N-1)/2): 
            x = r + r0;
            y = c + c0;
            e = -(r**2+c**2)/(sigma**2)
            #laplacian of gaussian formula
            log[y,x] = (1/(pi*sigma**4))*(1+e/2)*exp(e/2)
    #normalization    
    s=np.sum(np.sum(log));
    log=log/s;
    log=log-s/N**2;
        
    return log



# 2-d convolution using 'valid' mode
def convolution2d(image, kernel):
    m, n = kernel.shape
    y, x = image.shape[0], image.shape[1]
    y = y - m + 1
    x = x - m + 1    
    newImage = np.zeros((y,x))
    #"valid" mode
    for i in range(y):
        for j in range(x):
            newImage[i][j] = np.sum(image[i:i+m, j:j+m, 0]*kernel) 
            
    return newImage



win = visual.Window(size=[1536, 864], fullscr=True, screen=0, allowGUI=False, allowStencil=False,
    monitor='testMonitor', color=[0.1,0.1,0.1], colorSpace='rgb', blendMode='avg', useFBO=True)


#rectangle for covering up previous image 
rect = visual.Rect(win=win, name='polygon', width=(1.5, 1.5)[0], height=(1.5, 1.5)[1], ori=0, pos=(0, 0), lineWidth=1, lineColor=[0.1,0.1,0.1],
    lineColorSpace='rgb', fillColor=[0.1,0.1,0.1], fillColorSpace='rgb',opacity=1, depth=-2.0, interpolate=True)


dotSize = 0.15

#generate Gaussian blobs that uniformly distributed within circular region of specific radius
def generateDots(win=win, radius=2, nDots=128, offset=-6):
    count = 0
    dots = [] #all blobs array
    dots_1 = [] #white color blobs array
    dots_2 = [] #black color blobs array

    while count < nDots:
        valid = True
        a = random() * 2 * pi
        r = radius * np.sqrt(random())
        dotX = r * cos(a) + offset
        dotY = r * sin(a)
        for pos in dots:
            X = (pos[0] - dotX)**2
            Y = (pos[1] - dotY)**2
            dist = np.sqrt(X+Y)
            if dist < dotSize: #no overlap between blobs
                valid = False
        if valid == True:
            dots.append([dotX, dotY])
            count += 1
            color = choice(['black','white'])
            if color == 'white':
                dots_1.append([dotX, dotY])
            else:
                dots_2.append([dotX, dotY])
               
    stim1 = visual.ElementArrayStim(
        win=win,
        units="deg",
        nElements=len(dots_1),
        elementTex=None,
        xys=dots_1,
        sizes=dotSize,        
        colors="white",
        elementMask="gauss")

    stim2 = visual.ElementArrayStim(
        win=win,
        units="deg",
        nElements=len(dots_2),
        elementTex=None,
        xys=dots_2,
        sizes=dotSize,        
        colors="black",
        elementMask="gauss")
    
    return stim1, stim2



N = 101   
sigma_hi = 1.0 #high-frequency filter
kernel_hi =  laplacianGaussian(N, sigma_hi)
sigma_lo = 11.0 #low-frequency filter
kernel_lo =  laplacianGaussian(N, sigma_lo)

refRadius = 2
testRadius = 2

refDots = 128
testDots = 128

hiSum = []
loSum = []
ratio = []


for i in range(20):
    refOffset = choice([-5, 5])
    refStim1, refStim2 = generateDots(radius=refRadius, nDots=refDots, offset=refOffset)    
    testStim1, testStim2 = generateDots(radius=testRadius, nDots=testDots, offset=-refOffset)
    
    refStim1.colors = "black"
    testStim1.colors = "black"
    
    refStim1.draw() 
    refStim2.draw()
    testStim1.draw() 
    testStim2.draw()
        
    #take a screen shot of the image  
    I = win.getMovieFrame(buffer="back")
    win.saveMovieFrames('R'+str(testRadius)+'_'+'n'+str(testDots)+'_'+str(i)+".png")    
    
    #cover up previous image with a background rectangle
    rect.draw()
    win.flip()    
       
    image = np.asarray(I)
    #crop out useless background pixels
    image = image[80:790, 110:1440, :]
     
    conv_hi = convolution2d(image, kernel_hi)
    intensityS_hi = np.sum(np.absolute(conv_hi))
    hiSum.append(intensityS_hi)
    imshow(conv_hi)
   
    
    conv_lo = convolution2d(image, kernel_lo)
    intensityS_lo = np.sum(np.absolute(conv_lo))
    loSum.append(intensityS_lo)
    #imshow(conv_lo)
    ratio.append(intensityS_hi/intensityS_lo)

win.close()
