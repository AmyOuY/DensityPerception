# Compute Image Occupancy


import psychopy.visual as visual
import psychopy.event as event
import psychopy.core as core
import numpy as np  
from numpy.random import uniform, choice
import matplotlib.pyplot as plt


#count black and white pixels 
def pixelCount(img):
    black = 0
    white = 0
    for i in range(70, 790):
        for j in range(400, 1130):
            if img[i, j, 0] == 0:
                black +=1
            elif img[i, j, 0] == 255:
                white += 1
    return black, white



# Setup the Window
win = visual.Window(size=[1536, 864], fullscr=True, screen=0, allowGUI=False, allowStencil=False, monitor='testMonitor',
                    color=[0.1,0.1,0.1], colorSpace='rgb', blendMode='avg', useFBO=True)



# rectangle for cover up previous image
rect = visual.Rect(win=win, name='polygon', width=(11, 11)[0], height=(11, 11)[1], ori=0, pos=(0, 0), lineWidth=1, lineColor=[0.1,0.1,0.1],
                   lineColorSpace='rgb', fillColor=[0.1,0.1,0.1], fillColorSpace='rgb', opacity=1, depth=-2.0, interpolate=True)



#the field size and center are in deg of visual angle
field_x = 10
field_y = 10
totalPixels = 404496.0


def generateDots(win=win, n_dots=20, dot_size=0.2, overlap=0.5, color='white'):  
    xys = []     
    dot_counter = 0  
    while dot_counter < n_dots:
        valid = True               
        #randomly choose new dot position inside field
        dot_x = uniform(-field_x/2, field_x/2) 
        dot_y = uniform(-field_y/2, field_y/2) 
                
        #we check if new dot is not too close to existing dots
        for pos in xys:
            X = (pos[0] - dot_x)**2
            Y = (pos[1] - dot_y)**2
            distance = np.sqrt(X+Y)
            if distance <= overlap * dot_size:
                valid = False

        if valid == True:
            xys.append([dot_x, dot_y])
            dot_counter += 1
    
    
    stim = visual.ElementArrayStim(
        win=win,
        units="deg",
        nElements=n_dots,
        elementTex=None,
        elementMask="circle",
        xys=xys,
        sizes=dot_size,
        colors=color)
    
    return stim
    


front_ndots = 250
back_ndots = 250

dotSize = 0.28
overlap = 0.5


for i in range(5):    
    front_color = choice(['black','white'])
    if front_color == 'black':
        back_color = 'white'
    else:
        back_color = 'black'
        
    front_stim = generateDots(n_dots=front_ndots, dot_size=dotSize, overlap=overlap, color=front_color)
    back_stim = generateDots(n_dots=back_ndots, dot_size=dotSize, overlap=overlap, color=back_color)
    back_stim.draw()
    front_stim.draw() 
    
    I = win.getMovieFrame(buffer="back")
    win.saveMovieFrames('E500_0.4.png')
    img = np.asarray(I)
    black, white =pixelCount(img) 
    
    if front_color == 'black':
        frontCount = black
        backCount = white
    else:
        frontCount = white
        backCount = black

    #cover up the image to be ready for next image drawing
    rect.draw()
    win.flip()
    

    with open('d2.5_'+str(dotSize)+'_'+str(front_ndots)+'_'+str(back_ndots)+'.txt', 'a+') as f:        
        f.write("%d " %frontCount)                
        f.write("%d\n" %backCount)
            

            
win.close()    


frontCounts = []
backCounts = []
f = open('d2.5_0.28_250_250.txt', 'r')

for line in f:
    items = line.strip().split() 
    frontCounts.append(int(items[0]))
    backCounts.append(int(items[1]))
    
f.close()
