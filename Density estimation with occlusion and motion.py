"""
Density estimation with occlusion and motion involved.
This experiment was created using PsychoPy2 Experiment Builder (v1.90.3),
    on July 30, 2019, at 18:22
If you publish work using this script please cite the PsychoPy publications:
    Peirce, JW (2007) PsychoPy - Psychophysics software in Python.
        Journal of Neuroscience Methods, 162(1-2), 8-13.
    Peirce, JW (2009) Generating stimuli for neuroscience using PsychoPy.
        Frontiers in Neuroinformatics, 2:10. doi: 10.3389/neuro.11.010.2008
"""


from __future__ import absolute_import, division
from psychopy import locale_setup, sound, gui, visual, core, data, event, logging, clock
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER)
import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle
import os  # handy system and path functions
import sys  # to get file system encoding

# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__)).decode(sys.getfilesystemencoding())
os.chdir(_thisDir)

# Store info about the experiment session
expName = 'density10x10'  # from the Builder filename that created this script
expInfo = {u'session': u'001', u'participant': u''}
dlg = gui.DlgFromDict(dictionary=expInfo, title=expName)
if dlg.OK == False:
    core.quit()  # user pressed cancel
expInfo['date'] = data.getDateStr()  # add a simple timestamp
expInfo['expName'] = expName

# Data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
filename = _thisDir + os.sep + u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])

# An ExperimentHandler isn't essential but helps with data saving
thisExp = data.ExperimentHandler(name=expName, version='',
    extraInfo=expInfo, runtimeInfo=None,
    originPath=u'C:\\Users\\ouyim\\Desktop\\Yimiao Ou Project Report\\partII_occlusionMotionExperiment.psyexp',
    savePickle=True, saveWideText=True,
    dataFileName=filename)
# save a log file for detail verbose info
logFile = logging.LogFile(filename+'.log', level=logging.EXP)
logging.console.setLevel(logging.WARNING)  # this outputs to the screen, not a file

endExpNow = False  # flag for 'escape' or other condition => quit the exp

# Start Code - component code to be run before the window creation

# Setup the Window
win = visual.Window(
    size=[1536, 864], fullscr=True, screen=0,
    allowGUI=False, allowStencil=False,
    monitor='testMonitor', color=[0.2,0.2,0.2], colorSpace='rgb',
    blendMode='avg', useFBO=True)
# store frame rate of monitor if we can measure it
expInfo['frameRate'] = win.getActualFrameRate()
if expInfo['frameRate'] != None:
    frameDur = 1.0 / round(expInfo['frameRate'])
else:
    frameDur = 1.0 / 60.0  # could not measure, so guess


# Initialize components for Routine "instruction"
instructionClock = core.Clock()
instruction_text = visual.TextStim(win=win, name='instruction_text',
    text="You will be presented image containing two layers of moving dots.\n\nYour task is to choose which of the front or back layer dots appear more dense to you.\n\nPress 'i' for back and 'm' for front. Once you press 'i' or 'm', the next image will appear.\n\nPress 'space' to continue towards some training rounds.",
    font='Arial',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1,
    depth=0.0);


# Initialize components for Routine "functions"
functionsClock = core.Clock()
import numpy as np  
from numpy.random import uniform, choice


#the field size and center are in deg of visual angle
field_x = 10
field_y = 10

overlap = 0.5

# Function for generating dotted layer
def generateDots(win=win, n_dots=20, dot_size=0.2, overlap=overlap, color='white'):  

    xys = []     
    dot_counter = 0  
    while dot_counter < n_dots:
        valid = True               
        # randomly choose new dot position inside field
        dot_x = uniform(-field_x/2, field_x/2) 
        dot_y = uniform(-field_y/2, field_y/2) 
                
        # we check if new dot is not too close to existing dots
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
    


# Initialize components for Routine "training_round"
training_roundClock = core.Clock()


# Initialize components for Routine "buffer"
bufferClock = core.Clock()
buffer_text = visual.TextStim(win=win, name='buffer_text',
    text=None,
    font='Arial',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1,
    depth=0.0);

# Initialize components for Routine "train_to_trial"
train_to_trialClock = core.Clock()
train_to_trial_text = visual.TextStim(win=win, name='train_to_trial_text',
    text="You finished the training rounds.\n\nIf you want to take a break, please don't press any key.\n\nOnce you're ready, press 'space' to start.",
    font='Arial',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1,
    depth=0.0);
    
    
# define dot-size, dot-number and speed for different layers    
def addParameters(baseNum, nums, dotSize, ampSine, halfCycle):
    ans = []
    for k in nums.keys():
        pattern = {}
        pattern['baseNum'] = baseNum
        pattern['frontNum'] = k
        pattern['backNum'] = nums[k]
        pattern['dotSize'] = dotSize
        pattern['ampSine'] = ampSine
        pattern['halfCycle'] = halfCycle
        ans.append(pattern)
    return ans


d25_nums = {275:225, 269:231, 263:238, 256:244, 250:250, 244:256, 238:263, 231:269, 225:275}

d5_nums = {550:450, 538:463, 525:475, 513:488, 488:513, 475:525, 463:538, 450:550}

d10_nums = {1100:900, 1075:925, 1050:950, 1025:975, 1000:1000, 975:1025, 950:1050, 925:1075, 900:1100}

d500 = {500:500}


parameters = []

parameters.extend(addParameters(250, d25_nums, 0.28, 0.048, 28))
parameters.extend(addParameters(1000, d10_nums, 0.28, 0.03, 45))
 
parameters.extend(addParameters(500, d5_nums, 0.2, 0.048, 28))
parameters.extend(addParameters(500, d5_nums, 0.4, 0.048, 28))

parameters.extend(addParameters(500, d500, 0.2, 0.056, 24))
parameters.extend(addParameters(500, d500, 0.4, 0.056, 24))



shuffle(parameters)

trialCount = 0

# Initialize components for Routine "trial"
trialClock = core.Clock()


# Initialize components for Routine "buffer"
bufferClock = core.Clock()
buffer_text = visual.TextStim(win=win, name='buffer_text',
    text=None,
    font='Arial',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1,
    depth=0.0);

# Initialize components for Routine "subsessionBuffer"
subsessionBufferClock = core.Clock()
subsessionText = visual.TextStim(win=win, name='subsessionText',
    text="You have completed one sub_session.\n\nPress 'space' to continue.",
    font='Arial',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1,
    depth=0.0);


# Initialize components for Routine "end"
endClock = core.Clock()
end_text = visual.TextStim(win=win, name='end_text',
    text='You finished the experiment.\n\nThank you for your participation!\n',
    font='Arial',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1,
    depth=0.0);

# Create some handy timers
globalClock = core.Clock()  # to track the time since experiment started
routineTimer = core.CountdownTimer()  # to track time remaining of each (non-slip) routine 

# ------Prepare to start Routine "instruction"-------
t = 0
instructionClock.reset()  # clock
frameN = -1
continueRoutine = True
# update component parameters for each repeat
instruction_key = event.BuilderKeyResponse()
# keep track of which components have finished
instructionComponents = [instruction_text, instruction_key]
for thisComponent in instructionComponents:
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED

# -------Start Routine "instruction"-------
while continueRoutine:
    # get current time
    t = instructionClock.getTime()
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *instruction_text* updates
    if t >= 0.0 and instruction_text.status == NOT_STARTED:
        # keep track of start time/frame for later
        instruction_text.tStart = t
        instruction_text.frameNStart = frameN  # exact frame index
        instruction_text.setAutoDraw(True)
    
    # *instruction_key* updates
    if t >= 1 and instruction_key.status == NOT_STARTED:
        # keep track of start time/frame for later
        instruction_key.tStart = t
        instruction_key.frameNStart = frameN  # exact frame index
        instruction_key.status = STARTED
        # keyboard checking is just starting
        event.clearEvents(eventType='keyboard')
    if instruction_key.status == STARTED:
        theseKeys = event.getKeys(keyList=['space'])
        
        # check for quit:
        if "escape" in theseKeys:
            endExpNow = True
        if len(theseKeys) > 0:  # at least one key was pressed
            # a response ends the routine
            continueRoutine = False
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in instructionComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # check for quit (the Esc key)
    if endExpNow or event.getKeys(keyList=["escape"]):
        core.quit()
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "instruction"-------
for thisComponent in instructionComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
# the Routine "instruction" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# ------Prepare to start Routine "functions"-------
t = 0
functionsClock.reset()  # clock
frameN = -1
continueRoutine = True
# update component parameters for each repeat

# keep track of which components have finished
functionsComponents = []
for thisComponent in functionsComponents:
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED

# -------Start Routine "functions"-------
while continueRoutine:
    # get current time
    t = functionsClock.getTime()
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in functionsComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # check for quit (the Esc key)
    if endExpNow or event.getKeys(keyList=["escape"]):
        core.quit()
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "functions"-------
for thisComponent in functionsComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)

# the Routine "functions" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# set up handler to look after randomisation of conditions etc
train = data.TrialHandler(nReps=5, method='random', 
    extraInfo=expInfo, originPath=-1,
    trialList=[None],
    seed=None, name='train')
thisExp.addLoop(train)  # add the loop to the experiment
thisTrain = train.trialList[0]  # so we can initialise stimuli with some values
# abbreviate parameter names if possible (e.g. rgb = thisTrain.rgb)
if thisTrain != None:
    for paramName in thisTrain:
        exec('{} = thisTrain[paramName]'.format(paramName))

for thisTrain in train:
    currentLoop = train
    # abbreviate parameter names if possible (e.g. rgb = thisTrain.rgb)
    if thisTrain != None:
        for paramName in thisTrain:
            exec('{} = thisTrain[paramName]'.format(paramName))
    
    # ------Prepare to start Routine "training_round"-------
    t = 0
    training_roundClock.reset()  # clock
    frameN = -1
    continueRoutine = True
    
    
    # update component parameters for each repeat
    front_color = choice(['black','white'])
    if front_color == 'black':
        back_color = 'white'
    else:
        back_color = 'black'
      
        
    front_stim = generateDots(n_dots=900, dot_size=0.28, overlap=overlap, color=front_color)
    back_stim = generateDots(n_dots=1100, dot_size=0.28, overlap=overlap, color=back_color)
        
    back_stim.draw()
    front_stim.draw()  
    win.flip()
        
    
    amp = 0.03 #amplitude of sine function
    halfT = 45 #time for half cycle
    w = np.pi/halfT #angular velocity
    
    
    for i in range(halfT*4):        
        frontPos = front_stim.xys
        backPos = back_stim.xys    
    
        for j in range(len(frontPos)):
            frontPos[j][0] += amp*np.sin(w*i)       
       
        for j in range(len(backPos)):  
            backPos[j][0] -= amp*np.sin(w*i)
            
        front_stim.xys = frontPos
        back_stim.xys = backPos
    
    
        back_stim.draw()
        front_stim.draw() 
        win.flip()
        
        
    train_key = event.BuilderKeyResponse()
    # keep track of which components have finished
    training_roundComponents = [train_key]
    for thisComponent in training_roundComponents:
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    
    # -------Start Routine "training_round"-------
    while continueRoutine:
        # get current time
        t = training_roundClock.getTime()
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        
        # *train_key* updates
        if t >= 0.0 and train_key.status == NOT_STARTED:
            # keep track of start time/frame for later
            train_key.tStart = t
            train_key.frameNStart = frameN  # exact frame index
            train_key.status = STARTED
            # keyboard checking is just starting
            event.clearEvents(eventType='keyboard')
        if train_key.status == STARTED:
            theseKeys = event.getKeys(keyList=['i', 'm'])
            
            # check for quit:
            if "escape" in theseKeys:
                endExpNow = True
            if len(theseKeys) > 0:  # at least one key was pressed
                # a response ends the routine
                continueRoutine = False
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in training_roundComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # check for quit (the Esc key)
        if endExpNow or event.getKeys(keyList=["escape"]):
            core.quit()
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "training_round"-------
    for thisComponent in training_roundComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    front_stim.setAutoDraw(False)
    back_stim.setAutoDraw(False)
    # the Routine "training_round" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # ------Prepare to start Routine "buffer"-------
    t = 0
    bufferClock.reset()  # clock
    frameN = -1
    continueRoutine = True
    routineTimer.add(0.500000)
    # update component parameters for each repeat
    # keep track of which components have finished
    bufferComponents = [buffer_text]
    for thisComponent in bufferComponents:
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    
    # -------Start Routine "buffer"-------
    while continueRoutine and routineTimer.getTime() > 0:
        # get current time
        t = bufferClock.getTime()
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *buffer_text* updates
        if t >= 0.0 and buffer_text.status == NOT_STARTED:
            # keep track of start time/frame for later
            buffer_text.tStart = t
            buffer_text.frameNStart = frameN  # exact frame index
            buffer_text.setAutoDraw(True)
        frameRemains = 0.0 + 0.5- win.monitorFramePeriod * 0.75  # most of one frame period left
        if buffer_text.status == STARTED and t >= frameRemains:
            buffer_text.setAutoDraw(False)
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in bufferComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # check for quit (the Esc key)
        if endExpNow or event.getKeys(keyList=["escape"]):
            core.quit()
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "buffer"-------
    for thisComponent in bufferComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
# completed 5 repeats of 'train'


# ------Prepare to start Routine "train_to_trial"-------
t = 0
train_to_trialClock.reset()  # clock
frameN = -1
continueRoutine = True
# update component parameters for each repeat
train_to_trial_key = event.BuilderKeyResponse()

# keep track of which components have finished
train_to_trialComponents = [train_to_trial_text, train_to_trial_key]
for thisComponent in train_to_trialComponents:
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED

# -------Start Routine "train_to_trial"-------
while continueRoutine:
    # get current time
    t = train_to_trialClock.getTime()
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *train_to_trial_text* updates
    if t >= 0.0 and train_to_trial_text.status == NOT_STARTED:
        # keep track of start time/frame for later
        train_to_trial_text.tStart = t
        train_to_trial_text.frameNStart = frameN  # exact frame index
        train_to_trial_text.setAutoDraw(True)
    
    # *train_to_trial_key* updates
    if t >= 1 and train_to_trial_key.status == NOT_STARTED:
        # keep track of start time/frame for later
        train_to_trial_key.tStart = t
        train_to_trial_key.frameNStart = frameN  # exact frame index
        train_to_trial_key.status = STARTED
        # keyboard checking is just starting
        event.clearEvents(eventType='keyboard')
    if train_to_trial_key.status == STARTED:
        theseKeys = event.getKeys(keyList=['space'])
        
        # check for quit:
        if "escape" in theseKeys:
            endExpNow = True
        if len(theseKeys) > 0:  # at least one key was pressed
            # a response ends the routine
            continueRoutine = False
    
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in train_to_trialComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # check for quit (the Esc key)
    if endExpNow or event.getKeys(keyList=["escape"]):
        core.quit()
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "train_to_trial"-------
for thisComponent in train_to_trialComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)

# the Routine "train_to_trial" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# set up handler to look after randomisation of conditions etc
subsessions = data.TrialHandler(nReps=6, method='random', 
    extraInfo=expInfo, originPath=-1,
    trialList=[None],
    seed=None, name='subsessions')
thisExp.addLoop(subsessions)  # add the loop to the experiment
thisSubsession = subsessions.trialList[0]  # so we can initialise stimuli with some values
# abbreviate parameter names if possible (e.g. rgb = thisSubsession.rgb)
if thisSubsession != None:
    for paramName in thisSubsession:
        exec('{} = thisSubsession[paramName]'.format(paramName))

for thisSubsession in subsessions:
    currentLoop = subsessions
    # abbreviate parameter names if possible (e.g. rgb = thisSubsession.rgb)
    if thisSubsession != None:
        for paramName in thisSubsession:
            exec('{} = thisSubsession[paramName]'.format(paramName))
    
    # set up handler to look after randomisation of conditions etc
    trials = data.TrialHandler(nReps=36, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=[None],
        seed=None, name='trials')
    thisExp.addLoop(trials)  # add the loop to the experiment
    thisTrial = trials.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
    if thisTrial != None:
        for paramName in thisTrial:
            exec('{} = thisTrial[paramName]'.format(paramName))
    
    for thisTrial in trials:
        currentLoop = trials
        # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
        if thisTrial != None:
            for paramName in thisTrial:
                exec('{} = thisTrial[paramName]'.format(paramName))
        
        # ------Prepare to start Routine "trial"-------
        t = 0
        trialClock.reset()  # clock
        frameN = -1
        continueRoutine = True
        
        
        # update component parameters for each repeat
        valid = False
        
        while valid == False and trialCount < len(parameters):
            base_ndots = parameters[trialCount]['baseNum']
            front_ndots = parameters[trialCount]['frontNum'] #front layer dot number
            back_ndots = parameters[trialCount]['backNum'] #back layer dot number
            dotSize = parameters[trialCount]['dotSize'] #dot diameter
            amp = parameters[trialCount]['ampSine'] #amplitude of sine function
            halfT = parameters[trialCount]['halfCycle'] #time for half cycle
        
            w = np.pi/halfT  #angular velocity
        
            front_color = choice(['black','white'])
            if front_color == 'black':
                back_color = 'white'
            else:
                back_color = 'black'
            
            trialCount += 1
            valid = True
        
            if front_ndots > back_ndots:
                numerous = 'front'
                correctAns = 'm'
            else:
                numerous = 'back'
                correctAns = 'i'
                
            front_stim = generateDots(n_dots=front_ndots, dot_size=dotSize, overlap=overlap, color=front_color)
            back_stim = generateDots(n_dots=back_ndots, dot_size=dotSize, overlap=overlap, color=back_color)
               
            back_stim.draw()
            front_stim.draw()  
            win.flip()
              
            for i in range(halfT*4):
                frontPos = front_stim.xys
                backPos = back_stim.xys    
        
                for j in range(len(frontPos)):
                    frontPos[j][0] += amp*np.sin(w*i)       
           
                for j in range(len(backPos)):  
                    backPos[j][0] -= amp*np.sin(w*i)
                
                front_stim.xys = frontPos
                back_stim.xys = backPos
        
        
                back_stim.draw()
                front_stim.draw() 
                win.flip()
        
        
        thisExp.addData('baseNum', base_ndots)
        thisExp.addData('dotSize', dotSize)
        thisExp.addData('frontNum', front_ndots)
        thisExp.addData('backNum', back_ndots)
        thisExp.addData('numerous', numerous)
        thisExp.addData('correctAnswer', correctAns)
        
        
        trial_response = event.BuilderKeyResponse()
        # keep track of which components have finished
        trialComponents = [trial_response]
        for thisComponent in trialComponents:
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        
        # -------Start Routine "trial"-------
        while continueRoutine:
            # get current time
            t = trialClock.getTime()
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            
            # *trial_response* updates
            if t >= 0.0 and trial_response.status == NOT_STARTED:
                # keep track of start time/frame for later
                trial_response.tStart = t
                trial_response.frameNStart = frameN  # exact frame index
                trial_response.status = STARTED
                # keyboard checking is just starting
                win.callOnFlip(trial_response.clock.reset)  # t=0 on next screen flip
                event.clearEvents(eventType='keyboard')
            if trial_response.status == STARTED:
                theseKeys = event.getKeys(keyList=['i', 'm'])
                
                # check for quit:
                if "escape" in theseKeys:
                    endExpNow = True
                if len(theseKeys) > 0:  # at least one key was pressed
                    trial_response.keys = theseKeys[-1]  # just the last key pressed
                    trial_response.rt = trial_response.clock.getTime()
                    # a response ends the routine
                    continueRoutine = False
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in trialComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # check for quit (the Esc key)
            if endExpNow or event.getKeys(keyList=["escape"]):
                core.quit()
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # -------Ending Routine "trial"-------
        for thisComponent in trialComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        front_stim.setAutoDraw(False)
        back_stim.setAutoDraw(False)
        
        # check responses
        if trial_response.keys in ['', [], None]:  # No response was made
            trial_response.keys=None
        trials.addData('trial_response.keys',trial_response.keys)
        if trial_response.keys != None:  # we had a response
            trials.addData('trial_response.rt', trial_response.rt)
        # the Routine "trial" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # ------Prepare to start Routine "buffer"-------
        t = 0
        bufferClock.reset()  # clock
        frameN = -1
        continueRoutine = True
        routineTimer.add(0.500000)
        # update component parameters for each repeat
        # keep track of which components have finished
        bufferComponents = [buffer_text]
        for thisComponent in bufferComponents:
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        
        # -------Start Routine "buffer"-------
        while continueRoutine and routineTimer.getTime() > 0:
            # get current time
            t = bufferClock.getTime()
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *buffer_text* updates
            if t >= 0.0 and buffer_text.status == NOT_STARTED:
                # keep track of start time/frame for later
                buffer_text.tStart = t
                buffer_text.frameNStart = frameN  # exact frame index
                buffer_text.setAutoDraw(True)
            frameRemains = 0.0 + 0.5- win.monitorFramePeriod * 0.75  # most of one frame period left
            if buffer_text.status == STARTED and t >= frameRemains:
                buffer_text.setAutoDraw(False)
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in bufferComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # check for quit (the Esc key)
            if endExpNow or event.getKeys(keyList=["escape"]):
                core.quit()
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # -------Ending Routine "buffer"-------
        for thisComponent in bufferComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.nextEntry()
        
    # completed 36 repeats of 'trials'
    
    # get names of stimulus parameters
    if trials.trialList in ([], [None], None):
        params = []
    else:
        params = trials.trialList[0].keys()
    # save data for this loop
    trials.saveAsExcel(filename + '.xlsx', sheetName='trials',
        stimOut=params,
        dataOut=['n','all_mean','all_std', 'all_raw'])
    
    # ------Prepare to start Routine "subsessionBuffer"-------
    t = 0
    subsessionBufferClock.reset()  # clock
    frameN = -1
    continueRoutine = True
    # update component parameters for each repeat
    subsession_key_resp = event.BuilderKeyResponse()
    shuffle(parameters)
    trialCount = 0
    # keep track of which components have finished
    subsessionBufferComponents = [subsessionText, subsession_key_resp]
    for thisComponent in subsessionBufferComponents:
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    
    # -------Start Routine "subsessionBuffer"-------
    while continueRoutine:
        # get current time
        t = subsessionBufferClock.getTime()
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *subsessionText* updates
        if t >= 0.0 and subsessionText.status == NOT_STARTED:
            # keep track of start time/frame for later
            subsessionText.tStart = t
            subsessionText.frameNStart = frameN  # exact frame index
            subsessionText.setAutoDraw(True)
        
        # *subsession_key_resp* updates
        if t >= 1 and subsession_key_resp.status == NOT_STARTED:
            # keep track of start time/frame for later
            subsession_key_resp.tStart = t
            subsession_key_resp.frameNStart = frameN  # exact frame index
            subsession_key_resp.status = STARTED
            # keyboard checking is just starting
        if subsession_key_resp.status == STARTED:
            theseKeys = event.getKeys(keyList=['space'])
            
            # check for quit:
            if "escape" in theseKeys:
                endExpNow = True
            if len(theseKeys) > 0:  # at least one key was pressed
                # a response ends the routine
                continueRoutine = False
        
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in subsessionBufferComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # check for quit (the Esc key)
        if endExpNow or event.getKeys(keyList=["escape"]):
            core.quit()
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "subsessionBuffer"-------
    for thisComponent in subsessionBufferComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    
    # the Routine "subsessionBuffer" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    thisExp.nextEntry()
    
# completed 6 repeats of 'subsessions'

# get names of stimulus parameters
if subsessions.trialList in ([], [None], None):
    params = []
else:
    params = subsessions.trialList[0].keys()
# save data for this loop
subsessions.saveAsExcel(filename + '.xlsx', sheetName='subsessions',
    stimOut=params,
    dataOut=['n','all_mean','all_std', 'all_raw'])

# ------Prepare to start Routine "end"-------
t = 0
endClock.reset()  # clock
frameN = -1
continueRoutine = True
# update component parameters for each repeat
end_key = event.BuilderKeyResponse()
# keep track of which components have finished
endComponents = [end_text, end_key]
for thisComponent in endComponents:
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED

# -------Start Routine "end"-------
while continueRoutine:
    # get current time
    t = endClock.getTime()
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *end_text* updates
    if t >= 0.0 and end_text.status == NOT_STARTED:
        # keep track of start time/frame for later
        end_text.tStart = t
        end_text.frameNStart = frameN  # exact frame index
        end_text.setAutoDraw(True)
    
    # *end_key* updates
    if t >= 2 and end_key.status == NOT_STARTED:
        # keep track of start time/frame for later
        end_key.tStart = t
        end_key.frameNStart = frameN  # exact frame index
        end_key.status = STARTED
        # keyboard checking is just starting
        event.clearEvents(eventType='keyboard')
    if end_key.status == STARTED:
        theseKeys = event.getKeys()
        
        # check for quit:
        if "escape" in theseKeys:
            endExpNow = True
        if len(theseKeys) > 0:  # at least one key was pressed
            # a response ends the routine
            continueRoutine = False
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in endComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # check for quit (the Esc key)
    if endExpNow or event.getKeys(keyList=["escape"]):
        core.quit()
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "end"-------
for thisComponent in endComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
# the Routine "end" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()


# these shouldn't be strictly necessary (should auto-save)
thisExp.saveAsWideText(filename+'.csv')
thisExp.saveAsPickle(filename)
logging.flush()
# make sure everything is closed down
thisExp.abort()  # or data files will save again on exit
win.close()
core.quit()
