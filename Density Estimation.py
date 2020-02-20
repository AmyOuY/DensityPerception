"""
Part I: Density Estimation Experiment
This experiment was created using PsychoPy2 Experiment Builder (v1.90.3),
    on July 30, 2019, at 18:18
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
expName = 'density'  # from the Builder filename that created this script
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
    originPath=u'C:\\Users\\ouyim\\Desktop\\Yimiao Ou Project Report\\partI_densityExperiment.psyexp',
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
    monitor='testMonitor', color=[0.1,0.1,0.1], colorSpace='rgb',
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
    text="You will be presented two patches of dots.\n\nThe patches will stay on the screen for 250 milliseconds.\n\nYour task is to choose which of the left or right patch appears denser.\n\nPress 'left' for left patch and 'right' for right patch.\n\n(Press 'right' to continue.)",
    font='Arial',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1,
    depth=0.0);
    
    
# Set up stimuli conditions    
def addParameters(testNums, refRadius, testRadius):
    ans = []
    for num in testNums:
        pattern = {}
        pattern['refRadius'] = refRadius
        pattern['num'] = num
        pattern['testRadius'] = testRadius
        ans.append(pattern)
    return ans
    


r2_t2 = [64, 81, 102, 128, 162, 203, 256]
r2_t28 = [125, 158, 198, 251, 316, 399, 502]
r2_t4 = [256, 324, 408, 512, 648, 812, 1024]

r28_t2 = [33, 41, 51, 66, 82, 103, 130]
r28_t28 = [64, 81, 102, 128, 162, 203, 256]
r28_t4 = [132, 164, 204, 264, 328, 412, 520]

r4_t2 = [16, 20, 25, 32, 40, 51, 64]
r4_t28 = [32, 40, 50, 63, 79, 100, 126]
r4_t4 = [64, 81, 102, 128, 162, 203, 256]

parameters = []

parameters.extend(addParameters(r2_t2, 2, 2))
parameters.extend(addParameters(r2_t28, 2, 2.8))
parameters.extend(addParameters(r2_t4, 2, 4))
parameters.extend(addParameters(r28_t2, 2.8, 2))
parameters.extend(addParameters(r28_t28, 2.8, 2.8))
parameters.extend(addParameters(r28_t4, 2.8, 4))
parameters.extend(addParameters(r4_t2, 4, 2))
parameters.extend(addParameters(r4_t28, 4, 2.8))
parameters.extend(addParameters(r4_t4, 4, 4))


shuffle(parameters)


# Initialize components for Routine "trial"
trialClock = core.Clock()
import psychopy.visual as visual
import psychopy.event as event
from random import random, choice
from numpy import sqrt
from math import pi, cos, sin

# Generate Gaussian blobs pattern
dotSize = 0.15
trialCount = 0

def generateDots(win=win, radius=2, nDots=128, offset=-5):
    count = 0
    dots = []
    dots_1 = []
    dots_2 = []

    while count < nDots:
        valid = True
        a = random() * 2 * pi
        r = radius * sqrt(random())
        dotX = r * cos(a) + offset
        dotY = r * sin(a)
        for pos in dots:
            X = (pos[0] - dotX)**2
            Y = (pos[1] - dotY)**2
            dist = sqrt(X+Y)
            if dist < dotSize:
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
    
    
polygon1 = visual.Rect(
    win=win, name='polygon1',
    width=(1, 1)[0], height=(1, 1)[1],
    ori=0, pos=(0, 0),
    lineWidth=1, lineColor=[0.1,0.1,0.1], lineColorSpace='rgb',
    fillColor=[0.1,0.1,0.1], fillColorSpace='rgb',
    opacity=1, depth=-2.0, interpolate=True)


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
    depth=-1.0);


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
        theseKeys = event.getKeys(keyList=['right'])
        
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


# set up handler to look after randomisation of conditions etc
subsessions = data.TrialHandler(nReps=5, method='random', 
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
    trials = data.TrialHandler(nReps=63, method='random', 
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
        nDots = 128
        valid = False
        
        while valid == False and trialCount < len(parameters):
            refOffset = choice([-5, 5])
            refRadius = parameters[trialCount]['refRadius']
            refStim1, refStim2 = generateDots(radius=refRadius, nDots=128, offset=refOffset)
            
            testRadius = parameters[trialCount]['testRadius']
            testDots = parameters[trialCount]['num']
            testStim1, testStim2 = generateDots(radius=testRadius, nDots=testDots, offset=-refOffset)
            
            trialCount += 1
            valid = True
        
            if testDots/(pi*testRadius**2) >= 128/(pi*refRadius**2):
                denser = 'test'
                answer = -refOffset
            else:
                denser = 'ref'
                answer = refOffset
        
            if answer == -5:
                correctKey = 'left'
            elif answer == 5:
                correctKey = 'right'
        
        for i in range(18):
            refStim1.draw() 
            refStim2.draw()
            testStim1.draw() 
            testStim2.draw()
           
            win.flip()
        
       
        thisExp.addData('refRadius', refRadius)
        thisExp.addData('refDotNum', 128)
        thisExp.addData('testRadius', testRadius)
        thisExp.addData('testDotNum', testDots)
        thisExp.addData('testCenter', -refOffset)
        thisExp.addData('denser', denser)
        thisExp.addData('answer', correctKey)
        
        train_key1 = event.BuilderKeyResponse()
        # keep track of which components have finished
        trialComponents = [train_key1, polygon1]
        for thisComponent in trialComponents:
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        
        # -------Start Routine "trial"-------
        while continueRoutine:
            # get current time
            t = trialClock.getTime()
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            
            # *train_key1* updates
            if t >= 0.0 and train_key1.status == NOT_STARTED:
                # keep track of start time/frame for later
                train_key1.tStart = t
                train_key1.frameNStart = frameN  # exact frame index
                train_key1.status = STARTED
                # keyboard checking is just starting
                win.callOnFlip(train_key1.clock.reset)  # t=0 on next screen flip
                event.clearEvents(eventType='keyboard')
            if train_key1.status == STARTED:
                theseKeys = event.getKeys(keyList=['left', 'right'])
                
                # check for quit:
                if "escape" in theseKeys:
                    endExpNow = True
                if len(theseKeys) > 0:  # at least one key was pressed
                    train_key1.keys = theseKeys[-1]  # just the last key pressed
                    train_key1.rt = train_key1.clock.getTime()
                    # a response ends the routine
                    continueRoutine = False
            
            # *polygon1* updates
            if t >= 3 and polygon1.status == NOT_STARTED:
                # keep track of start time/frame for later
                polygon1.tStart = t
                polygon1.frameNStart = frameN  # exact frame index
                polygon1.setAutoDraw(True)
            
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
        refStim1.setAutoDraw(False) 
        refStim2.setAutoDraw(False)
        testStim1.setAutoDraw(False) 
        testStim2.setAutoDraw(False)
        # check responses
        if train_key1.keys in ['', [], None]:  # No response was made
            train_key1.keys=None
        trials.addData('train_key1.keys',train_key1.keys)
        if train_key1.keys != None:  # we had a response
            trials.addData('train_key1.rt', train_key1.rt)
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
            if t >= 0 and buffer_text.status == NOT_STARTED:
                # keep track of start time/frame for later
                buffer_text.tStart = t
                buffer_text.frameNStart = frameN  # exact frame index
                buffer_text.setAutoDraw(True)
            frameRemains = 0 + 0.5- win.monitorFramePeriod * 0.75  # most of one frame period left
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
        
    # completed 63 repeats of 'trials'
    
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
    
# completed 5 repeats of 'subsessions'

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
#filename2 = filename+'test.xlsx'
#stairs.saveAsExcel(filename2)
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
