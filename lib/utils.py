# -*- coding: utf-8 -*-
"""
Part of PointObject (https://github.com/nberliner/PointObject)

An IPython Notebook based anlysis tool for point-localisation
super-resolution data. 


Author(s): Niklas Berliner (niklas.berliner@gmail.com)

Copyright (C) 2015 Niklas Berliner
"""
import numpy as np
import matplotlib.pylab as plt
from scipy.signal import convolve2d

import multiprocessing
import pickle


def moving_average_2d(data, window):
    # Credit goes here: http://stackoverflow.com/a/8115539/1922650
    """Moving average on two-dimensional data.
    """
    # Makes sure that the window function is normalized.
    window /= window.sum()
    # Makes sure data array is a numpy array or masked array.
    if type(data).__name__ not in ['ndarray', 'MaskedArray']:
        data = np.asarray(data)

    # The output array has the same dimensions as the input data 
    # (mode='same') and symmetrical boundary conditions are assumed
    # (boundary='symm').
    return convolve2d(data, window, mode='same', boundary='symm')


def getFigure(title, nrFigs, figSize=(16,7), figTitleSize=16, axesLabelSize=12):
    # Check how many frames have been made so see how many subfigures are needed
    # There will be two figures next to each other
    nrRows = np.ceil( float(nrFigs) / 2.0 )
    
    # calculate the necessary figure size
    figsize = (figSize[0],nrRows*figSize[1])
    
    # Create the figure
    fig = plt.figure(figsize=figsize)
    fig.suptitle(title, size=figTitleSize)
    
    for frame in range(1,nrFigs+1):
        ax = fig.add_subplot(nrRows,2,frame)
        ax.set_aspect('equal')
        ax.set_title("Frame %d" %frame)
        ax.set_xlabel("x position in nm", size=axesLabelSize)
        ax.set_ylabel("y position in nm", size=axesLabelSize)
        yield frame, ax


def savePointObject(pointObject, fname=None):
    if fname is None:
        fname = pointObject.name + '.pointObject.p'
    assert( fname is not None )
    print("Saving to %s" %fname)
    pickle.dump( pointObject, open(fname, 'wb') )

def loadPointObject(fname):
    print("Loading %s" %fname)
    return pickle.load( open(fname, "rb") )

## This implementation was taken from
## http://stackoverflow.com/a/16071616/1922650

def fun(f,q_in,q_out):
    while True:
        i,x = q_in.get()
        if i is None:
            break
        q_out.put((i,f(x)))

def parmap(f, X, nprocs = multiprocessing.cpu_count()):
    q_in   = multiprocessing.Queue(1)
    q_out  = multiprocessing.Queue()

    proc = [multiprocessing.Process(target=fun,args=(f,q_in,q_out)) for _ in range(nprocs)]
    for p in proc:
        p.daemon = True
        p.start()

    sent = [q_in.put((i,x)) for i,x in enumerate(X)]
    [q_in.put((None,None)) for _ in range(nprocs)]
    res = [q_out.get() for _ in range(len(sent))]

    [p.join() for p in proc]

    return [x for i,x in sorted(res)]

## End implementation






class movieFrameGenerator(object):
    """ Generator to create IDs based on how often the function was called """
    def __init__(self, limit, movieType):
        self.limit     = limit
        self.movieType = movieType
        
        self.ID    = 1
        self.count = 1
    
    def __call__(self, frame=None):
        if self.movieType == 'nrPoints':
            # This should be done if the movie should be generated based on the number of localisations
            if self.count >= self.limit:
                self.count = 1
                self.ID += 1
            self.count += 1
            return self.ID
        
        if self.movieType == 'nrFrame':
            # This should be done if the movie is assembles based on the number of frames
            assert( frame is not None )
            if frame >= self.count * self.limit:
                self.count += 1
                self.ID    += 1
            return self.ID



class IPyNotebookStyles(object):
    """ Style definitions that are inherited by all classes """
    
    def __init__(self):
        
        # Some plotting paramters
        self.doubleFigure      = (16,7)
        self.singleFigure      = (10,7)
        self.singleFigureLarge = (12,9)
        
        self.figTitleSize = 16
        self.axesLabelSize = 12





class Color:
    """
    Helper to assign colors to float or integer values mapped to a given range.
    """
    def __init__(self, scaleMin=None, scaleMax=None):
        self.Nglobal = dict()
        self.cmap = plt.get_cmap('seismic_r')

        self.scaleMin = scaleMin
        self.scaleMax = scaleMax
        
        if scaleMin == scaleMax and scaleMin is not None:
            print('Warning: trying to set zero scaling range!')
            self.scaleMin = scaleMin
            self.scaleMax = scaleMin * 1.1

    def __call__(self, N):
        
        if self.scaleMin is None and self.scaleMax is not None:
            c = float(N) / self.scaleMax
        elif self.scaleMin is not None and self.scaleMax is None:
            c = float(N)  - self.scaleMin
        elif self.scaleMin is None and self.scaleMax is None:
            c = float(N)
        else:
            c = (float(N) - self.scaleMin) / ( self.scaleMax - self.scaleMin )
  
        return self.cmap(c)
    
    def getColorbar(self):
        return plt.cm.ScalarMappable(cmap=plt.get_cmap('seismic_r'), norm=plt.Normalize(vmin=self.scaleMin, vmax=self.scaleMax))






















