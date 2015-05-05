# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 14:59:38 2015

@author: berliner
"""
import numpy as np
import matplotlib.pylab as plt
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter1d

import multiprocessing



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


def curvature(x, y, absolute=False):
    # Credit goes here: http://stackoverflow.com/a/28310758/1922650
    #first and second derivative
    x1 = gaussian_filter1d(x, sigma=1, order=1, mode='wrap')
    x2 = gaussian_filter1d(x1, sigma=1, order=1, mode='wrap')
    y1 = gaussian_filter1d(y, sigma=1, order=1, mode='wrap')
    y2 = gaussian_filter1d(y1, sigma=1, order=1, mode='wrap')
    if absolute:
        return np.abs(x1*y2 - y1*x2) / np.power(x1**2 + y1**2, 3./2)
    else:
        return (x1*y2 - y1*x2) / np.power(x1**2 + y1**2, 3./2)


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
        ax.set_title("Frame %d" %frame)
        ax.set_xlabel("x position in nm", size=axesLabelSize)
        ax.set_ylabel("y position in nm", size=axesLabelSize)
        yield frame, ax



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


























