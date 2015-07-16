# -*- coding: utf-8 -*-
"""
Part of PointObject (https://github.com/nberliner/PointObject)

An IPython Notebook based anlysis tool for point-localisation
super-resolution data. 


Author(s): Niklas Berliner (niklas.berliner@gmail.com)

Copyright (C) 2015 Niklas Berliner
"""
from myMultiprocessing import runMultiCore
import morphsnakes
from utils import *

from operator import itemgetter

def circle_levelset(shape, center, sqradius, scalerow=1.0):
    """Build a binary function with a circle as the 0.5-levelset."""
    grid = np.mgrid[list(map(slice, shape))].T - center
    phi = sqradius - np.sqrt(np.sum((grid.T)**2, 0))
    u = np.float_(phi > 0)
    return u

class Morphsnake(object):
    
    def __init__(self, data, smoothing=1, lambda1=1, lambda2=1):
        
        self.data       = data
        self.smoothing  = smoothing
        self.lambda1    = lambda1
        self.lambda2    = lambda2
        self.iterations = 1500
        
        self.macwes = None
    
    def __call__(self):
        self.run()
        return self
    
    def run(self):
        # Select the starting point for each image
        self._autolevelset()
        
        # Set the interations
        for macwe in self.macwes:
            macwe.iterations = self.iterations
        
        # Multi-core or not?
        if len(self.macwes) > 1:
            # Run the algorithm and sort the result after it is done.
            self.macwes = runMultiCore(self.macwes)
            self.macwes = self._sort(self.macwes)
        else:
            macwe = self.macwes[0]
            macwe() # run the algorithm
            self.macwes = [macwe, ]
    
    def _sort(self, macwes):
        tmp = [ (item.frame, item) for item in macwes ]
        tmp = sorted(tmp, key=itemgetter(0))
        return [ item[1] for item in tmp ]
    
    def _autolevelset(self):
        self.macwes = list()
        for frame, img in enumerate(self.data):
            # Set up the image
            macwe = morphsnakes.MorphACWE(img, smoothing=self.smoothing ,\
                                               lambda1=self.lambda1     ,\
                                               lambda2=self.lambda2
                                         )
            macwe.frame = frame
            
            x,y = np.where(img == np.min(img))
            macwe.levelset = circle_levelset(img.shape, (x[0], y[0]), 10)
            
            self.macwes.append(macwe)
    
    def advance(self, iterations=500, frame=None):
        if frame is None:
            for macwe in self.macwes:
                macwe.iterations = iterations
                self.macwes = runMultiCore(self.macwes)
                self.macwes = self._sort(self.macwes)
        else:        
            self.macwes[frame].run(iterations)
    
    def levelset(self, frame):
        return self.macwes[frame-1].levelset