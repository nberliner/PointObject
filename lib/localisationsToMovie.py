# -*- coding: utf-8 -*-
"""
Part of PointObject (https://github.com/nberliner/PointObject)

An IPython Notebook based anlysis tool for point-localisation
super-resolution data. 


Author(s): Niklas Berliner (niklas.berliner@gmail.com)

Copyright (C) 2015 Niklas Berliner
"""


import numpy as np
from pandas   import DataFrame


class LocalisationsToMovie(object):
    """
    Group super-resolution localisation data into 'movie frames'.
    """
    
    def __init__(self, data, verbose=False):
        """ Input must be a pandas DataFrame """
        assert( isinstance(data,DataFrame) ) # the input must be a Pandas DataFrame
        
        if 'movieFrame' in data.columns: # check if a movie was already made
            if verbose:
                print("The data appears to already have a movie created. It will be dropped!")
            del data['movieFrame']
        
        self.data = data

    
    def create(self, frameLength, stepSize):
        """
        Group the localisations for form "movie frames".
        
        Input:
          frameLength  How long should one movie frame be, i.e. number 
                       of original TIFF frames
          stepSize     Gap between to frames, if stepSize < frameLength 
                       there will be overlap
        
        Output:
          pandas DataFrame with a new column movieFrame.
          Note that if an overlap is given the overlapping localisations will
          appear for each respective entry in movieFrame.
        """
        nrFrames = int(np.max( self.data.frame ))
        newData = list()
        for movieFrame, frames in self._createFrameTuple(nrFrames, frameLength, stepSize).items():
            ## Get the data of the current frame
            frameData = self.data[ np.in1d(self.data.frame, frames) ].values
            
            ## Add the movieFrame column
            currentMovieFrame      = np.zeros( (np.shape(frameData)[0],1) )
            currentMovieFrame[:,:] = movieFrame # set all values to new movie frame
            
            frameData = np.concatenate((frameData,currentMovieFrame), axis=1)
            newData.append( frameData )

        ## Combine all data into one array
        newDataArray = np.concatenate(newData, axis=0)
        
        ## Create the new column header information
        newColumns = list(self.data.columns)
        newColumns.append('movieFrame')
        
        df = DataFrame(newDataArray, columns=newColumns)
        df.frame      = np.asarray(df.frame.values, dtype=np.int)
        df.movieFrame = np.asarray(df.movieFrame.values, dtype=np.int)
        
        return df
    
    def _createFrameTuple(self, nrFrames, frameLength, stepSize):
        
        frames = dict()
        
        for frame, step in enumerate(range(0,nrFrames,stepSize), start=1):
            if step+frameLength > nrFrames: # don't allow a badly sampled frame at the end
                break
            for i in range(frameLength):
#                if step+i > nrFrames:
#                    break
                frames.setdefault(frame, list()).append(step+i)
        
        return frames




    
    
    
    
    