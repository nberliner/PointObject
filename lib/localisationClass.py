# -*- coding: utf-8 -*-
"""
Part of PointObject (https://github.com/nberliner/PointObject)

An IPython Notebook based anlysis tool for point-localisation
super-resolution data. 


Author(s): Niklas Berliner (niklas.berliner@gmail.com)

Copyright (C) 2015 Niklas Berliner
"""
from copy import deepcopy

from readLocalisations  import *

class localisations():
    """
    Container for point localisation super-resolution data.
    
    This is a striped down version of the one used in the high-throughput project
    
    It needs to be inherited by classes that add localisation file specifics.
    
    """
    
    def __init__(self):
        """
        Input:

          data (DataFrame):         Pandas DataFrame object containing the localisations.
                                    The column names represent which values are stored,
                                    the row names contain the frame number. Since a column
                                    with the frame number is required the row name is
                                    not normally needed.
        
          dataFiltered (DataFrame): Pandas DataFrame. Like data but containing
                                    the filtered data.
        
        
          filtered (bool):   Flag indicating the a filter was applied on the data
          
          pixelSize (float): The pixel size in nm

        """
        
        self.frameLimit   = False
        self.filtered     = False
        self.data         = None # pandas DataFrame
        self.dataFiltered = None # pandas DataFrame
        
        self.linkedLocalisations = False
        self.grouped             = False
        self.groupedData         = None
        self.groupedDataFiltered = None
        
        self.fiducialsSearchedFor = False
        self.fiducialsDetected    = False
        self.fiducials            = None
        
        self.driftCalculated             = False
        self.drift                       = None
        self.driftCorrectedData          = None
        self.driftCorrectedDataFiltered  = None
        self.driftCorrectedDataUngrouped = None
        self.driftCorrectedDataUngroupedFiltered = None
        
        self.gapLength = 0
    
    def localisations(self, dataType=None, dataFilter=True):
        """
        Return the point localisations in a pandas DataFrame.
        
        If nothing is specified the most "advanced" localisations are returned.
        This means, if the data was filtered the filtered data will be returned,
        if the data was grouped the grouped data will be returned, etc.
        This is accumulative meaning that the "optimal" localisations are
        returned by default.
        
        Input:
          dataType (str):    Controls which localisations are returned.Possible
                             values are: 'original', 'grouped', 'driftCorrected',
                             'driftCorrectedUngrouped', None
        
          dataFilter (bool): Return filtered data (if available) or unfiltered
        """
        doFilter = self.filtered and dataFilter
        if dataType == None:
            if doFilter:
                if self.driftCalculated and self.fiducialsDetected:
                    return self.driftCorrectedDataFiltered
                elif self.grouped:
                    return self.groupedDataFiltered
                else:
                    return self.dataFiltered
            else:
                if self.driftCalculated and self.fiducialsDetected:
                    return self.driftCorrectedData
                elif self.grouped:
                    return self.groupedData
                else:
                    return self.data
        
        elif dataType == 'original':
            if doFilter:
                return self.dataFiltered
            else:
                return self.data
        elif dataType == 'grouped':
            if doFilter:
                return self.groupedDataFiltered
            else:
                return self.groupedData
        elif dataType == 'driftCorrected':
            if doFilter:
                return self.driftCorrectedDataFiltered
            else:
                return self.driftCorrectedData
        elif dataType == 'driftCorrectedUngrouped':
            if doFilter:
                return self.driftCorrectedDataUngroupedFiltered
            else:
                return self.driftCorrectedDataUngrouped
        else: # we should never reach this point
            print('Warning: DataType not understood!')
    
    def queryLocalisations(self, dataType=None, dataFilter=True):
        """ 
        Query the localisation type that is returned by localisations()
        
        Use for bug fixing. Since the class is self aware and should be smart
        about which localisations (i.e. grouped, filtered, etc.) are returned
        the returned type can be queried for verification.
        See the docstring of localisations() for a description of the input
        parameters.
        """
        doFilter = self.filtered and dataFilter
        if dataType == None:
            if doFilter:
                if self.driftCalculated and self.fiducialsDetected:
                    return 'self.driftCorrectedDataFiltered'
                elif self.grouped:
                    return 'self.groupedDataFiltered'
                else:
                    return 'self.dataFiltered'
            else:
                if self.driftCalculated and self.fiducialsDetected:
                    return 'self.driftCorrectedData'
                elif self.grouped:
                    return 'self.groupedData'
                else:
                    return 'self.data'
        
        elif dataType == 'original':
            if doFilter:
                return 'self.dataFiltered'
            else:
                return 'self.data'
        elif dataType == 'grouped':
            if doFilter:
                return 'self.groupedDataFiltered'
            else:
                return 'self.groupedData'
        elif dataType == 'driftCorrected':
            if doFilter:
                return 'self.driftCorrectedDataFiltered'
            else:
                return 'self.driftCorrectedData'
        elif dataType == 'driftCorrectedUngrouped':
            if doFilter:
                return 'self.driftCorrectedDataUngroupedFiltered'
            else:
                return 'self.driftCorrectedDataUngrouped'
        else: # we should never reach this point
            return 'Warning: DataType not understood!'
    
    def readFile(self, fname): # implement in child class to adapt to input format
        pass
        
    def numberOfLocalisations(self, dataType=None):
        return len(self.localisations(dataType=dataType))
    
    def localisationsPerFrame(self, dataType=None):
        data = self.localisations(dataType=dataType)
        # Get the frames and the number of localisations per frame
        frames          = np.asarray(data['frame'], dtype=np.int64)
        perFrames       = np.asarray(data.groupby('frame').count().index, dtype=np.int64)
        nrLocalisations = np.asarray(data.groupby('frame').count().x.values)
        return frames, (nrLocalisations, perFrames)

    def _getXYT(self, data):
        """ Get the x,y,t data from the DataFrame """
        x = np.array(data['x'])
        y = np.array(data['y'])
        t = np.array(data['frame'])
        return x, y, t
 
    def _overwriteDataWithFiltered(self):
        """ If prefiltering is used, overwrite the unfiltered data variables """
        self.data                        = self.dataFiltered
        self.groupedData                 = self.groupedDataFiltered
        self.driftCorrectedData          = self.driftCorrectedDataFiltered
        self.driftCorrectedDataUngrouped = self.driftCorrectedDataUngroupedFiltered
    
    def filterAll(self, filterValues, relative=False):
        """
        Filter the data based on the criteria in filterValues.
        """
        assert( isinstance(filterValues, dict) )
        self.filterLocalisations() # This resets the filter
        for dataType in filterValues: # This applies the new filters
            minValue, maxValue = filterValues[dataType]
            self.filterLocalisations(minValue, maxValue, dataType, relative)

    def filterLocalisations(self, minValue=None, maxValue=None, dataType=None, \
                            relative=True, overwrite=False):
        """ 
        Filter the localisations.
        
        Apply a filter to the localisation data. Data can be filtered by any
        information that is present in the localisations DataFrame as column.
        The allowed dataType will thus depend on the input type of the localisations.
        
        To reset the filter and continue working with the original data call
        the function with no argument (or minValue, maxValue, and dataType all
        set to None).
        
        Input:
          minValue (float):  Lower bound of the filter. If None, minus infinity will be used.
          
          maxValue (float):  Upper bound of the filter If None, plus infinity will be used.
          
          dataType (str):    Sorry this is misleading, espacially since it has
                             nothing to do with the dataType parameter from
                             the localisations() function. This specifies the column
                             name that should be used for filtering. E.g. it could
                             be "Photon Count" for rapdistorm or maybe llr_threshold
                             for the sCMOS code.
        
          relative (bool):   Do relative filtering, i.e. the minValue and maxValue
                             will be interpreted as percentage values. The filtering
                             will be done relative to the maximum value. The
                             percentage range can be between [0,1] or [1,100]
        
          overwrite (bool):  Replace the internal storage of the data with the
                             filtered data. If True the original data will be
                             fogotten and the filtered data will be taken as new
                             "original" input data. I do not remember right now
                             why exactly I added this option but I think I needed
                             to "reset" the memory of the class at some point..
                             
        """
        if minValue==None and maxValue==None and dataType==None: #reset filter
            self.filtered = False
            return       
        
        # Set the minimum filter value
        if minValue == None:
            minValue = - np.inf
        else:
            if relative:
                if minValue > 1.0: # it was given as e.g. 20%
                    minValue /= 100.0
                minValue = self.data[dataType].max() * minValue
            else:
                minValue = minValue

        # Set the maximum filter value
        if maxValue == None:
            maxValue = np.inf
        else:
            if relative:
                if maxValue > 1.0: #it was given as e.g. 20%
                    maxValue /= 100.0
                maxValue = self.data[dataType].max() * maxValue
            else:
                maxValue = maxValue

        ## The following is a bit ugly. I am keeping track of the applied
        ## filter/modifactions to compare the individual steps.
        if self.filtered: # apply additional filter
            d   = deepcopy(self.dataFiltered)
        else: # filter from the original data
            d   = deepcopy(self.data)
        self.dataFiltered                   = deepcopy(d[   (  d[dataType] >= minValue) & (  d[dataType] <= maxValue) ])
        
        if self.grouped:
            if self.filtered: # apply additional filter
                gd  = deepcopy(self.groupedDataFiltered)
            else:
                gd  = deepcopy(self.groupedData)
            self.groupedDataFiltered        = deepcopy(gd[  ( gd[dataType] >= minValue) & ( gd[dataType] <= maxValue) ])

        
        if self.driftCalculated and self.fiducialsDetected:
            if self.filtered: # apply additional filter
                dgd = deepcopy(self.driftCorrectedDataFiltered)
                dd  = deepcopy(self.driftCorrectedDataUngroupedFiltered)
            else:
                dgd = deepcopy(self.driftCorrectedData)
                dd  = deepcopy(self.driftCorrectedDataUngrouped)
            self.driftCorrectedDataFiltered = deepcopy(dgd[ (dgd[dataType] >= minValue) & (dgd[dataType] <= maxValue) ])
            self.driftCorrectedDataUngroupedFiltered = deepcopy(dd[ (dd[dataType] >= minValue) & (dd[dataType] <= maxValue) ])
        
        
        if overwrite:
            self._overwriteDataWithFiltered()
            
        self.filtered = True # set the filtered flag
        return

    def writeToFile(self, fname, dataType=None, pixelSize=1.0):
        if dataType == None:
            data = self.localisations()
        elif dataType == 'original':
            data = self.localisations('original')
        elif dataType == 'grouped':
            data = self.localisations('grouped')
        elif dataType == 'driftCorrected':
            data = self.localisations('driftCorrected')
        elif dataType == 'driftCorrectedUngrouped':
            data = self.localisations('driftCorrectedUngrouped')
        elif dataType == 'fiducials':
            data = self.fiducials

        try:
            data[['x','y']] = data[['x','y']] * float(pixelSize)
#            data.to_csv(fname, sep='\t', columns=['x','y','frame'], index=False)
            data.to_csv(fname, sep='\t', index=False)
        except:
            print('Sorry, could not write the data to disk!')


class rapidstormLocalisations(localisations):
    
    def __init__(self):
        localisations.__init__(self)

    def readFile(self, fname, photonConversion=1.0, pixelSize=1.0):
        self.data = readRapidStormLocalisations(fname, photonConversion, pixelSize)
            
    def frame(self, frame):
        assert( isinstance(frame, int) )
        return self.data.ix['frame_'+str(frame)]

    def allPoints(self):
        for point in self.data.iterrows():
            yield point



class XYTLocalisations(localisations):
    """
    
    Load generic localisation files
    
    """
    def __init__(self):
        localisations.__init__(self)

    def readFile(self, fname, pixelSize=1.0):
        """
        The first row is used as header information, the following columns must
        be present: 'x', 'y', and 'frame' (note: this is case sensitive!)
        """
        self.data = readXYTLocalisations(fname, pixelSize=pixelSize)











