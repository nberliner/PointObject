# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 13:43:35 2015

@author: berliner
"""
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pylab as plt
import pickle as pickle

from pandas   import DataFrame
from datetime import datetime
from time     import sleep
from copy     import deepcopy

from localisationsToMovie import LocalisationsToMovie
from localisationClass import rapidstormLocalisations, readXYTLocalisations
from mplWidgets        import RoiSelector
from cluster           import Cluster
from contour           import Contour
from movieMaker        import MovieGenerator
from skeleton          import Skeleton
from shape             import Shape
from curvature         import Curvature
from utils             import *




class PointObject(IPyNotebookStyles):
    """ Handles super-resolution data and allows the outline (shaoe) finding """
    def __init__(self):
        
        super(PointObject, self).__init__()
        
        self.dataFrame         = None
        self.data              = None
        self.originalDataFrame = None
        
        self.movieMade  = False
        self.runCluster = False
        
        self.ROIedges = None
        
        
        self.cluster   = None
        self.contour   = None
        self.backbone  = None
        self.curvature = None
    
    def loadFile(self, fname, dataType='rapdistorm'):
        """ Load super-resolution data. """
        if not isinstance(fname, DataFrame):
            if dataType == 'rapdistorm':
                data = rapidstormLocalisations()
            elif dataType == 'xyt':
                data = readXYTLocalisations()
            data.readFile(fname)
            self.dataFrame = data.data # we only need the DataFrame
        else:
            self.dataFrame = fname # you can also handover the DataFrame directly
        
        # Add the movieFrame column
        self.dataFrame['movieFrame'] = 1
        
        # Make a backup of the original data without ROI information
        self.originalDataFrame = deepcopy(self.dataFrame)

        # Convert the DataFrame to the Dictionary lookup
        self._convertDataFrameToDict()
    
    def save(self, folderName):
        """ Save extracted structure and contour lines as x y data. """
        # Note: not very save in the sense that it might fail if the 
        # contour was not properly calculated. It also does not allow to select
        # if the smoothed contour should be saved etc.
        if not os.path.isdir(folderName):
            print("The specified folder doe not exist")
            answer = input("Create folder %s ? (y/n)" %folderName)
            if answer.lower() == 'y' or answer.lower() == 'yes':
                os.mkdir(folderName)
            else:
                print("Not saving the data.")
                return
        
        clusterData = self.cluster.getResult()
        contourData = self.contour.getResult()
        
        # Quick sanity check
        if clusterData is None:
            print("The data seems not be clustered yet. Not saving")
            return
        if contourData is None:
            print("The contour seems not be calculated yet. Not saving")
            return
        
        for frame in range(1,len(self.data)+1):
            # Define the file names
            clusterFile = os.path.join( folderName, 'clusterData_frame_%02d.dat' %frame )
            contourFile = os.path.join( folderName, 'contourData_frame_%02d.dat' %frame )
            # Save the data
            with open(clusterFile, 'w') as f:
                f.write('x_in_nm\ty_in_nm\n')
                _, XY, _ = clusterData[frame]
                for i in range( np.shape(XY)[0] ):
                    f.write( str(XY[i,0]) + '\t' + str(XY[i,1]) + '\n' )
            
            with open(contourFile, 'w') as f:
                f.write('x_in_nm\ty_in_nm\n')
                XY = contourData[frame]
                for item in XY:
                    for i in range( np.shape(item)[0] ):    
                        f.write( str(item[i,0]) + '\t' + str(item[i,1]) + '\n' )
        
        return
    
    def clusterData(self, eps, min_samples, frame=None, clusterSizeFiler=50):
        """
        Initialisation of data clustering. After selecting the FOV, this
        step runs the first clustering step and is necessary to extract the
        object of interest from the total data points.
        """
        self.cluster = Cluster()
        self.cluster.setData(self.dataFrame)
        self.cluster.cluster(eps, min_samples, frame, clusterSizeFiler) # run DBSCAN
        self.runCluster = True
    
    def calculateContour(self, kernel='gaussian', bandwidth=30.0):
        """
        Initialise the contour calculation based on a 2D kernel density estimate.
        """
        if not self.runCluster:
            print('You need to run the clustering first!')
            return
        
        self.contour = Contour()
        self.contour.setData( self.cluster.getResult() )
        self.contour.calculateContour(kernel=kernel, bandwidth=bandwidth)
    
    def calculateCurvature(self, smooth=True, window=2):
        """
        Initialise the curvature
        """
        if self.contour is None:
            print('You need to run the contour selection first!')
            return
        
        self.curvature = Curvature()
        self.curvature.setData( self.contour.getResult() )
        self.curvature.calculateCurvature(smooth=smooth, window=window)
    
    def skeletonize(self, thres, binSize=10.0, sigma=5.0):
        """
        Initialise the backbone finding routine. The "backbone" is approximated
        by using a skeletonization algorithm. It will thus find a backbone with
        with branches.
        
        To find the skeleton, the localisations are binned in a 2D histogram,
        blurred by a gaussian and the resulting image is binarized using the
        threshold value. From the binary images pixel at the edges are taken
        away until only the skeleton remains.
        """
        if not self.runCluster:
            print('You need to run the clustering first!')
            return
        
        self.backbone = Skeleton()
        self.backbone.setData( self.cluster.getResult() )
        self.backbone.threshold(thres, binSize=binSize, sigma=sigma)
    
    def initShape(self):
        shape = Shape()
        shape.setData(self.contour.getResult(), self.backbone.getResult(), self.cluster.getResult())
        shape.show()

    def makeMovie(self, frameLength, stepSize):
        """ 
        Bin the localisations into frames. Sampling density can be controlled
        by selecting the number of frames that should be grouped.
        
        Input:
          frameLength  How long should one movie frame be, i.e. number 
                       of original TIFF frames
          stepSize     Gap between to frames, if stepSize < frameLength 
                       there will be overlap
        """
        assert( self.dataFrame is not None )
        startTime = datetime.now() # set the calculation start time
        
        movie          = LocalisationsToMovie(self.dataFrame)
        self.dataFrame = movie.create(frameLength, stepSize)
        self.movieMade = True
        
        # Make a backup of the original data without ROI information
        self.originalDataFrame = deepcopy(self.dataFrame)
        
        # We're done, print some interesting messages
        time = datetime.now()-startTime
        print("Finished making the movie in:", str(time)[:-7])
        print("Generated {:.0f} frames".format( self.dataFrame['movieFrame'].max() ))
        
        
#    def makeMovie(self, nrPoints=None, nrFrames=None):
#        """
#        Bin the localisations into frames. Sampling density can be controlled
#        by either selecting the number of frames that should be grouped or the
#        number of localisations that should be taken for each frame.
#        """
#        startTime = datetime.now() # set the calculation start time
#   
#        assert( not (nrPoints is None     and nrFrames is None) )     # either one or the other has to be specified
#        assert( not (nrPoints is not None and nrFrames is not None) ) # only one can be specified
#        
#        if nrPoints is not None and nrFrames is None: # movie frames are based on the localisation number
#            gen = movieFrameGenerator(nrPoints, 'nrPoints')
#            movieFrames = [ gen() for point in self.dataFrame['x'] ]
#        elif nrFrames is not None and nrPoints is None: # movie frames are based on frames
#            gen = movieFrameGenerator(nrFrames, 'nrFrame')
#            movieFrames = [ gen(frame) for frame in self.dataFrame['frame'] ]
#        
#        # Add the movieFrame column
#        self.dataFrame['movieFrame'] = movieFrames
#        self.movieMade = True # We're done here
#        
#        # We're done, print some interesting messages
#        time = datetime.now()-startTime
#        print("Finished making the movie in:", str(time)[:-7])
#        print("Generated {:.0f} frames".format( self.dataFrame['movieFrame'].max() ))
#        
#        # Make a copy of the data that will not be touched
#        self.originalDataFrame = deepcopy(self.dataFrame)

    def movieFrames(self):
        """ Iterate over the localisation data in the movie frames. """
        assert( self.movieMade )
        for name, group in self.data.groupby('movieFrame'):
            yield name, group

    def getFrame(self, frame):
        """ Return the localisations in the given frame. """
        return self.data.groupby('movieFrame').get_group(frame)
    
    def saveMovie(self, fname, plotContour=True, sigma=10.0):
        """
        Create an .mp4 file showing the PointObject.
        """
        if self.data is None or not self.movieMade:
            print('You need to load data and make a movie first')
            return
        
        if self.runCluster:
            movieData = self.cluster.getResult()
        else:
            movieData = self.data
        
        if plotContour and self.contour is not None:
            movieContour = self.contour.getResult()

        m = MovieGenerator(movieData, movieContour, sigma)
        m.make(fname)
    
    def setFOV(self, frame=1, convert=True):
        """ Show the initial frame to set a FOV wich should be used. """
        # Check if switched to qt mode
        if not mpl.get_backend() == 'Qt4Agg':
            print('Switch to the Qt backend first by executing "%pylab qt"')
            return False
            
        # Reset any prior ROI selection
        self.dataFrame = self.originalDataFrame
        
        # Select first frame
        frameData = self.dataFrame[ self.dataFrame['movieFrame'] == frame ]
        XY = np.asarray( frameData[['x','y']] )
        
        fig = plt.figure(figsize=self.singleFigure)
        ax  = fig.add_subplot(111)
        ax.set_title("Frame %d" %frame)
        ax.set_xlabel("x position in nm", size=self.axesLabelSize)
        ax.set_ylabel("y position in nm", size=self.axesLabelSize)
        
        ax.set_xlim( [ np.min(XY[:,0]), np.max(XY[:,0]) ] )
        ax.set_ylim( [ np.min(XY[:,1]), np.max(XY[:,1]) ] )
        
        ax.scatter(x=XY[:,0], y=XY[:,1], edgecolor='none', facecolor='blue', s=2)
        
        rect = RoiSelector(ax)
        self.ROIedges = rect.edges # These will always be x1,y1,x2,y2 with x1,y2 bottom left, x2,y2 top right
        # The ROI was drawn, let the user look at it for a second and then go on
        sleep(1)
        plt.close(fig)
        
        # Select the ROI
        self._selectFOVdata()
        if convert:
            self._convertDataFrameToDict()

    def _selectFOVdata(self):
        if self.ROIedges is None:
            print('No ROI selected yet. Run setROI() first')
            return
        
        # Filter the data
        xbl, ybl, xtr, ytr = self.ROIedges
        self.dataFrame = self.dataFrame[ self.dataFrame.x >= xbl ]
        self.dataFrame = self.dataFrame[ self.dataFrame.x <= xtr ]
        self.dataFrame = self.dataFrame[ self.dataFrame.y >= ybl ]
        self.dataFrame = self.dataFrame[ self.dataFrame.y <= ytr ]
    
    def _convertDataFrameToDict(self):
        self.data = dict()
        if self.movieMade:
            for movieFrame in set(self.dataFrame.movieFrame):
                XY = np.asarray( self.dataFrame[ self.dataFrame.movieFrame == movieFrame ][['x','y']] )
                self.data[movieFrame] = [movieFrame, XY, None ]
        else:
            XY = np.asarray( self.dataFrame[['x','y']] )
            self.data[1] = [1, XY, None ]




#class mitochondria(object):
#    
#    def __init__(self, fname):
#        
#        if not isinstance(fname, DataFrame):
#            data = rapidstormLocalisations()
#            data.readFile(fname)
#            data = data.data # we only need the DataFrame
#        else:
#            data = fname # you can also handover the DataFrame directly
#        
#        assert( isinstance(data, DataFrame) )
#            
#        self.data = data # should be a DataFrame
#        self.originalData = None # will be populated after making the movie
#        self.mito = dict()
#        self.mitoList = list() # list of cluster per frame that are the mitochondria
#        self.mitoROI = dict()
#        self.clustering = None
#        self.ROIs = dict()
#
#        self.ROIedges = None
#        self.contour = None
#        self.contourSmooth = None
#        self.kdfEstimate = None
#
#        
#
#        
#        self.clusterSizeFiler = 200.
#    
#    def saveMito(self, fname):
#        if fname[-2:] != '.p':
#            fname = fname + '.p'
#        
#        args = self.data, self.originalData, self.mito, self.mitoList, \
#               self.mitoROI, self.clustering, self.ROIs, self.ROIedges, \
#               self.contour, self.contourSmooth, self.kdfEstimate, \
#               self.movieMade
#        
#        pickle.dump( args, open( fname, "wb" ) )
#    
#    def loadMito(self, fname):
#        
#        args = pickle.load( open( fname, "rb" ) )
#        
#        self.data, self.originalData, self.mito, self.mitoList, \
#           self.mitoROI, self.clustering, self.ROIs, self.ROIedges, \
#           self.contour, self.contourSmooth, self.kdfEstimate, \
#           self.movieMade = args
#    
#    def saveData(self, fname):
#        pass

