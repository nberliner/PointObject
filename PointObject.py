# -*- coding: utf-8 -*-
"""
PointObject (https://github.com/nberliner/PointObject)

An IPython Notebook based anlysis tool for point-localisation
super-resolution data. 


Author(s): Niklas Berliner (niklas.berliner@gmail.com)

Copyright (C) 2015 Niklas Berliner
"""
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pylab as plt
import warnings

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
        
        self.name = None
        
        self.dataFrame         = None
        self.data              = None
        self.originalDataFrame = None
        
        self.movieMade  = False
        self.runCluster = False
        self.edgePoints = True
        
        self.ROIedges = None
        
        
        self.cluster   = None
        self.contour   = None
        self.backbone  = None
        self.curvature = None
    
    def loadFile(self, fname, dataType='rapdistorm'):
        """ Load super-resolution data. """
        if not isinstance(fname, DataFrame):
            self.name = fname # set the name of the object
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
    
    def save(self, folderName=None):
        warnings.warn("save() is deprecated. Please use export() instead.", DeprecationWarning)
        self.export(folderName)
        
    def export(self, folderName=None):
        """
        Save the extracted structure (i.e. clusters), contour lines, and
        curvature values to a text file.
        
        Export the caluculated data to a folder. For each step of the calculation
        a single file will be created containing the frame number and the relevant
        values. See the header for more information.
        The output files will be named based on the input data file name if no
        folderName is given. The input file name will be appended by 
        "_clusterData.dat", "_contourData.dat", and "_curvatureData.dat" respectively.
        
        Input:
            folderName (str, None):  Export data into this folder (will be created if
                                     it does not exist. Existing files will be overwritten!)
        
        """
        # Use the input folder if not specified otherwise
        if folderName is None:
            folderName, _ = os.path.split(self.name)
        
        # Check if the folder exists, and if not ask if it should be created
        if not os.path.isdir(folderName):
            print("The specified folder doe not exist")
            answer = input("Create folder %s ? (y/n)" %folderName)
            if answer.lower() == 'y' or answer.lower() == 'yes':
                os.mkdir(folderName)
            else:
                print("Not saving the data.")
                return
        
        # Get the results
        try:
            clusterData = self.cluster.getResult()
        except AttributeError:
            clusterData = None
            print("Clustering data not present. Not saving.")
        try:
            contourData = self.contour.getResult()
        except AttributeError:
            contourData = None
            print("Contour data not present. Not saving.")
        try:
            curvatureData         = self.curvature.getResult()
            curvatureDataSelected = self.curvature.contourSelected
        except AttributeError:
            curvatureData = None
            print("Curvature data not present. Not saving.")

        # Get the name of the object
        _, objectName = os.path.split(self.name)
        # Define open the file hooks
        if clusterData is not None:
            clusterFile   = open(os.path.join( folderName, '%s_clusterData.dat' %objectName ), 'w')
            clusterFile.write('x_in_nm\ty_in_nm\tframe\n')
        if contourData is not None:
            contourFile   = open(os.path.join( folderName, '%s_contourData.dat' %objectName ), 'w')
            contourFile.write('x_in_nm\ty_in_nm\tframe\n')
        if curvatureData is not None:
            curvatureFile = open(os.path.join( folderName, '%s_curvatureData.dat' %objectName ), 'w')
            curvatureFileSelected = open(os.path.join( folderName, '%s_curvatureDataSelected.dat' %objectName ), 'w')
            curvatureFile.write('x_in_nm\ty_in_nm\tcurvature_in_(1/nm)\tframe\n')
            curvatureFileSelected.write('x_in_nm\ty_in_nm\tcurvature_in_(1/nm)\tframe\n')
        
        # Save the data
        for frame in range(1,len(self.data)+1):
            # Save the cluster data
            if clusterData is not None:
                XY = clusterData[frame]
                for row in range( np.shape(XY)[0] ):
                    clusterFile.write("%.3f\t%.3f\t%d\n" %(XY[row,0], XY[row,1], frame) )
            
            # Save the contour data
            if contourData is not None:
                contourPaths = contourData[frame]
                for path in contourPaths:
                    XY = path.vertices
                    for row in range( np.shape(XY)[0] ):    
                        contourFile.write("%.3f\t%.3f\t%d\n" %(XY[row,0], XY[row,1], frame) )
            
            # Save the curvature data
            if curvatureData is not None:
                # Write the full curvature data
                for contourPath, curvature in curvatureData[frame]:
                    XY = contourPath.vertices
                    for row, value in enumerate(curvature):
                        curvatureFile.write("%.3f\t%.3f\t%.6f\t%d\n" %(XY[row,0], XY[row,1], value, frame) )
                
                # Write the selected region only
                for _, (contourPath, curvature, _, _, _) in curvatureDataSelected[frame]:
                    XY = contourPath.vertices
                    for row, value in enumerate(curvature):
                        curvatureFileSelected.write("%.3f\t%.3f\t%.6f\t%d\n" %(XY[row,0], XY[row,1], value, frame) )
        
        print("Saving data to %s done." %folderName)
        return
    
    def clusterData(self, eps, min_samples, frame=None, clusterSizeFiler=50, askUser=True):
        """
        Run DBSCAN to identify the objects of interest and supress noise.
        
        
        The density based clustering algorithm DBSCAN is used to cluster the
        point localisations within each movie frame. The user is then asked to
        select the clusters that correspond to the object of interest (multiple
        selections are allowed) and the selection will be kept and used for
        future calculations.
        
        Each movie frame will be run on one CPU core thus making use of multi-core
        systems. The user can restrict the frames that should be computed to
        optimse the clustering parameters.
        
        Input:
            eps (float):       The eps parameter of the sklearn DBSCAN implementation
            
            min_samples (int): The min_samples parameter of the sklearn DBSCAN implementation
                               For more details on the implementation see: http://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html
            
            frame (int,None):  If not None specifies the frame that should used
            
            clusterSizeFiler (int): Display filter for small clusters. CLusters
                                    below the specified size will not be displayed
                                    
            askUser (bool):    Whether or not prompt the user to select the clusters
        
        
        """
        self.cluster = Cluster() # initialise the cluster object
        self.cluster.setData(self.dataFrame)
        self.cluster.cluster(eps, min_samples, frame, clusterSizeFiler, askUser) # run DBSCAN
        self.runCluster = True # set the cluster flag so that subsequent calls now it was run
    
    def calculateContour(self, kernel='gaussian', bandwidth=None, iterations=1500, 
                         smoothing=2, lambda1=1, lambda2=1, kde=True, morph=True):
        """
        Contour finding based on a 2D kernel density estimate.and contour fitting.

        
        Find the contour of the selected point localisations. This is done in two
        steps. The first step is to generate a high-resolution "image" of the
        localisations. The second step is to find the contour via a morphological
        contour fitting algorithm (https://github.com/pmneila/morphsnakes).

        If no bandwidth is specified an optimal bandwidth parameter is estimated
        using cross-validation. This is the default behaviour.
        
        The contour fitting is controlled using three parameters, i.e. smoothing,
        lamda1, and lambda2. Here the description given in the original source code
        smoothing : scalar
            The number of repetitions of the smoothing step (the
            curv operator) in each iteration. In other terms,
            this is the strength of the smoothing. This is the
            parameter µ.
        lambda1, lambda2 : scalars
            Relative importance of the inside pixels (lambda1)
            against the outside pixels (lambda2).
        Furthermore the parameter iterations is used to select the number of
        iterations the contour fitting should run. If it did not converge yet,
        further steps might be excecuted by calling contour.advanceContourMorph()
        
        
        Input:
            kernel (str):  Kernel to be used for the kernel density estimation
                           Possible values are 'gaussian' (default) and 'tophat'
            
            bandwidth (float,None):  The bandwidth for the kernel density estimation
                                     If set to None cross-validation will be used
                                     to find the optimal parameter.
            
            iterations (int):  Number of steps the morphological contour fitting
                               algorithm should advance.
            
            smoothing (scalar):  See above

            lambda1 (scalar):    See above

            lambda2 (scalar):    See above
            
            kde (bool):  Run Kernel Density Estimation. Set to False if
                         calculateContour() has already been run and only the
                         contour fitting should be repeated.
            
            morph (bool):  Run the morphological contour fitting
            
        """
        if not self.runCluster:
            print('You need to run the clustering first!')
            return
        
        if kde:
            self.contour = Contour()
            self.contour.setData( self.cluster.getResult(self.edgePoints) )
            self.contour.kernelDensityEstimate(kernel=kernel, bandwidth=bandwidth)
        if morph:
            self.contour.findContourMorph(iterations=iterations ,\
                                          smoothing=smoothing   ,\
                                          lambda1=lambda1       ,\
                                          lambda2=lambda2
                                          )
    
    def calculateCurvature(self, smooth=True, window=2, smoothedContour=False):
        """
        Initialise the curvature
        """
        if self.contour is None:
            print('You need to run the contour selection first!')
            return
        
        self.curvature = Curvature()
        self.curvature.setData( self.contour.getResult(smoothed=smoothedContour) )
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

    def makeMovie(self, nrFrames=2000, stepSize=500):
        """ 
        Bin the localisations into frames. Sampling density can be controlled
        by selecting the number of frames that should be grouped.
        
        Imaging live objects implies that the object moves and changes during
        acquisiton time. By selecting a time gap (i.e. nrFrames) in which the
        object can be assumed to be static, a snapshot at this "time point" can
        be generated and a super-resolution image can be created. To increase
        the temporal "resolution" stepSize can be specified which is the gap
        between the first frames of consecutive movie frames. If the stepSize
        is smaller than nrFrames there will consequently an overlap of data
        between consecutive points. This will however, effectivly increase the
        frames/sec of the movie that will be produced and will increase the
        chance of resolving the event of interest.
        
        Here a brief schematic of the movie frame generation:
        
            -------------------------------------------- (frames)
            | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |   |    (movie frames without overlap)
              |   |   |   |   |   |   |   |   |   |   |  (additional movie frames due to overlap)
        
        Input:
          nrFrames     How long should one movie frame be, i.e. number 
                       of original TIFF frames
          stepSize     Gap between to frames, if stepSize < nrFrames 
                       there will be overlap
        """
        assert( self.dataFrame is not None )
        startTime = datetime.now() # set the calculation start time
        
        movie          = LocalisationsToMovie(self.dataFrame)
        self.dataFrame = movie.create(nrFrames, stepSize)
        self.movieMade = True
        
        # Make a backup of the original data without ROI information
        self.originalDataFrame = deepcopy(self.dataFrame)
        
        # We're done, print some interesting messages
        time = datetime.now()-startTime
        print("Finished making the movie in:", str(time)[:-7])
        print("Generated {:.0f} frames".format( self.dataFrame['movieFrame'].max() ))
        


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
        """
        Depreceated. Please use setROI() instead!
        
        Only kept for backwards compatibility
        """
        warnings.warn("Depreceated. Please use setROI() instead!", DeprecationWarning)
        self.setROI(frame=frame, convert=convert)
        
    def setROI(self, frame=1, convert=True):
        """
        Use one frame to set a ROI for further analysis.
        
        Note: Cannot be used with %pylab inline
        
        Input:
            frame   (int):   Frame number that will be used to select the ROI
            convert (bool):  Do set the selected ROI as new data
        
        """
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

