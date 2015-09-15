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
import tifffile as Tiff

from pandas   import DataFrame
from datetime import datetime
from time     import sleep
from copy     import deepcopy

from localisationsToMovie import LocalisationsToMovie
from localisationClass import rapidstormLocalisations, XYTLocalisations
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
        self.images            = None
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
        """
        Load a localisation file into PointObject.
        
        Currently supported input formats are rapidstorm and xyt. For xyt data
        has to be tab separated file with one header line containing 'x, 'y', 'frame'
        as labels for the columns. Additional columns may be present which will
        be ignored.
        
        Input:
          frame (str):   File that should be loaded into PointObject
          
          dataType (str):  Data type of the input file. Possible values are
                           'rapidstorm' and 'xyt'.
        
        """
        if not isinstance(fname, DataFrame):
            self.name = fname # set the name of the object
            if dataType == 'rapdistorm':
                data = rapidstormLocalisations()
            elif dataType == 'xyt':
                data = XYTLocalisations()
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
    
    def resetImage(self):
        """
        Reset the contour finding routine.
        """
        self.contour = None
        return
    
    def calculateKDE(self, kernel='gaussian', bandwidth=None):
        """
        Calculate the Kernel Density Estimation.
        
        This will generate a high-resolution "image" of the localisations
        based on the kernel density estimate of the point localisations.
        
        If no bandwidth is specified an optimal bandwidth parameter is estimated
        using cross-validation. This is the default behaviour.
        
        Input:
            kernel (str):  Kernel to be used for the kernel density estimation
                           Possible values are 'gaussian' (default) and 'tophat'
            
            bandwidth (float,None):  The bandwidth for the kernel density estimation
                                     If set to None cross-validation will be used
                                     to find the optimal parameter.
                                     
        """
        if not self.runCluster:
            print('You need to run the clustering first!')
            return
        if self.contour is not None:
            print("It seems as if you already set up a pixel image.")
            print("Use resetImage() and try again")
            return
        
        self.contour = Contour()
        self.contour.setData( self.cluster.getResult(self.edgePoints) )
        self.contour.kernelDensityEstimate(kernel=kernel, bandwidth=bandwidth)
    
    def loadPixelImage(self, fname, pixelSize=1.0, start=None, end=None, padding=[20,20]):
        """
        Load a pixel image into the pipeline.
        
        Instead of generating a super-resolved image from point localisation
        data the pipeline can load any pixel image instead. The consecutive
        steps of contour fitting and curvature calculation can then be used
        on this input.
        
        Input:
          fname (str):       Image file to load
          
          pixelSize (float): Set the pixel size in nm
          
          start (int):       The first frame to load
          
          end (int):         The last frame to include
          
          padding (list):    Must be a list of two int specifying the width
                             in pixels of the dark region that will be added
                             to the images at the outside edges. This is done
                             to allow the constricting contour fitting to fully
                             close on objects even if they are extending out
                             of the frame.
          
        """
        if self.contour is not None:
            print("It seems as if you already set up a pixel image.")
            print("Use resetImage() and try again")
            return
            
        # Load the image file using tifffile.py
        tmpImage    = Tiff.TiffFile(fname)
        self.images = [ tmpImage[i].asarray() for i in range(len(tmpImage)) ]
        self.images = self.RGBtoGreyscale(self.images)
        
        # Select the frame range
        if start is None:
            start = 1
        if end is None:
            end = len(self.images)
        
        self.images = self.images[start-1:end]
        
        # Add some dark pixel paddings around the image. This makes contour
        # detection easier if the object is extending out of the image border.
        if padding:
            xpad, ypad = padding
            newImages = list()
            for image in self.images:
                # Create arrays containing zeros
                zerosX = np.zeros( [np.shape(image)[0], xpad] )
                zerosY = np.zeros( [ypad, np.shape(image)[1]+2*xpad] )
                
                # Add them to the original image
                newImage = np.concatenate( [zerosX, image, zerosX], axis=1 )
                newImage = np.concatenate( [zerosY, newImage, zerosY], axis=0 )
                
                # Add the padded image to the list
                newImages.append(newImage)
            
            # Replace the original images with the padded ones.
            self.images = newImages

        # Initialise the Contour class and set the images
        self.contour = Contour()
        self.contour.images = self.images
        self.contour.pixelSize = [pixelSize, pixelSize, pixelSize, pixelSize]

        print("Finished loading %d frames from %s" %(len(self.images), fname))
        return
    
    def RGBtoGreyscale(self, images):
        """
        Convert RGB images to greyscale
        
        The SIM images seem to be returned as RGB TIFF images containing three
        channels containing each the same information. The pipeline expects
        greyscale images. If images of the described are detected they are
        converted to simple greyscale images.
        
        Input:
          images (list):  List containing np.arrays holding the image data
          
        """
        assert( isinstance(images, list) )
        if len( np.shape(images[0]) ) == 4: # i.e. RGB
            # Assert that all color channels are actually the same
            assert( np.all( images[0][:,:,:,0] == images[0][:,:,:,1] ) )
            assert( np.all( images[0][:,:,:,0] == images[0][:,:,:,2] ) )
            images = [ item[:,:,:,0] for item in images ]
        
        if len(images) == 1:
            images = [ images[0][frame,:,:] for frame in range(np.shape(images[0])[0]) ]
        
        return images
        
    def calculateContour(self, iterations=1500, smoothing=2, lambda1=1, \
                         lambda2=1, startPoints="min"):
        """
        Find the contour of a super-resolved pixel image.
        
        The contour is fitted using a morphological contour fitting algorithm
        (https://github.com/pmneila/morphsnakes).
        
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
            iterations (int):  Number of steps the morphological contour fitting
                               algorithm should advance.
            
            smoothing (scalar):  See above

            lambda1 (scalar):    See above

            lambda2 (scalar):    See above
            
            startPoints (list or str) This determines the seed point for the
                                      contour fitting algorithm, i.e. the starting
                                      point. Can be either "max" or "min" and
                                      one correspdoning pixel is choses based
                                      of the condition. If individual frames
                                      require different starting points this
                                      can be specified using a list. Each element
                                      in the list will then be taken for the
                                      corresponding frame. E.g. ["max","max,"min"]
                                      would take the "max" for frame 1 and 2 and
                                      "min" for frame 3.
        """
        if self.contour is None:
            print("The image is not yet set yet.")
            return

        self.contour.findContourMorph(iterations=iterations ,\
                                      smoothing=smoothing   ,\
                                      lambda1=lambda1       ,\
                                      lambda2=lambda2       ,\
                                      startPoints=startPoints
                                      )
        self.contour.selectContour()
    
    def calculateCurvature(self, smooth=True, window=2, smoothedContour=False, isclosed=True, percentiles=[99,1]):
        """
        Calculate the curvature based on the expression for local curvature
        ( see https://en.wikipedia.org/wiki/Curvature#Local_expressions )
        
        Input:
          smooth (bool):  Smooth the curvature data using the gaussian weighted
                          rolling window approach.
            
          window (float):  Sigma of the gaussian (The three sigma range of the
                           gaussian will be used for the averaging with each 
                           localisation weighted according to the value of 
                           the gaussian).
        
          smoothedContour (boolean): Use the smoothed contour for the calculation?
                                     Default is False.
        
          isclosed (boolean): Treat the contour as closed path. Default is True
                              and should always be the case if padding was added
                              to the image when loading image files or when the
                              FOV was sufficiently large when loading point
                              localisation data. Note that there might be unexpected
                              if the contour is not closed.
        
          percentiles (list): Must be a list of two floats. Specifies the max and
                              min values for displaying the curvature on a color
                              scale. This can be important if unnatural kinks are
                              observed which would dominate the curvature to an
                              extent that the color scale would be skewed.
                              By setting the percentiles one can set the range
                              in a "smart" way.
        
        """
        if self.contour is None:
            print('You need to run the contour selection first!')
            return
        
        self.curvature = Curvature()
        self.curvature.setData( self.contour.getResult(smoothed=smoothedContour) )
        self.curvature.calculateCurvature(smooth=smooth, window=window, isclosed=isclosed, percentiles=percentiles)
    
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
    
    def saveMovie(self, fname, plotContour=True, plotCurvature=True, lw=10, alpha=0.8):
        """
        Create an .mp4 file showing the progression of the PointObject.
        
        Requires the ffmpeg binaries to work. It will create a movie file showing
        the frame-by-frame result as a movie.
        
        Input:
          fname (str):           The filename of the movie
          
          plotContour (bool):    Plot the contour
          
          plotCurvature (bool):  Color code the contour line based on the local
                                 curvature.
          
          lw (int):              Width of the contour line
          
          alpha (float):         Transparency of the contour line
          
        """
        if self.data is None or not self.movieMade:
            print('You need to load data and make a movie first')
            return
        
        if self.contour is None:
            print('You need to run the KDE estimate first and generate the "image"')
            return
        
        try:
            movieData = self.contour.kdfEstimate
            assert( movieData is not None )
        except:
            print("It looks as if something went wrong with getting the KDE image.")
            print("Did you run the kernel density estimation already?")
            return

        try:
            movieContour = self.contour.getResult()
        except:
            movieContour = None
        try:
            movieCurvature = self.curvature.getResult()
        except:
            movieCurvature = None
        
        if not plotCurvature:
            movieCurvature = None
        
        m = MovieGenerator(movieData, movieContour, movieCurvature, plotContour=plotContour, lw=lw, alpha=alpha)
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

