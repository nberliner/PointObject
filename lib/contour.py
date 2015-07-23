# -*- coding: utf-8 -*-
"""
Part of PointObject (https://github.com/nberliner/PointObject)

An IPython Notebook based anlysis tool for point-localisation
super-resolution data. 


Author(s): Niklas Berliner (niklas.berliner@gmail.com)

Copyright (C) 2015 Niklas Berliner
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pylab as plt

from sklearn.neighbors.kde import KernelDensity
#from scipy.spatial import distance
from skimage import measure

from sklearn.grid_search import GridSearchCV

from datetime import datetime
import warnings
import itertools

from morphsnakesWrapper import Morphsnake
from myMultiprocessing  import runMultiCore
from utils import *

# Function definition to be used with parmap
# Allows running the function on multiple cores
def calculateKernelDensity(args):
    try:
        frame, XY, kernel, bandwidth, positions, Xgrid, Ygrid, extend = args # the input parameters

        # Compute the kernel density
        kdf = KernelDensity(kernel=kernel, bandwidth=float(bandwidth), algorithm='kd_tree')
        kdf.fit(XY)

        # Evaluate the kernel on the grid
        Z = kdf.score_samples(positions)
        Z = Z.reshape(Xgrid.shape) # put the result back into the grid shape
#        Z = remap0to1(Z) # map array to [0,1]
    except:
        # For debugging puropses it helps to first create a NoneType error outside
        # the multiprocessing part. If an error occurs in the multiprocessing
        # the thread is not finishing and no traceback is printed (it appears
        # as if the process is still running).
        #raise
        frame, kdf, Z, Xgrid, Ygrid, extend = None, None, None, None, None, None
    
    return [frame, (kdf, Z, Xgrid, Ygrid, extend)]


def remap0to1(array):
    """ Map the array values to [0,1]. Returns the modified array."""
    maxValue = np.max(array)
    minValue = np.min(array)
    return ( array - minValue ) / np.abs(maxValue - minValue)



class Contour(IPyNotebookStyles):
    
    def __init__(self):
        
        super(Contour, self).__init__()
        
        self.data          = None
        self.kdfEstimate   = None
        self.contour       = None
        self.contourSmooth = None
        
        self.pixelSize     = None
        self.macweOptimise = None
    
    def _getFigure(self, title, nrFigs=None):
        if nrFigs is None:
            nrFigs = len(self.data)
        else:
            nrFigs = nrFigs
        return getFigure(title, nrFigs, self.doubleFigure, self.figTitleSize, self.axesLabelSize)
    
    def setData(self, data):
        self.data = data
    
    def getResult(self, smoothed=False):
        if self.contour is None:
            print('You need to first pick the contour level.')

        if smoothed:
            if self.contourSmooth is None:
                print('The smoothed contour was not calculated yet. Was this desired?')
                print('Using the unsmoothed data!')
                return self.contour
            return self.contourSmooth
        else:
            return self.contour
    
    def queryContourData(self):
        return self.data
    
    def _getGridPositions(self, XYData, delta=200.0):
        # Get the extend of the scattered data
        # delta = how much space around the structure should be added (in nm)

        ## Get the extend of the data
        xmins, xmaxs = list(), list()
        ymins, ymaxs = list(), list()
        
        for XY in XYData: # check each frame
            xmin, xmax = np.min(XY[:,0])-delta, np.max(XY[:,0])+delta
            ymin, ymax = np.min(XY[:,1])-delta, np.max(XY[:,1])+delta
        
            xmins.append(xmin)
            xmaxs.append(xmax)
            ymins.append(ymin)
            ymaxs.append(ymax)
        # Take the absolute maximum
        xmin, xmax = np.min(xmins), np.max(xmaxs)
        ymin, ymax = np.min(ymins), np.max(ymaxs)
        
        extend = [xmin, xmax, ymin, ymax]

        ## Create a grid with spacing of 5nm on which the kdf is evaluated
        nrPointsX = np.ceil( (xmax - xmin ) / 5.0 )
        nrPointsY = np.ceil( (ymax - ymin ) / 5.0 )

        xpos = np.linspace(xmin, xmax, nrPointsX, endpoint=True)
        ypos = np.linspace(ymin, ymax, nrPointsY, endpoint=True)
        
        ## Calculate the pixel size to map the information obtained from the
        ## kdf grid back to nm positions of the scatter data.
        pixelSizeX = (xmax - xmin ) / float(nrPointsX)
        pixelSizeY = (ymax - ymin ) / float(nrPointsY)
        pixelSize  = (pixelSizeX, xmin, pixelSizeY, ymin)
        
        ## The grid
        Xgrid, Ygrid = np.meshgrid(xpos, ypos)
        positions = np.vstack([Xgrid.ravel(), Ygrid.ravel()]).T # the points of the grid
        
        return positions, Xgrid, Ygrid, pixelSize, extend
    
    def kernelDensityEstimate(self, kernel='gaussian', bandwidth=None):
        """
        Calculate a kernel density estimate to obtain the contour lines
        of localisation data. Uses multiprocessing to run the estimate for each
        frame in parallel.
        """
        assert( kernel in ['gaussian', 'tophat'] )
        
        # Check if the data was already set
        if self.data is None:
            print('The data was not yet correctly set.')
            return
        
        # Set the calculation start time
        startTime = datetime.now()
        
        # Get the data
        XYData = [ self.data[frame] for frame in self.data ]
        
        positions, Xgrid, Ygrid, self.pixelSize, extend = self._getGridPositions(XYData)
        
        # Find the best parameters for kernel density estimate
        if bandwidth is None:
            bandwidth = self._optimiseBandwidth()
    
        # Calculate the KernelDensity functions and evaluate them on a grid on multiple cores
        kdfEstimate = parmap( calculateKernelDensity, [ (frame, XY, kernel, bandwidth, positions, Xgrid, Ygrid, extend)  for frame, XY in enumerate(XYData, start=1) ] )
    
        # Convert the result into a dict for fast lookup
        self.kdfEstimate = { key: value for key, value in kdfEstimate }
        
        # We're done with clustering, print some interesting messages
        time = datetime.now()-startTime
        print("Finished kernel density estimation in:", str(time)[:-7])
        return
    
    def _optimiseBandwidth(self, lower=15, upper=60, num=45, frame=1):
        """
        Run the cross-validation for determining the optimal bandwidth parameter
        for the Kernel Density Estimation.
        
        Input:
          lower (float):   Lower bound of the bandwidth parameter range that will be sampled
          
          upper (float):   Upper bound of the bandwidth parameter range that will be sampled
          
          num (int):       Number of points on the range from lower to upper that will be sampled
          
          frame (int):     The frame that will be used for bandwidth estimation
          
        """
        ## See https://jakevdp.github.io/blog/2013/12/01/kernel-density-estimation/
    
        # Set the calculation start time
        startTime = datetime.now()
        
        # Create the parameter space that should be sampled
        params = {'bandwidth': np.linspace(lower, upper, num=num, endpoint=True), \
                  'algorithm': ['kd_tree',], \
                  'kernel':    ['gaussian', ] }
        
        grid = GridSearchCV(KernelDensity(),
                            params,
                            cv=20,
                            n_jobs=-1) # 20-fold cross-validation, multi core
        
        # Select the data from frame
        XY = self.data[frame]
        grid.fit(XY)
        
        # We're done with caluclation, print some interesting messages
        time = datetime.now()-startTime
        print("Finished parameter estimation in: %s\n" %str(time)[:-7] )
        print("Using the best estimated paramters:")
        for key, value in grid.best_params_.items():
            print( "%s:\t%s" %(str(key), str(value)) )
        print("\n")
        
        if grid.best_params_['bandwidth'] == lower or grid.best_params_['bandwidth'] == upper:
            warnings.warn("Warning: The bandwidth parameter was estimated to be optimal at one sampling boundary")
            warnings.warn("Try shifting the sampling window!")
        
        return grid.best_params_['bandwidth']
    
    
    def findContourMorph(self, iterations=1500, smoothing=1, lambda1=4, lambda2=1):
        # Set the calculation start time
        startTime = datetime.now()
        
        # Set up the kernel density images which should be used
        # The sklearn package for the Kernel Density Estimation returns the log
        # of the likelyhood. We need to exponentiate the result before use.
        imgs = [ np.exp(self.kdfEstimate[frame][1]) for frame in range(1,len(self.kdfEstimate)+1) ]
        
        # Set up the Morphsnake class and run the first iterations
        self.morph = Morphsnake(imgs, smoothing, lambda1, lambda2, iterations)
        self.morph.run()
        
        # Plot the result
        self._plotMorph()
        
        # We're done with caluclation, print some interesting messages
        time = datetime.now()-startTime
        print("Finished contour fitting in:", str(time)[:-7])
        return

    def advanceContourMorph(self, iterations=500, frame=None):
        # Run some more iterations
        self.morph.advance(iterations=iterations, frame=frame)
        
        # Plot the result
        self._plotMorph()
    
    def _plotMorph(self):
        for frame, ax in self._getFigure("Morph"):
            kernel, Z, Xgrid, Ygrid, extent = self.kdfEstimate[frame]
            macwe = self.morph.macwes[frame-1]
            
            XY = self.data[frame]
            ax.scatter(x=XY[:,0], y=XY[:,1], edgecolor='None', s=10, alpha=0.8)
            ax.contour(macwe.levelset, [0.5], colors='r', extent=extent)
    
    def findFittingParameters(self, frame, smoothing, lambda1, lambda2, iteration=1000, \
                              scatter=True, s=10, alpha=0.8, xlim=False, ylim=False):
        """
        Find the best parameters for the morphological contour fitting algorithm.
        
        Running the morphological contour fitting algorithm is computationally
        expansive. In order to not have to optimise the parameters on all frames
        and to have a direct comparison of the result this function may be used.
        By specifing the desired parameters in list() objects, the full
        combination of parameters is evaluated on a single frame.
        
        Input:
            frame (int):  The frame that should be used
            
            smoothing (list):  The smoothng parameters that should evaluated.
                               Note: the values must be integers in a list!
            
            lambda1 (list):    The lambda1 parameters that should be evaluated.
                               Note: the values must be integers in a list!
            
            lambda2 (list):    The lambda2 parameters that should be evaluated.
                               Note: the values must be integers in a list!
            
            iterations (int):  The number of iterations the contour fitting
                               should run.
            
            scatter (bool):    Plot the localisation data
            
            s (int):           Size of the scattter points
            
            alpha (float):     Set the transparancy of the scatter points
                               (cf. matplotlib documentation for more details)
            
            xlim (list):       Limit the x range of the plot. Must be list of
                               length 2 with lower limit as first value and
                               upper limit as second value.
            
            ylim (list):       Limit the y range of the plot. (see also xlim)
        """
        if self.kdfEstimate is None:
            print('Kernel density not yet calculated. Run calculateContour() first')
            return
        
        # Set the calculation start time
        startTime = datetime.now()
        
        # Assemble the combination of input parameters
        assert( isinstance(frame, int) )
        if frame == 0:
            print("Please note that frames are counted starting from 0! Setting the frame to 1.")
            frame = 1
        assert( isinstance(smoothing, list) )
        assert( isinstance(lambda1,   list) )
        assert( isinstance(lambda2,   list) )
        # Thanks to: http://stackoverflow.com/a/798893
        parameters = list(itertools.product( *[smoothing, lambda1, lambda2] ))
        print("Testing %d parameter combinations" %len(parameters))
            
        # Get the kernel density estimate
        img = np.exp(self.kdfEstimate[frame][1])

        # Assemble the morph snakes and run them on multi cores
        self.macweOptimise = list()
        for s, l1, l2 in parameters:
            self.macweOptimise.append( Morphsnake([img, ], s, l1, l2) )
            
        self.macweOptimise = runMultiCore(self.macweOptimise)

        # Show the result
        self._plotOptimise(frame, scatter=scatter, s=s, alpha=alpha, xlim=xlim, ylim=ylim)        
        
        # We're done with caluclation, print some interesting messages
        time = datetime.now()-startTime
        print("Finished contour parameter screen:", str(time)[:-7])
        return
        
    
    def _plotOptimise(self, frame, scatter=True, s=10, alpha=0.8, xlim=False, ylim=False):
        for idx, ax in self._getFigure("Contour fitting parameters", len(self.macweOptimise)):
            # Set the title
            sm  = self.macweOptimise[idx-1].smoothing
            l1 = self.macweOptimise[idx-1].lambda1
            l2 = self.macweOptimise[idx-1].lambda2
            ax.set_title("smoothing: %d, lambda1: %d, lambda2: %d" %(sm, l1, l2) )
            
            
            # Get the data
            kernel, Z, Xgrid, Ygrid, extent = self.kdfEstimate[frame]
            macwe = self.macweOptimise[idx-1].macwes[0]
            img   = self.macweOptimise[idx-1].data[0]
            
            # Plot the point localisations
            if scatter:
                XY = self.data[frame]
                ax.scatter(x=XY[:,0], y=XY[:,1], facecolor='magenta', edgecolor='None', s=s, alpha=alpha)
            
            # Plot the contour line
            ax.contour(macwe.levelset, [0.5], colors='r', extent=extent)
            
            # Plot the kernel density image
            ax.imshow(img, origin='lower', extent=extent, cmap=plt.cm.Greys_r)
            
            # Set the axes limits
            if xlim:
                assert( isinstance(xlim, list) and len(xlim) == 2 )
                ax.set_xlim(xlim)
            if ylim:
                assert( isinstance(ylim, list) and len(ylim) == 2 )
                ax.set_ylim(ylim)
            
    
    def selectContour(self, level=0.5, minPathLength=100, xlim=False, ylim=False):
        """
        Input:
          level           "Probability" level of the contour of type float
          minPathLength   Filter for small islands of contour levels that can,
                          for example appear inside the object.
        Output:
          None, the contour lines are stored as (x,y) data in self.contour
         
        Description:
          Select the contour level that is used as outline. The kernel density is
          mapped to [0,1] with 1 being very high probability of having the underlying
          structure there. `level` selects the "probability" level of the outline.
          There is a balance between kernel size (i.e. the bandwidth in `calculateContour`)
          and the selected `level` value. The broader the bandwidth, i.e. the
          slower the decay of the probability, the higher the `level` needs to
          be set to achive a tight contouring.
        
          Small islands of contour lines can be filtered with `minPathLength`
        """
        if self.kdfEstimate is None:
            print('Kernel density not yet calculated. Run calculateContour() first')
            return

        self.contour = dict()
        for frame, ax in self._getFigure("Contour levels at %.1f" %level):
            # Set the axes limits
            if xlim:
                assert( isinstance(xlim, list) and len(xlim) == 2 )
                ax.set_xlim(xlim)
            if ylim:
                assert( isinstance(ylim, list) and len(ylim) == 2 )
                ax.set_ylim(ylim)
            
            XY = self.data[frame]
#            kernel, Z, Xgrid, Ygrid, extend = self.kdfEstimate[frame]
            Z = self.morph.levelset(frame)
            
            contours = self._findContour(Z, self.pixelSize, level=level, threshold=minPathLength)

            for cont in contours:
                # Store the paths
                self.contour.setdefault(frame, list()).append(cont)
                X = cont.vertices[:, 0]
                Y = cont.vertices[:, 1]
#                print('X',np.max(X),np.min(X))
#                print('Y',np.max(Y),np.min(Y))
                ax.plot(X, Y, color='red', linewidth=2)
            
            ax.scatter(x=XY[:,0], y=XY[:,1], edgecolor='None', s=1.5, alpha=0.5)


    def checkContour(self, frame, level=0.5, minPathLength=100, scatter=True, image=True, \
                     xlim=False, ylim=False, s=2, lw=2, alpha=0.7, useSmoothed=False):
        """
        Look at individual contour fits.
        
        After selecting the contour level, the contour can be examined for single
        frames in more detail. Plotting the rendered image will force the plot
        to have a "true" aspect ratio. Supressing the the rendered image will
        result in a "false" aspect ratio.
        
        
        Input:
            frame (int):  The frame to be investiagted
            
            level (float): The contour level that should be selected. Should
                           usually be 0.5
            
            minPathLength (int):  Threshold to below which contour paths will be
                                  discarded. Used to filter small islands of
                                  contour lines that do not belong to the main
                                  body.
            
            scatter (bool):  Plot the localisation data
            
            image (bool):  Plot the kernel density estimation (i.e. the
                           rendered image)
            
            s (int):           Size of the scattter points
            
            lw (int):          Linewidth of the contour line.
            
            alpha (float):     Set the transparancy of the scatter points
                               (cf. matplotlib documentation for more details)
            
            xlim (list):       Limit the x range of the plot. Must be list of
                               length 2 with lower limit as first value and
                               upper limit as second value.
            
            ylim (list):       Limit the y range of the plot. (see also xlim)
            
            useSmoothed (bool):  Used the smoothed contour lines. Most probably
                                 not needed.
        
        """
        
        if self.kdfEstimate is None:
            print('Kernel density not yet calculated. Run calculateContour() first')
            return
        
        # Create the figure
        fig = plt.figure(figsize=self.singleFigureLarge)
        fig.suptitle("Selected contour in frame %d" %frame, size=self.figTitleSize)
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')
        ax.set_xlabel("x position in nm", size=self.axesLabelSize)
        ax.set_ylabel("y position in nm", size=self.axesLabelSize)

        if s == 1: # Make sure that the size is not set to zero
            s = 2
        
        XY = self.data[frame]
        Z = self.morph.levelset(frame)
        
        if useSmoothed:
            contours = self.contourSmooth[frame]
        else:
            contours = self._findContour(Z, self.pixelSize, level=level, threshold=minPathLength)

        for cont in contours:
            X = cont.vertices[:, 0]
            Y = cont.vertices[:, 1]
            ax.plot(X, Y, color='red', linewidth=lw)
        
        if scatter:
            ax.scatter(x=XY[:,0], y=XY[:,1], facecolor='magenta', edgecolor='None', s=s, alpha=alpha)
        
        if image:
            _, img, _, _, extent = self.kdfEstimate[frame]
            ax.imshow(np.exp(img), origin='lower', extent=extent, cmap=plt.cm.Greys_r)
        
        # Set the axes limits
        if xlim:
            assert( isinstance(xlim, list) and len(xlim) == 2 )
            ax.set_xlim(xlim)
        if ylim:
            assert( isinstance(ylim, list) and len(ylim) == 2 )
            ax.set_ylim(ylim)

    def smoothContour(self, smoothWindow):
        if self.contour is None:
            print('The contour was not yet selected. Please run selectContour() first.')
            return
        
        m, n = smoothWindow
        window = np.ones((m, n))
        
        self.contourSmooth = dict()
        
        for frame, ax in self._getFigure("Smoothed contour lines"):
        
            for contourPaths in self.contour[frame]:
                xc = contourPaths.vertices[:,0]
                yc = contourPaths.vertices[:,1]
                contourLine = np.vstack([xc.ravel(),yc.ravel()]).T
                contourLineSmooth = moving_average_2d(contourLine, window)
                
                contourLineSmoothPath = mpl.path.Path(contourLineSmooth, closed=True)
                self.contourSmooth.setdefault(frame, list()).append(contourLineSmoothPath)
            
                ax.scatter(x=contourLine[:,0],       y=contourLine[:,1],       color='blue', s=5, alpha=0.6)
                patch = mpl.patches.PathPatch(contourLineSmoothPath, facecolor='none', edgecolor='red', lw=2)
                ax.add_patch(patch)

        return

    def _findContour(self, Z, pixelSize, level=0.995, threshold=100):
        # Figure out the mapping that needs to be applied to the (x,y) coordinates of the Z matrix
        mapping = self._mappingCounturFunction(pixelSize)
                
        c = measure.find_contours(Z, level)
        
        contours = list()
        for contour in c:
            if len(contour) <= threshold:
                pass
            else:
                currentContour = np.asarray([ mapping(x,y) for (x,y) in contour ])
                contours.append( mpl.path.Path(currentContour, closed=True) )
        return contours
    
    def _mappingCounturFunction(self, pixelSize):
        # Need to shift additionally by half the pixelsize to match perfectly
        # pixelSizes are inverted.. the "x" one is for "y" and vice versa
#        return lambda x,y: ( x*float(pixelSize[2])+pixelSize[3]+pixelSize[2]/2., \
#                             y*float(pixelSize[0])+pixelSize[1]+pixelSize[0]/2 )
        return lambda y,x: ( x*float(pixelSize[0])+pixelSize[1]+pixelSize[0]/2.0, \
                             y*float(pixelSize[2])+pixelSize[3]+pixelSize[2]/2.0  )

    def _distance(self, XY, XYother):
        return np.sqrt(  np.power( (XY[0] - XYother[0]),2 ) + np.power( (XY[1] - XYother[1]),2 )  )

    def _clearBranches(self, path):
        
        XYlist = list()
        d=list()
        for idx, (XY, code) in enumerate(path.iter_segments()):
            XYlist.append(XY)
            if idx == 0:
                continue
            d.append(self._distance(XY, XYlist[-2]))
            
        plt.plot(d)








#    def _generatePathFromXY(self, XY, closed=False):
#        """ Takes an array of shape (-1,2), removes duplicate points and sorts
#        them based on proximity. Returns a matplotlib Path object """
#        XYCleaned = self._cleanVertices(XY)
#        XYSorted  = self._sortVertices(XYCleaned)
#    
#        return mpl.path.Path(XYSorted, closed=closed)
#    
#    def _cleanVertices(self, XY):
#        """ Remove points in the path that are "duplicate", i.e. that are very close
#        to each other. This is redundant and can mess up selecting a segment. """
#        def removeInverts(pointList):
#            newPointList = list()
#            for point in pointList:
#                if (point[1], point[0]) in newPointList:
#                    pass
#                else:
#                    newPointList.append(point)
#            return newPointList
#        # Calcualte a distance matrix    
#        dist = distance.cdist( XY, XY, 'euclidean')
#    
#        ## Remove points that are too close
#        idx = np.where( dist <= 10.0 )
#        idx = np.concatenate( (idx[0].reshape(-1,1), idx[1].reshape(-1,1)), axis=1)
#        idx = idx[ np.invert( np.isclose(idx[:,0], idx[:,1]) ) ] # drop distance to self
#    
#        # Remove (x,y) with (y,x) pairs, otherwise everyting would be removed
#        closePoints = removeInverts( [ (x,y) for (x,y) in idx ] )
#        closePoints = [ x for (x,y) in closePoints ]
#        
#        idx = [ i for i in range(len(XY)) if i not in closePoints ]
#    
#        XYCleaned = XY[ idx ]
#        
#        return XYCleaned
#    
#    def _sortVertices(self, XY):
#        """
#        Input: np.array of shape (-1,2) with the (x,y) coordinates of the vertices.
#        The function takes an arbitrary starting point and sorts the points based
#        on proximity. No orientation is specified
#        Output: Sortet array of shape(-1,2)
#        """
#        
#        XYSorted = [[XY[0,:]],] # the first point
#        XY = np.delete(XY, [0], axis=0) # remove the first point from the list
#    
#        # Check for each point which one is the next closest in the list
#        for i in range(1,len(XY)):
#            # Calculate the distance to each other point
#            dist = distance.cdist( XYSorted[-1], XY, 'euclidean' )
#            # Get the closest point
#            closestPointIdx = np.where( dist == np.min(dist) )[1]
#            assert( len(closestPointIdx) == 1 )
#            # Add the point to the list and remove it from the list that still has to be assigned.
#            XYSorted.append( XY[closestPointIdx] )
#            XY = np.delete(XY, [closestPointIdx], axis=0)
#    
#        return np.asarray( [ (item[0][0],item[0][1]) for item in XYSorted ] )









