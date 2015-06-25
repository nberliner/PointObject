# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 15:43:44 2015

@author: berliner
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

from morphsnakesWrapper import Morphsnake
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
        
        self.pixelSize = None
    
    def _getFigure(self, title):
        nrFigs = len(self.data)
        return getFigure(title, nrFigs, self.doubleFigure, self.figTitleSize, self.axesLabelSize)
    
    def setData(self, data):
        self.data = data
    
    def getResult(self, smoothed=True):
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
        XYData = [ self.data[frame][1] for frame in self.data ]
        
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
    
    def _optimiseBandwidth(self, lower=15, upper=40, num=25, frame=1):
        ## See https://jakevdp.github.io/blog/2013/12/01/kernel-density-estimation/
        
        # Create the parameter space that should be sampled
        params = {'bandwidth': np.linspace(lower, upper, num=num, endpoint=True), \
                  'algorithm': ['kd_tree',], \
                  'kernel':    ['gaussian', ] }
        
        grid = GridSearchCV(KernelDensity(),
                            params,
                            cv=20,
                            n_jobs=-1) # 20-fold cross-validation, multi core
        
        # Select the data from frame
        XY = self.data[frame][1]
        grid.fit(XY)
        print("Using the estimated paramters:")
        print(grid.best_params_)
        
        if grid.best_params_['bandwidth'] == lower or grid.best_params_['bandwidth'] == upper:
            warnings.warn("Warning: see bandwidth parameter was estimated to be optimal at one sample boundary")
            warnings.warn("Try shifting the sample window!")
        
        return grid.best_params_['bandwidth']
    
    
    def findContourMorph(self, iterations=1500):
        # Set the calculation start time
        startTime = datetime.now()
        
        # Set up the kernel density images which should be used
        # The sklearn package for the Kernel Density Estimation returns the log
        # of the likelyhood. We need to exponentiate the result before use.
        imgs = [ np.exp(self.kdfEstimate[frame][1]) for frame in range(1,len(self.kdfEstimate)+1) ]
        
        # Set up the Morphsnake class and run the first iterations
        self.morph = Morphsnake(imgs)
        self.morph.run(iterations)
        
        # We're done with caluclation, print some interesting messages
        time = datetime.now()-startTime
        print("Finished contour finding in:", str(time)[:-7])
        return
        
        # Plot the result
        self._plotMorph()
    
    def advanceContourMorph(self, iterations=500, frame=None):
        # Run some more iterations
        self.morph.advance(iterations=iterations, frame=frame)
        
        # Plot the result
        self._plotMorph()
    
    def _plotMorph(self):
        for frame, ax in self._getFigure("Morph"):
            kernel, Z, Xgrid, Ygrid, extent = self.kdfEstimate[frame]
            macwe = self.morph.macwes[frame-1]
            
            XY = self.data[frame][1]
            ax.scatter(x=XY[:,0], y=XY[:,1])
            ax.contour(macwe.levelset, [0.5], colors='r', extent=extent)
    
    
    def selectContour(self, level=0.5, minPathLength=100):
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
        
#        if level >= 1.0:
#            print('The selected level is too high. The density is mappd to [0,1].')
#            return

        self.contour = dict()
        for frame, ax in self._getFigure("Contour levels at %.1f" %level):
            
            XY = self.data[frame][1]
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


#    def checkContour(self, contourLevels=None, levelMax=None, line=False):
#        
#        if contourLevels is not None and levelMax is not None:
#            print('You can only specify contourLevels OR levelMax.')
#            return
#        
#        for frame, ax in self._getFigure("Calculated contours per frame"):
#            
#            XY = self.data[frame][1]
#            kernel, Z, Xgrid, Ygrid, extend = self.kdfEstimate[frame]
#            
#            if contourLevels is None and levelMax is None:
#                levels = np.linspace(0.95, 0.99, 5)
#            elif contourLevels is not None:
#                levels = contourLevels
#            else:
#                levelMax = float(levelMax)
#                levels = np.linspace((levelMax*Z.max())-0.1, levelMax*Z.max(), 2)
#            
#            if line:
#                ax.contour(Xgrid, Ygrid, Z, levels=levels, cmap=plt.cm.Reds, extent=extend, aspect='auto')
#            else:
#                ax.contourf(Xgrid, Ygrid, Z, levels=levels, cmap=plt.cm.Reds, extent=extend, aspect='auto')
#            
#            ax.scatter(x=XY[:,0], y=XY[:,1]);
#        return

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









