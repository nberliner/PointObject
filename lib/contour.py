# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 15:43:44 2015

@author: berliner
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pylab as plt

from sklearn.neighbors.kde import KernelDensity

from datetime import datetime

from utils import *

# Function definition to be used with parmap
# Allows running the function on multiple cores
def calculateKernelDensity(args):
    try:
        frame, XY, kernel, bandwidth = args # the input parameters
        
        # Get the extend of the scattered data
        xmin, xmax = np.min(XY[:,0]), np.max(XY[:,0])
        ymin, ymax = np.min(XY[:,1]), np.max(XY[:,1])
        extend = [xmin, xmax, ymin, ymax]

        # Create a grid with spacing of 10nm
        Xgrid, Ygrid = np.meshgrid(np.arange(xmin-100, xmax+100, 50), np.arange(ymin-100, ymax+100, 50))
        positions = np.vstack([Xgrid.ravel(), Ygrid.ravel()]).T # the points of the grid

        # Compute the kernel density
        kdf = KernelDensity(kernel=kernel, bandwidth=float(bandwidth), algorithm='kd_tree')
        kdf.fit(XY)

        # Evaluate the kernel on the grid
        Z = kdf.score_samples(positions)
        Z = Z.reshape(Xgrid.shape) # put the result back into the grid shape
    except:
        frame, kdf, Z, Xgrid, Ygrid, extend = None, None, None, None, None, None
    
    return [frame, (kdf, Z, Xgrid, Ygrid, extend)]
            

class Contour(IPyNotebookStyles):
    
    def __init__(self):
        
        super(Contour, self).__init__()
        
        self.data          = None
        self.kdfEstimate   = None
        self.contour       = None
        self.contourSmooth = None
    
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
                print('The smoothed contour was not calcualted yet. Was this desired?')
            return self.contourSmooth
        else:
            return self.contour
    
    def queryContourData(self):
        return self.data
    
    def calculateContour(self, kernel='gaussian', bandwidth=30.0):

        # Check if the data was already set
        if self.data is None:
            print('The data was not yet correctly set.')
            return
        
        # Set the calculation start time
        startTime = datetime.now()
        
        # Get the data
        XYdata = [ self.data[frame][1] for frame in self.data ]
    
        # Calculate the KernelDensity functions on multiple cores
        kdfEstimate = parmap( calculateKernelDensity, [ (frame, XY, kernel, bandwidth)  for frame, XY in enumerate(XYdata, start=1) ] )
        
        # Convert the result into a dict for fast lookup
        self.kdfEstimate = { key: value for key, value in kdfEstimate }
        
        # We're done with clustering, print some interesting messages
        time = datetime.now()-startTime
        print("Finished kernel density estimation in:", str(time)[:-7])
        return

    def selectContour(self, levelMax=1.5, minPathLength=80):
        # see: http://scikit-image.org/docs/dev/auto_examples/plot_contours.html
    
        if self.kdfEstimate is None:
            print('Kernel density not yet calculated. Run calculateContour() first')
            return
        
        levelMax = float(levelMax)
        self.contour = dict()
        for frame, ax in self._getFigure("Contour levels at %.1f percent" %levelMax):
            
            XY = self.data[frame][1]
            kernel, Z, Xgrid, Ygrid, extend = self.kdfEstimate[frame]
            
            levels = np.linspace((levelMax*Z.max())-0.1, levelMax*Z.max(), 2)
            CS = ax.contourf(Xgrid, Ygrid, Z, levels=levels, cmap=plt.cm.Reds, extent=extend, aspect='auto')
            ax.cla()
#            XC, YC = list(), list()
            for idx, myline in enumerate(CS.collections[-1].get_paths()):
                if len(myline) <= minPathLength: # ignore short segments
                    continue

                # Store the paths
                self.contour.setdefault(frame, list()).append(myline)

                # For plotting                
#                v = myline.vertices
#                XC.append(v[:,0])
#                YC.append(v[:,1])
                
                patch = mpl.patches.PathPatch(myline, facecolor='none', edgecolor='red', lw=2)
                ax.add_patch(patch)
            
#            ax.cla()
            ax.scatter(x=XY[:,0], y=XY[:,1])
#            for xc, yc in zip(XC, YC):
#                ax.scatter(x=xc, y=yc, color='red', s=2)
        
#            self.contour[frame] = np.vstack([xc.ravel(),yc.ravel()]).T
#            self.contour[frame] = list(zip(XC, YC))
        
        return


    def checkContour(self, contourLevels=None, levelMax=None, line=False):
        
        if contourLevels is not None and levelMax is not None:
            print('You can only specify contourLevels OR levelMax.')
            return
        
        for frame, ax in self._getFigure("Calculated contours per frame"):
            
            XY = self.data[frame][1]
            kernel, Z, Xgrid, Ygrid, extend = self.kdfEstimate[frame]
            
            if contourLevels is None and levelMax is None:
                levels = np.linspace(-1000, Z.max(), 10)
            elif contourLevels is not None:
                levels = contourLevels
            else:
                levelMax = float(levelMax)
                levels = np.linspace((levelMax*Z.max())-0.1, levelMax*Z.max(), 2)
            
            if line:
                ax.contour(Xgrid, Ygrid, Z, levels=levels, cmap=plt.cm.Reds, extent=extend, aspect='auto')
            else:
                ax.contourf(Xgrid, Ygrid, Z, levels=levels, cmap=plt.cm.Reds, extent=extend, aspect='auto')
            
            ax.scatter(x=XY[:,0], y=XY[:,1]);
        return

    def smoothContour(self, smoothWindow):
        if self.contour is None:
            print('The contour was not yet selected. Please run selectContour() first.')
            return
        
        m, n = smoothWindow
        window = np.ones((m, n))
        
        self.contourSmooth = dict()
        
        for frame, ax in self._getFigure("Smoothed contour lines"):
#            for xc, yc in self.contour[frame]:
#                contourLine = np.vstack([xc.ravel(),yc.ravel()]).T
#                contourLineSmooth = moving_average_2d(contourLine, window)
#                self.contourSmooth.setdefault(frame, list()).append(contourLineSmooth)
#            
#                ax.scatter(x=contourLine[:,0],       y=contourLine[:,1],       color='blue', s=5, alpha=0.6)
#                ax.scatter(x=contourLineSmooth[:,0], y=contourLineSmooth[:,1], color='red',  s=2)
            
            for contourPaths in self.contour[frame]:
                xc, yc = contourPaths.vertices[:,0], contourPaths.vertices[:,1]
                contourLine = np.vstack([xc.ravel(),yc.ravel()]).T
                contourLineSmooth = moving_average_2d(contourLine, window)
                
                contourLineSmoothPath = mpl.path.Path(contourLineSmooth, closed=True)
#                self.contourSmooth.setdefault(frame, list()).append(contourLineSmooth)
                self.contourSmooth.setdefault(frame, list()).append(contourLineSmoothPath)
            
                ax.scatter(x=contourLine[:,0],       y=contourLine[:,1],       color='blue', s=5, alpha=0.6)
#                ax.scatter(x=contourLineSmooth[:,0], y=contourLineSmooth[:,1], color='red',  s=2)
                patch = mpl.patches.PathPatch(contourLineSmoothPath, facecolor='none', edgecolor='red', lw=2)
                ax.add_patch(patch)

        return


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










