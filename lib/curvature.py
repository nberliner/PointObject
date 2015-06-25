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
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import distance

from time import sleep

from utils import *
from mplWidgets import CurvatureSelector
from HTMLtable import HTMLtable


def curvature(x, y, sigma=2, absolute=False):
    # Credit goes here: http://stackoverflow.com/a/28310758/1922650
    #first and second derivative
    x1 = gaussian_filter1d(x,  sigma=sigma, order=1, mode='wrap')
    x2 = gaussian_filter1d(x1, sigma=sigma, order=1, mode='wrap')
    y1 = gaussian_filter1d(y,  sigma=sigma, order=1, mode='wrap')
    y2 = gaussian_filter1d(y1, sigma=sigma, order=1, mode='wrap')
    if absolute:
        return np.abs(x1*y2 - y1*x2) / np.power(x1**2 + y1**2, 3./2)
    else:
        return (x1*y2 - y1*x2) / np.power(x1**2 + y1**2, 3./2)


def averageCurvature(C, width, weight=None):
    # Check which weighting scheme should be applied
    if weight == None or weight.lower() == 'none':
        gfunc = lambda x, center: 1
    elif weight == 'gaussian':
        gfunc = lambda x, center: 1.0 * np.exp( - np.power(x-center,2) / (2.0*np.power(width,2)) )
        width = 3 * width # cover the three sigma range
    else:
        print('Weighting factor not understood')
        return C
    
    # Calculate the rolling window
    delta = int(np.ceil(width))
    Cavg = np.zeros(len(C))
    for idx in range(len(C)):
        for i in range(idx-delta, idx+delta+1):
            if i >= len(C): # go back to start
                i -= len(C)
            Cavg[idx] += ( C[i] * gfunc(i, idx) )
    Cavg = np.asarray(Cavg)
    return Cavg


class Curvature(IPyNotebookStyles):
    
    def __init__(self):
        
        super(Curvature, self).__init__()
        
        self.data = None
        self.dataCurvature = None
        
        # helpers to keep the curvature color consitent
        self.curvatureMax = None
        self.curvatureMin = None
        self.color        = None
    
    def setData(self, data):
        assert( isinstance( data, dict ) )
        self.data = data
    
    def getResult(self):
        pass
    
    def calculateCurvature(self, smooth=True, window=2):
        self.dataCurvature = dict()
        for frame, ax in self._getFigure("Curvature calculation"):
            # Calculate the curvature
            for contour in self.data[frame]:
                XY     = contour.vertices
                xc, yc = XY[:,0], XY[:,1]
                Corig  = curvature(xc, yc)
                Cavg   = averageCurvature(Corig, window, 'gaussian') # Aply an average based on a gaussian
                
                if smooth:
                    C = Cavg
                else:
                    C = Corig
                    
                self.curvatureMax   = np.max(C)
                self.curvatureMin   = np.min(C)
                self.color          = Color(scaleMin=self.curvatureMin, scaleMax=self.curvatureMax)

                self.dataCurvature.setdefault(frame, list()).append( (contour, C) )
                
                # PLot the curvature color coded
                cColor = [ self.color(i) for i in C ]
                ax.scatter(x=xc, y=yc, c=cColor, alpha=1, edgecolor='none', s=10, cmap=plt.cm.seismic_r)
    
    def selectCurvature(self):

        # Check if switched to qt mode
        if not mpl.get_backend() == 'Qt4Agg':
            print('Switch to the Qt backend first by executing "%pylab qt"')
            return False
        
        self.contourSelected = dict()
        for frame in self.data.keys():
            (x1, y1), (x2, y2), (x3, y3), (x4, y4) = self._select(frame)
            self.contourSelected.setdefault(frame, list()).append( (1, self._getLineLength(frame, x1, y1, x2, y2 )) )
            self.contourSelected.setdefault(frame, list()).append( (2, self._getLineLength(frame, x3, y3, x4, y4 )) )
    
    def showSelected(self):
        for frame, ax in self._getFigure("Curvature selection"):
            for index, (contour, C, (x1, y1, x2, y2), pc1, pc2) in self.contourSelected[frame]:
                XY = contour.vertices
                (xc1, yc1) = pc1
                (xc2, yc2) = pc2
                cColor = [ self.color(i) for i in C ]
                ax.scatter(x=XY[:,0], y=XY[:,1], c=cColor, alpha=0.8, edgecolor='none', s=10, cmap=plt.cm.seismic_r)
                
                self._placeText(index, ax, pc1, pc2)
                
                ax.scatter(x=x1, y=y1, c='red', alpha=0.8, edgecolor='none', s=30, cmap=plt.cm.seismic_r)
                ax.scatter(x=x2, y=y2, c='red', alpha=0.8, edgecolor='none', s=30, cmap=plt.cm.seismic_r)
                ax.scatter(x=xc1, y=yc1, facecolor='none', alpha=0.8, edgecolor='blue', s=30, cmap=plt.cm.seismic_r)
                ax.scatter(x=xc2, y=yc2, facecolor='none', alpha=0.8, edgecolor='blue', s=30, cmap=plt.cm.seismic_r)
    
    
    def table(self):
        rows    = list(self.contourSelected.keys())
        columns = ['Frame', '1', '2', 'Sum']
        header  = "95 percentile of the curvature segments"
        
        htmlTable = HTMLtable()
        
        return htmlTable(rows, columns,header) % self._returnTableValues()
        
    
    def _returnTableValues(self):
        result = list()
        lookup = dict()
        frames = list()
        for frame, values in self.contourSelected.items():
            frames.append(frame)
            
            for value in values:
                index     = value[0]
                # Calculate the curvature value for the selected segment
                # We need to consider that we actually want to take the low
                # percentile if the overall curvature is negative.
                percent = 95
                if np.sum(value[1][1]) > 0:
                    curvature = np.percentile(value[1][1], percent)
                else:
                    curvature = np.percentile(value[1][1], 100-percent)
                lookup[ (frame, index) ] = curvature
        
        for (frame, index) in [ (f,i) for f in frames for i in range(1,4) ]:
            try:
                result.append( lookup[ (frame, index) ] )
            except KeyError:
                result.append( lookup[(frame, 1)] + lookup[(frame, 2)] )
        
        return tuple(result)
    
    
    def plotCurvature(self):
        
        title = "Curvature progression"
        figs = getFigure(title, 2, self.doubleFigure, self.figTitleSize, self.axesLabelSize)
        
        values = self._returnTableValues()
        dataSideOne = [ values[i  ] for i in range(0,len(values),3)]
        dataSideTwo = [ values[i+1] for i in range(0,len(values),3)]
        curv = [dataSideOne, dataSideTwo]
        X    = [ i for i in range(1,len(dataSideOne)+1) ]
        
        for frame, ax in figs:
            # Configure the subfigure
            ax.set_aspect('auto')
            ax.set_title("Side %d" % frame)
            ax.axhline(y=0,c="black",linewidth=0.5,zorder=0)
            
            # Plot the data
            ax.plot(X,curv[frame-1])
        
        return
        
    def _placeText(self, text, ax, pc1, pc2):
        (xc1, yc1) = pc1
        (xc2, yc2) = pc2
        
        x_center = (xc1 + xc2) / 2.0
        y_center = (yc1 + yc2) / 2.0
        
        ax.text(x=x_center, y=y_center, s=str(text), size=14, color='black', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.3'))
    
    def _select(self, frame):

        # Create the figure
        fig = plt.figure(figsize=self.singleFigure)
        ax  = fig.add_subplot(111)
        ax.set_title("Frame %d" %frame)
        ax.set_xlabel("x position in nm", size=self.axesLabelSize)
        ax.set_ylabel("y position in nm", size=self.axesLabelSize)
        ax.set_aspect('equal')
        
        # Get the data
        for contour, C in self.dataCurvature[frame]:
            XY = contour.vertices
            cColor = [ self.color(i) for i in C ]
            ax.scatter(x=XY[:,0], y=XY[:,1], c=cColor, alpha=0.8, edgecolor='none', s=10, cmap=plt.cm.seismic_r)
        
        cman = CurvatureSelector(ax)
        
        # The ROI was drawn, let the user look at it for a second and then go on
        sleep(1)
        plt.close(fig)
        
        return cman.points
    
    def _getLineLength(self, frame, x1, y1, x2, y2):
        count = 0
        # Get the contour
        for contour, C in self.dataCurvature[frame]:
            
            if self._pointOnContour( (x1, y1), contour ):
                count += 1 # for sanity check
                # Get the points on the contour
                pc1, pc2 = self._closestPoints(contour, (x1, y1), (x2, y2) )
                # Get the segment and its indexes
                segment, idx = self._extractPointsOnSegment(contour, pc1, pc2)
                # Select the curvature of the segment
                curvature = np.asarray(C)[idx]
        
        assert( count == 1 )
        return segment, curvature, (x1, y1, x2, y2), pc1, pc2
    
    def _pointOnContour(self, point, contour):
        XY   = contour.vertices
        x, y = point
        
        dist = distance.cdist( np.asarray( [[x,y]] ), XY, 'euclidean')
        # Get the closest point
        minDist         = np.min(dist)
        closestPointIdx = np.where( dist == minDist )[1]
        
        assert( len(closestPointIdx) == 1 ) # Otherwise there are multiple closest points

        if minDist >= 100.0: # must be the other contour line
            return False
        else:
            return True
    
    def _closestPoints(self, contour, point1, point2):
        XY     = contour.vertices
        x1, y1 = point1
        x2, y2 = point2
        
        # Get the start point
        dist = distance.cdist( np.asarray( [[x1,y1]] ), XY, 'euclidean')
        minDist         = np.min(dist)
        closestPointIdx = np.where( dist == minDist )[1]

        # Get the end point
        distEnd = distance.cdist( np.asarray( [[x2,y2]] ), XY, 'euclidean')
        minDistEnd         = np.min(distEnd)
        closestPointEndIdx = np.where( distEnd == minDistEnd )[1]
        
        assert( len(closestPointIdx)    == 1 ) # Otherwise there are multiple closest points
        assert( len(closestPointEndIdx) == 1 ) # Otherwise there are multiple closest points
        
        closestPoint1 = XY[closestPointIdx][0]
        closestPoint2 = XY[closestPointEndIdx][0]
        
        return closestPoint1, closestPoint2
        
    
    def _extractPointsOnSegment(self, path, point1, point2):
        """ Select the segment between point1 and point 2. For a closed path
        there are two segments ( p1 -> p2 and p2 -> p1 ) and the shorter one
        will be returned """
        def isclose(p1, p2):
            return np.allclose(p1, p2)
            
        START    = False
        END      = False
        ENDafter = False

        segment1    = list()
        segment1Idx = list()
        segment2    = list()
        segment2Idx = list()

        for idx, (point, _) in enumerate(path.iter_segments()):

            if ( isclose(point,point1) or isclose(point,point2) ) and not START :
                START = True
            elif ( isclose(point,point1) or isclose(point,point2) ) and START:
                ENDafter = True
            
            if START and not END:
                segment1.append(list(point))
                segment1Idx.append(idx)
            else:
                segment2.append(list(point))
                segment2Idx.append(idx)
            
            if ENDafter: # Take the last point
                END = True

        assert( START and END ) # make sure we actually found both points

        if len(segment1) < len(segment2):
            segment = mpl.path.Path(segment1, closed=False)
            idx = segment1Idx
        else:
            segment = mpl.path.Path(segment2, closed=False)
            idx = segment2Idx
        
        return segment, idx

    
    def _getFigure(self, title):
        nrFigs = len(self.data)
        return getFigure(title, nrFigs, self.doubleFigure, self.figTitleSize, self.axesLabelSize)





class Color:
    """
    Helper to assign colors to float or integer values mapped to a given range.
    """
    def __init__(self, scaleMin=None, scaleMax=None):
        self.Nglobal = dict()
        self.cmap = plt.get_cmap('seismic_r')

        self.scaleMin = scaleMin
        self.scaleMax = scaleMax
        
        if scaleMin == scaleMax and scaleMin is not None:
            print('Warning: trying to set zero scaling range!')
            self.scaleMin = scaleMin
            self.scaleMax = scaleMin * 1.1

    def __call__(self, N):
        
        if self.scaleMin is None and self.scaleMax is not None:
            c = float(N) / self.scaleMax
        elif self.scaleMin is not None and self.scaleMax is None:
            c = float(N)  - self.scaleMin
        elif self.scaleMin is None and self.scaleMax is None:
            c = float(N)
        else:
            c = (float(N) - self.scaleMin) / ( self.scaleMax - self.scaleMin )
  
        return self.cmap(c)
    
    def getColorbar(self):
        return plt.cm.ScalarMappable(cmap=plt.get_cmap('seismic_r'), norm=plt.Normalize(vmin=self.scaleMin, vmax=self.scaleMax))


















