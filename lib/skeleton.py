# -*- coding: utf-8 -*-
"""
Created on Sun May  3 13:58:16 2015

@author: berliner
"""

import numpy as np
from skimage.morphology import skeletonize
from scipy.ndimage.filters import gaussian_filter
from matplotlib.path import Path

from utils import *


class Skeleton(IPyNotebookStyles):
    
    def __init__(self):
        
        super(Skeleton, self).__init__()
        
        self.data      = None
        self.imgBinary = None
        self.binSize   = None
        self.backbone  = None
    
    def setData(self, data):
        self.data = data
    
    def getResult(self, smoothed=True):
        if self.backbone is None:
            print("Warning: the backbone was not yet calculated")
        return self.backbone
    
    def _getFigure(self, title):
        nrFigs = len(self.data)
        return getFigure(title, nrFigs, self.doubleFigure, self.figTitleSize, self.axesLabelSize)
        
    def threshold(self, thres, binSize=10.0, sigma=5.0):
        self.binSize = float(binSize)
        self.imgBinary = dict()
        for frame, ax in self._getFigure("Binarized mitochondria clusters, threshold = %.2f" %thres):
            # Get the histogram and binarize
            H, extent = self._imageHistogram(frame, binSize, sigma)
            binary = H > float(thres)
            self.imgBinary[frame] = (binary, extent)
            ax.imshow(binary, extent=extent, interpolation='nearest', origin='upper', cmap=plt.cm.Greys_r)
    
    def getBackbone(self):
        if self.imgBinary is None:
            print("You need to threshold the clusters first.")
            return
        
        self.backbone = dict()
        for frame, ax in self._getFigure("Mitochondria skeletons"):
            # Create the skeleton from the binary image
            binary, extent = self.imgBinary[frame]
            imgSkeleton = skeletonize(binary)
            skeleton = self._skeletonToPath(imgSkeleton, self.binSize, extent)
            self.backbone[frame] = skeleton # skeleton is a matplotlib.path object
            
            # The vertices of the skeleton
            Xvert = skeleton.vertices[:,0]
            Yvert = skeleton.vertices[:,1]
            
            # The point localisations
            _, XYpoint, _ = self.data[frame]
            
            # Show an overlay of the binarized image and the skeleton
#            ax.imshow(binary, extent=extent, origin='upper', cmap=plt.cm.Greys_r)
            ax.scatter(x=XYpoint[:,0], y=XYpoint[:,1], color='blue', s=2)
            ax.scatter(x=Xvert, y=Yvert, color='red', s=4)
        return
    
    def _imageHistogram(self, frame, binSize, sigma):
        # Retrieve the data
        _, XY, _ = self.data[frame]
        X = XY[:,1] # for the histogram2d the x and y values are flipped
        Y = XY[:,0]
        
        # Calculate the number of bins that are necessary to reach binSize per bin 
        binsX = np.ceil( (np.max(X) - np.min(X)) / binSize )
        binsY = np.ceil( (np.max(Y) - np.min(Y)) / binSize )
        
        # Get the histogram and apply gaussian blur
        H, xedges, yedges = np.histogram2d(X, Y, bins=(binsX,binsY))
        H = gaussian_filter(H, float(sigma))

        # The extent of the axes (needed to scale the image correcly)        
        extent = [yedges[0], yedges[-1], xedges[-1], xedges[0]]
        
        return H, extent
    
    def _skeletonToPath(self, binaryImage, binSize, extent):
        # Get the points
        points = self._binaryImageToVertices(binaryImage, binSize)
        
        # Shift them to match the extent
        points[:,0] += extent[0]
        points[:,1] += extent[3]
        
        path = Path(points)
        
        return path
        
    def _binaryImageToVertices(self, binaryImage, binSize):
        
        points  = list()        
        # Find the pixels that are foreground
        idx = np.where( binaryImage )
        idx = zip(idx[1], idx[0]) # idx is now a list of tuples. Each tuple is giving
                                  # the row, column indices where True was found
                                  # x value is in column, y value is in row
        
        # Convert the pixels to nm
        for point in idx:
            points.append( ( (point[0])*binSize, (point[1])*binSize) )
        
        return np.asarray(points)