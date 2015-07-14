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

from utils import *



class Shape(IPyNotebookStyles):
    
    def __init__(self):
        
        super(Shape, self).__init__()
        
        self.contourData  = None
        self.skeletonData = None
    
    def setData(self, contourData, skeletonData, clusterData):
        self.contourData = contourData
        self.skeletonData = skeletonData
        self.clusterData = clusterData
    
    def _getFigure(self, title):
        nrFigs = len(self.contourData)
        return getFigure(title, nrFigs, self.doubleFigure, self.figTitleSize, self.axesLabelSize)
    
    def show(self):
        
        for frame, ax in self._getFigure("Mitochondria outlines with backbone estimate"):

            skeleton = self.skeletonData[frame]
#            patch = mpl.patches.PathPatch(skeleton, facecolor='none', edgecolor='red', lw=1)
#            ax.add_patch(patch)
            
            Xvert = skeleton.vertices[:,0]
            Yvert = skeleton.vertices[:,1]
            
#            window = (7,1)
            window = np.ones((7, 1))
            
            XY = np.vstack([Xvert.ravel(),Yvert.ravel()]).T
            XYSmooth = moving_average_2d(XY, window)
                
            ax.scatter(x=Xvert, y=Yvert, color='red', s=1, zorder=20)
#            ax.scatter(x=XYSmooth[:,0], y=XYSmooth[:,1], color='red', s=1)
            
            for contour in self.contourData[frame]:
                patch = mpl.patches.PathPatch(contour, facecolor='none', edgecolor='blue', lw=2)
                ax.add_patch(patch)
            
            _, XYcluster, _= self.clusterData[frame]
            ax.scatter(x=XYcluster[:,0], y=XYcluster[:,1], color='blue', s=1, alpha=0.7)
        
        plt.draw()