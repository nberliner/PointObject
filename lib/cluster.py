# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 18:29:00 2015

@author: berliner
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pylab as plt
from matplotlib.patches import Patch
from sklearn.cluster import DBSCAN

from pandas import DataFrame

from datetime import datetime
from random import shuffle
from copy import deepcopy
from time import sleep

from utils import *
from mplWidgets import LassoManager

#def _pickle_method(method):
#    func_name = method.im_func.__name__
#    obj = method.im_self
#    cls = method.im_class
#    return _unpickle_method, (func_name, obj, cls)
#
#def _unpickle_method(func_name, obj, cls):
#    for cls in cls.mro():
#        try:
#            func = cls.__dict__[func_name]
#        except KeyError:
#            pass
#        else:
#            break
#    return func.__get__(obj, cls)
#
#import copy_reg
#import types
#copy_reg.pickle(types.MethodType, _pickle_method, _unpickle_method)

class C(object):
    def __init__(self, eps, min_samples):
        self.eps = eps
        self.min_samples = min_samples
    def __call__(self, args):
        frameNr, XY = args
        return frameNr, DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(XY)

class Cluster(IPyNotebookStyles):
    
    def __init__(self):
        
        super(Cluster, self).__init__()
        
        self.data       = None
        self.dataROI    = None
        self.dataFrame   = None
        self.clustersList = None
        self.clusterTable = None
        self.clustering = None
    
    def setData(self, dataFrame):
        assert( isinstance(dataFrame, DataFrame) )
        self.dataFrame = dataFrame
    
    def getResult(self):
        if self.clustering is None:
            print('Please run the clustering first.')
            return
        
        if self.dataROI is None:
            return self.data
        else:
            return self.dataROI
    
    def _getFigure(self, title):
        nrFigs = len(self.data)
        return getFigure(title, nrFigs, self.doubleFigure, self.figTitleSize, self.axesLabelSize)
    
    def cluster(self, eps, min_samples, frame=None, clusterSizeFiler=50, askUser=True):
        startTime = datetime.now() # set the calculation start time
        
        if frame is None:
            frames = [ frameData for _, frameData in self.dataFrame.groupby('movieFrame') ]
        else:
            frames = [ self.dataFrame[ self.dataFrame.movieFrame==frame ], ]
        
        # First do the clustering on multiple cores
#        func = lambda (frameNr, XY): (frameNr, DBSCAN(eps=eps, min_samples=min_samples).fit(XY))
        func = C(eps, min_samples)
        # This is a bit short.. this is essentially what is happening..
        # for frameNr, frameData in enumerate(frames):
        #     XY = np.asarray( frameData[['x','y']] ) # get the x,y data
        #     db = DBSCAN(eps=eps, min_samples=min_samples).fit(XY)       
        clustering = parmap( func, [ (frameNr, np.asarray(frameData[['x','y']]) ) \
                                    for (frameNr, frameData) in enumerate(frames, start=1) ] )
        
        # convert the result into a dict for fast lookup
        self.clustering = { key: value for key, value in clustering }
        
        # We're done with clustering, print some interesting messages
        time = datetime.now()-startTime
        print("Finished clustering in:", str(time)[:-7])
        
        
        # Now get the user to link the detected clusters
        self.clustersList = list()
        for frameNr, frameData in enumerate(frames, start=1):
            XY = np.asarray( frameData[['x','y']] ) # get the x,y data
            db = self.clustering[frameNr] # get the clustering result
            
            core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
            core_samples_mask[db.core_sample_indices_] = True
            labels = db.labels_
            
            # Number of clusters in labels, ignoring noise if present.
            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

            # Show a plot of detected clusters
            fig = plt.figure(figsize=self.singleFigure)
            fig.suptitle('Frame %d \n Estimated number of clusters: %d' %(frameNr, n_clusters_), size=self.figTitleSize)
            ax = fig.add_subplot(111)  
            legend_patches = list()
            
            xlimits = [np.min(XY[:,0]), np.max(XY[:,0])]
            ylimits = [np.min(XY[:,1]), np.max(XY[:,1])]
            ax.set_xlim(xlimits)
            ax.set_ylim(ylimits)
            
            ax.set_xlabel("x position in nm", size=self.axesLabelSize)
            ax.set_ylabel("y position in nm", size=self.axesLabelSize)
            
            # Black removed and is used for noise instead.
            unique_labels = list(set(labels))
            shuffle(unique_labels) # if not shuffled they are ordered by size and the interesting
                                   # clusters are closer on the colorscale and harder to distinguish
            colors = plt.cm.Paired(np.linspace(0, 1, len(unique_labels)))
            for k, col in zip(unique_labels, colors):
                if k == -1:
                    # Black used for noise.
                    col = 'k'
            
                class_member_mask = (labels == k)
            
                xyCore = XY[class_member_mask & core_samples_mask]
                ax.plot(xyCore[:, 0], xyCore[:, 1], 'o', markerfacecolor=col,
                         markeredgecolor='none', markersize=2);
            
                xyEdge = XY[class_member_mask & ~core_samples_mask]
                ax.plot(xyEdge[:, 0], xyEdge[:, 1], 'o', markerfacecolor=col,
                         markeredgecolor='none', markersize=1);
                
                # Create the legend but keep only big clusters
                if len(xyCore) > clusterSizeFiler and not k == -1:
                    legend_patches.append(Patch(color=col, label=str(k)))
                    
                    # Plot the label in the center
                    x_center = np.mean(xyCore[:,0])
                    y_center = np.mean(xyCore[:,1])
                    ax.text(x=x_center, y=y_center, s=str(k), size=24, color='blue', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.3'))
            
            ax.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            plt.show()
            
            if frame is None and askUser:
                keep = input("Which cluster(s) to keep? Please type integer value(s) ")
                if keep == "":
                    pass
                else:
                    self.clustersList.append(keep)
                print('\n') # Some spacing makes it more pleasant to the eye
        
        # Convert the mito list into a dict for better lookup
        if frame is None:
            self.data = self._selectClusters()

            print('\nYou selected the following clusters:')
            for frame, cluster in list(self.data.items()):
                print('Frame %d: %s' %(frame, cluster[0]))
            print('\nUse checkClusters()   to verify the cluster assignment.')
            print('Use confineClusters() to select the ROI for each cluster.')
    
    
    def _selectClusters(self):
        assert( self.clustering is not None )
        # Convert the mito list into a dict for better lookup
        self.clusterTable = { frame: clusters for frame, clusters in enumerate(self.clustersList, start=1) }
        
        # Add the localisation data to the dict
        _data = dict()
        for frame, clusterIDs in list(self.clusterTable.items()):
            # Get the frame data
            frameData = self.dataFrame.groupby('movieFrame').get_group(frame)
            XY = np.asarray( frameData[['x','y']] )
            
            # Get the clustering result
            db = self.clustering[frame]
            
            # Select the core localisations
            core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
            core_samples_mask[db.core_sample_indices_] = True
            
            # Select the user selected clusters
            XYmitoCore = list()
            XYmitoEdge = list()
            c = deepcopy(clusterIDs.strip())
            for k in c.split(' '):
                class_member_mask = (db.labels_ == int(k))
                XYmitoCore.append( XY[class_member_mask & core_samples_mask]  )
                XYmitoEdge.append( XY[class_member_mask & ~core_samples_mask] )
            
            XYmitoCore = np.concatenate(XYmitoCore, axis=0)
            XYmitoEdge = np.concatenate(XYmitoEdge, axis=0)
            
            _data[frame] = (clusterIDs, XYmitoCore, XYmitoEdge)
        
        return _data
            
        
    def confineClusters(self, frame=None):
        # Check if switched to qt mode
        if not mpl.get_backend() == 'Qt4Agg':
            print('Switch to the Qt backend first by executing "%pylab qt"')
            return False
        
        # Check which frame(s) should be refined
        if frame is None:
            frames = range(1,len(self.data)+1)
        elif isinstance(frame, list):
            frames = frame
        else:
            assert( isinstance(frame, int) )
            frames = [frame, ]
        
        if self.dataROI is None:
            self.dataROI = dict()
        # Let the user select the ROI
        for frame in frames:
            # Get the ROI
            ROI = self._selectROI(frame)
            
            # Filter the points in the ROI
            clusters, XYdataCore, XYdataEdge = self.data[frame] # the unfiltered data
            XYdataCoreROI = XYdataCore[ ROI.contains_points(XYdataCore) ]
            if XYdataEdge is None:
                XYdataEdgeROI = []
            else:
                XYdataEdgeROI = XYdataEdge[ ROI.contains_points(XYdataEdge) ]
            
            # Add the confined data
            self.dataROI[frame] = [clusters, XYdataCoreROI, XYdataEdgeROI]
    
#        print('Selected the structure in %i frames.' %frame)
    
    def _selectROI(self, frame):
        # Check if switched to qt mode
        if not mpl.get_backend() == 'Qt4Agg':
            print('Switch to the Qt backend first by executing "%pylab qt"')
            return False, False

        _, XYdataCore, XYdataEdge = self.data[frame]
        
        fig = plt.figure(figsize=self.singleFigure)
        ax  = fig.add_subplot(111)
        ax.set_title("Frame %d" %frame)
        ax.set_xlabel("x position in nm", size=self.axesLabelSize)
        ax.set_ylabel("y position in nm", size=self.axesLabelSize)
        
        ax.set_xlim( [ np.min(XYdataCore[:,0])-500, np.max(XYdataCore[:,0])+500 ] )
        ax.set_ylim( [ np.min(XYdataCore[:,1])-500, np.max(XYdataCore[:,1])+500 ] )

        ax.scatter(x=XYdataCore[:,0], y=XYdataCore[:,1], edgecolor='none', facecolor='blue', s=2)
        if XYdataEdge != []:
            ax.scatter(x=XYdataEdge[:,0], y=XYdataEdge[:,1], edgecolor='none', facecolor='red', s=1)

        # Create the widget that lets the user draw the ROI
        lman = LassoManager(fig, ax)
        # The ROI was drawn, let the user look at it for a second and then go on
        sleep(1)

        plt.close(fig)
        return lman.collection

 
    def showClusters(self, original=False):

        for frame, ax in self._getFigure("Selected clusters per frame"):
            
            if self.dataROI is not None and not original:
                _, XYdataCore, XYdataEdge = self.dataROI[frame]
            else:
                _, XYdataCore, XYdataEdge = self.data[frame]

            ax.scatter(x=XYdataCore[:,0], y=XYdataCore[:,1], edgecolor='none', facecolor='blue', s=2)
        
        plt.show()
        print('\nYou can look at idividual clusters with checkCluster(frame)')
    
    
    def checkCluster(self, frame=1, s=4, xlim=False, ylim=False, original=False):
        # Get the data
        if self.dataROI is not None and not original:
            _, XYdataCore, XYdataEdge = self.dataROI[frame]
        else:
            _, XYdataCore, XYdataEdge = self.data[frame]
        
        # Create the figure
        fig = plt.figure(figsize=self.singleFigureLarge)
        fig.suptitle("Mitochondria clusters in frame %d" %frame, size=self.figTitleSize)
        ax = fig.add_subplot(111)
        ax.set_xlabel("x position in nm", size=self.axesLabelSize)
        ax.set_ylabel("y position in nm", size=self.axesLabelSize)
        
        # Set the axes limits
        if xlim:
            assert( isinstance(xlim, list) and len(xlim) == 2 )
            ax.set_xlim(xlim)
        if ylim:
            assert( isinstance(ylim, list) and len(ylim) == 2 )
            ax.set_ylim(ylim)

        if s == 1: # Make sure that the size is not set to zero
            s = 2
        ax.scatter(x=XYdataCore[:,0], y=XYdataCore[:,1], edgecolor='none', facecolor='blue', s=s)
        ax.scatter(x=XYdataEdge[:,0], y=XYdataEdge[:,1], edgecolor='none', facecolor='red',  s=s-1)
        plt.show()




