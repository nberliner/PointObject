# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 12:12:36 2015

@author: berliner
"""
import matplotlib.pylab as plt
from matplotlib.widgets import Lasso, RectangleSelector, Button
from matplotlib import path
import numpy as np

# Adapted from here: http://stackoverflow.com/a/23348176
# See link for information on how to adapt it to subplots

class LassoManager(object):
    def __init__(self, fig, ax):
        
        self.fig    = fig
        self.axes   = ax
        self.canvas = ax.figure.canvas
        
        # Adjust the figure layout and add the buttons
        print("Adding buttons")
        fig.subplots_adjust(bottom=0.2)
        axprev = self.fig.add_axes([0.7, 0.05, 0.1, 0.075])
        axnext = self.fig.add_axes([0.81, 0.05, 0.1, 0.075])
        bnext = Button(axnext, 'Accept')
        bnext.on_clicked(self.accept)
        bprev = Button(axprev, 'Reject')
        bprev.on_clicked(self.reject)
        
        self.collection = None # Stores the Path object
        self.accepted = False
        self.rejected = True

        # Connect the user interaction
        self.cid = self.canvas.mpl_connect('button_press_event', self.onpress)

        # start a blocking event loop
        self.canvas.start_event_loop(timeout=-1)
   

    def callback(self, verts):
        self.collection = path.Path(verts)
#        self.canvas.stop_event_loop() # release the loop so that the computation continues

    def onpress(self, event):
        if self.canvas.widgetlock.locked(): return
        if event.inaxes is None: return
        self.lasso = Lasso(event.inaxes, (event.xdata, event.ydata), self.callback)
        # acquire a lock on the widget drawing
        self.canvas.widgetlock(self.lasso)

    def accept(self, event):
        print("accepting")
        self.accepted = True
        self.canvas.stop_event_loop() # release the loop so that the computation continues
        
    def reject(self, event):
        print("removing")
        self.canvas.stop_event_loop() # release the loop so that the computation continues


class RoiSelector(object):
    
    def __init__(self, ax):
        
        self.axes   = ax
        self.canvas = ax.figure.canvas
        
        self.edges = None # Stores the drawn rectangle edges

        # Connect the user interaction
        self.cid = self.canvas.mpl_connect('button_press_event', self.onpress)

        # start a blocking event loop
        self.canvas.start_event_loop(timeout=-1)
   

    def callback(self, eclick, erelease):
        # Get the edges
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        # Sort them to have bottom left, top right order
        xbl = np.min([x1,x2])
        ybl = np.min([y1,y2])
        xtr = np.max([x1,x2])
        ytr = np.max([y1,y2])
        self.edges = (xbl, ybl, xtr, ytr)
        self.canvas.stop_event_loop() # release the loop so that the computation continues

    def onpress(self, event):
        if self.canvas.widgetlock.locked(): return
        if event.inaxes is None: return
        self.rect = RectangleSelector(self.axes, self.callback,
                                       drawtype='box', useblit=True,
                                       button=[1,3], # don't use middle button
                                       minspanx=5, minspany=5,
                                       spancoords='data')
        # acquire a lock on the widget drawing
        self.canvas.widgetlock(self.rect)



class CurvatureSelector(object):
    
    def __init__(self, ax):
        
        self.ax     = ax
        self.canvas = ax.figure.canvas
        
        self.points = list()
    
        # Connect the user interaction
        self.cid = self.canvas.mpl_connect('button_press_event', self.onpress)

        # start a blocking event loop
        self.canvas.start_event_loop(timeout=-1)
    
    def onpress(self, event):
#        if self.canvas.widgetlock.locked(): return
        if event.inaxes is None: return
        
        # Get the point
        x, y = event.xdata, event.ydata
        
        self._addPoint(x,y)


    def _addPoint(self, x, y):
        
        self.points.append( (x,y) ) # store the point
        
        self.ax.scatter( x=x, y=y, color='red', s=50) # plot the point
        plt.draw()

        if len(self.points) == 4:
            self.canvas.stop_event_loop() # release the loop so that the computation continues













