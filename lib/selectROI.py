# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 14:24:28 2015

@author: berliner
"""
import matplotlib as mpl
import matplotlib.pylab as plt

def selectROI(frame, mito):
    # Check if switched to qt mode
    if not mpl.get_backend() == 'Qt4Agg':
        print('Switch to the Qt backend first by executing "%pylab qt"')
        return
    
    global selected
    selected = False

    def callback(event):
        import sys
        print("clicked:", event)
        sys.stdout.flush()
        selected = True
        
    _, XYcore, XYedge = mito[frame]
    
    fig = plt.figure(figsize=(10,7))
    ax  = fig.add_subplot(111)
    ax.set_title("Frame %d" %frame)
    ax.set_xlabel("x position in nm", size=12)
    ax.set_ylabel("y position in nm", size=12)
    
    ax.scatter(x=XYcore[:,0], y=XYcore[:,1], edgecolor='none', facecolor='blue', s=2)
    
#    from matplotlib.widgets import Button
#    b1 = Button(ax, 'Button 1')
#    b1.on_clicked(callback)
    
#    lman = LassoManager(ax, XYcore)