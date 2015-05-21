# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 16:26:43 2015

@author: berliner
"""
import sys
import os
import numpy as np
from scipy.ndimage.filters import gaussian_filter
import matplotlib.animation as animation
from matplotlib import pyplot as plt
import matplotlib as mpl


class MovieGenerator(object):
    
    def __init__(self, data, contour=None, sigma=10.0):
        
        assert( isinstance(data, dict) )
        
        self.data, self.binsX, self.binsY = self._frameData(data)
        self.contour = contour
        self.sigma = float(sigma)
        
        self.fig = plt.figure(figsize=(10,10), dpi=120)
        plt.axis('off')
        plt.tight_layout()

    
    def make(self, fname):
        
        if not fname[-4:] == '.mp4':
            fname = fname + '.mp4'
            
        frames = self._initFrames()
        ani = animation.ArtistAnimation(self.fig, frames, interval=10000, blit=False,
                                        repeat_delay=1000, repeat=True)
                                        
        # save the animation as an mp4.  This requires ffmpeg or mencoder to be
        # installed.  The extra_args ensure that the x264 codec is used, so that
        # the video can be embedded in html5.  You may need to adjust this for
        # your system: for more information, see
        # http://matplotlib.sourceforge.net/api/animation_api.html
        # The ffmpeg binaries should be residing in the PointObject Path
        if 'linux' in sys.platform:
            ffmpegFolder = os.path.join('..', 'external', 'ffmpeg', 'linux', 'ffmpeg-2.4.4', 'ffmpeg')
        elif 'win' in sys.platform:
            ffmpegFolder = os.path.join('..', 'external', 'ffmpeg', 'win64', 'ffmpeg-20150501-git-02001ad-win64-static', 'bin', 'ffmpeg')

#        plt.rcParams['animation.ffmpeg_path'] = '/home/berliner/bin/ffmpeg/ffmpeg-2.4.4/ffmpeg'
        plt.rcParams['animation.ffmpeg_path'] = ffmpegFolder
        mywriter = animation.FFMpegWriter(fps=1)
        ani.save(fname,writer=mywriter, savefig_kwargs={'facecolor':'black'})

    
    def _frameData(self, data, binSize=10.0):
        """ Ensures that the data has the same extend for each frame """
        # Get the top right corner point
        xMaxTotal = np.max( [ np.max(points[:,0]) for _, points, _ in list(data.values()) ] )
        yMaxTotal = np.max( [ np.max(points[:,1]) for _, points, _ in list(data.values()) ] )
        # Get the bottom left corner point
        xMinTotal = np.min( [ np.min(points[:,0]) for _, points, _ in list(data.values()) ] )
        yMinTotal = np.min( [ np.min(points[:,1]) for _, points, _ in list(data.values()) ] )
        
        # Add them to each frame
        newData = dict()
        for frame in data:
            _, XY, _ = data[frame]
            XY = np.concatenate( (XY, np.asarray( [[xMinTotal, yMinTotal],[xMaxTotal, yMaxTotal]] ) ) )
            newData[frame] = XY
        
        # Calculate how many bins are needed to reach binSize for each bin
        binsY = np.ceil( (xMaxTotal - xMinTotal) / float(binSize) )
        binsX = np.ceil( (yMaxTotal - yMinTotal) / float(binSize) )
        
        return newData, binsX, binsY

    
    def _initFrames(self, apply_gaussian_filter=False, vmin=0, vmax=0.1):
        images = list()
        # Make the frames
        for frameNr in range(1,len(self.data)+1):
            # From the docs we read "Values in x are histogrammed along the first dimension"
            # so we flip around to make it comparable to the image.
            X = self.data[frameNr][:,1]
            Y = self.data[frameNr][:,0]
            
            # Compute the 2D histogram
            H, xedges, yedges = np.histogram2d(X, Y, bins=(self.binsX,self.binsY))
            H += 1
            H = np.log( H )
            extent = [yedges[0], yedges[-1], xedges[-1], xedges[0]] # the boundaries of the histogram
            
            # Apply the gaussian filter
            if apply_gaussian_filter:
                H = gaussian_filter(H, self.sigma)
            
            # Plot the localisations
            im = plt.imshow(H, extent=extent, interpolation='nearest', origin='upper', cmap=plt.cm.Greys_r)
            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
            im.set_norm(norm)
            
            artists = [im, ]
            
            # If contour was given plot it
            if self.contour is not None:
                contourPaths = list()
                ax = im.get_axes() # get the image axes
                
                for path in self.contour[frameNr]:
                    patch = mpl.patches.PathPatch(path, facecolor='none', edgecolor='red', lw=3)
                    contourPaths.append(ax.add_patch(patch))
                artists.extend(contourPaths)
                
            images.append( artists )
        
        return images
        







