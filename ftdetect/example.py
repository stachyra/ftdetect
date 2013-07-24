"""
example.py

This demonstration module shows how to correctly use the following image
feature detection algorithms:

    features.wangbrady()
    features.susanEdge()
    features.susanCorner()

Required Python modules:

    PIL
    matplotlib
    numpy
    scipy

All code in this package has been tested using Enthought Python
Distribution 2.7.3 with Enthought Canopy Express 1.0.0.1160 (available
for free at www.enthought.com/downloads/)

AUTHOR: Andrew L. Stachyra
DATE: 5/17/2013
"""

try:
    from PIL import Image
except ImportError:
    raise SystemExit('PIL must be installed to run this example')
try:
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import ndimage
except ImportError:
    raise SystemExit('matplotlib/numpy/scipy must be installed to run ' + \
                     'this example')
import os
import warnings
import sys
# Package internal import syntax changes in 3.x, attempt to forestall breakage
try:
    from . import features
except:
    import features

def demo(imagefiles=None):

    """
    Runs a simple demo, using canned images if the user doesn't specify
    any, or alternatively, processes image files provided as inputs or
    at the command line.

    Parameters
    ----------

    imagefiles: array_like, optional

        A list of strings containing image file names (.png, .jpg, .gif,
        etc.) for the demo to process.  If no files are given, the demo
        uses a pair of canned examples instead.  For an initial test, it
        is recommended that if the user supplies his own images, they
        should be quite small (e.g., 256 X 256 pixels) as the SUSAN
        algorithms can be quite slow.

    Returns
    -------

    None
    
    """

    # If imported and then called interactively with no arguments...
    if imagefiles is None:
        imagefiles = []

    # Calls from the command line with no arguments OTOH look like this...
    if len(imagefiles) == 0:
        # If user supplies no input files, generate a canned example
        sampledir = 'images'
        samplefiles = ['blocksTest.gif', 'house.png']
        for fname in samplefiles:
            imagefiles.append(os.path.join(os.path.dirname(__file__),
                                           sampledir, fname))

    # Relative location of figure text labels
    xoffset, yoffset = 0.975, 0.08

    for fname in imagefiles:
        # Open image data file and convert to grayscale numpy array
        try:
            print('Image file: ' + fname)
            pilobj = Image.open(fname)
        except:
            warnings.warn("Can't open image file: " + fname)
            continue
        img = np.asarray(pilobj.convert('L'))

        # Absolute position of figure text labels
        xpos, ypos = xoffset*img.shape[1], yoffset*img.shape[0]
        
        # Uncomment this line in order to preprocess with Gaussian blurring
        #img = ndimage.filters.gaussian_filter(img, sigma=1)

        # Calculate results for three different image feature detection algorithms
        sys.stdout.write('    running Wang-Brady corner detection...')
        sys.stdout.flush()
        wbcorner = features.wangbrady(img)
        print('done!')

        sys.stdout.write('    running SUSAN edge detection...')
        sys.stdout.flush()
        suedge = features.susanEdge(img, minmem=6) # Bonus processing: minmem=6
        print('done!')

        sys.stdout.write('    running SUSAN corner detection...')
        sys.stdout.flush()
        sucorner = features.susanCorner(img)
        print('done!')

        # Plot the original unmodified input figure
        fig = plt.figure()
        fig.canvas.set_window_title(fname)
        ax = fig.add_axes((0.01, 0.51, 0.48, 0.48))
        ax.set_axis_off()
        ax.imshow(img, interpolation='nearest', cmap='Greys_r')
        plt.text(xpos, ypos, 'Original', color='k', horizontalalignment='right')

        # Plot the results of the Wang-Brady corner detector
        ax = fig.add_axes((0.51, 0.51, 0.48, 0.48))
        ax.set_axis_off()
        ax.imshow(img, interpolation='nearest', cmap='Greys_r')
        ax.autoscale(tight=True)
        vidx, hidx = wbcorner.nonzero()
        ax.plot(hidx, vidx, 'go')
        plt.text(xpos, ypos, 'Wang-Brady corners', color='g',
                 horizontalalignment='right')

        # Duplicating the original grayscale image into three color channels...
        imgedge = img.reshape((img.shape[0], img.shape[1], 1)).repeat(3,2)
        # ...superimpose the results of SUSAN edge detection in red...
        vidx, hidx = suedge.nonzero()
        for ii in range(len(vidx)):
            imgedge[vidx[ii]][hidx[ii]][0] = 255
            imgedge[vidx[ii]][hidx[ii]][1] = 0
            imgedge[vidx[ii]][hidx[ii]][2] = 0
        # ...and plot the results of the SUSAN edge detector
        ax = fig.add_axes((0.01, 0.01, 0.48, 0.48))
        ax.set_axis_off()
        ax.imshow(imgedge, interpolation='nearest')
        ax.autoscale(tight=True)
        plt.text(xpos, ypos, 'SUSAN edges', color='r', horizontalalignment='right')

        # Plot the results of the SUSAN corner detector
        ax = fig.add_axes((0.51, 0.01, 0.48, 0.48))
        ax.set_axis_off()
        ax.imshow(img, interpolation='nearest', cmap='Greys_r')
        ax.autoscale(tight=True)
        vidx, hidx = sucorner.nonzero()
        ax.plot(hidx, vidx, 'bo')
        plt.text(xpos, ypos, 'SUSAN corners', color='b',
                 horizontalalignment='right')

    plt.show()

if __name__ == '__main__':
    demo(sys.argv[1:])
