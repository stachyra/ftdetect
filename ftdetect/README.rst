========
ftdetect
========

Python Image Feature Detection
------------------------------

This package implements a small assortment of (currently, three) algorithms related to computer vision and image feature detection:

    - Wang-Brady_ corner detection [WB1995]_
    - SUSAN_ edge detection [SB1995]_ [SB1997]_
    - SUSAN_ corner detection [SB1995]_ [SB1997]_

The code has been developed and tested using the EPD free Python 2.7.3 distribution (now offered as `Enthought Canopy Express`_).
      
.. _Wang-Brady: http://en.wikipedia.org/wiki/Corner_detection#The_Wang_and_Brady_corner_detection_algorithm
.. _SUSAN: http://en.wikipedia.org/wiki/Corner_detection#The_SUSAN_corner_detector
.. _Enthought Canopy Express:  https://www.enthought.com/downloads/

Package Contents
----------------

The package is organized into four modules, briefly described below:

    **features**
        Contains the top-level end user functions which actually return the image features

        **.wangbrady()**: Implements Wang-Brady corner detection

        **.susanEdge()**: Implements SUSAN edge detection

        **.susanCorner()**: Implements SUSAN corner detection 

    **masks**
        Generates footprints and eventually passes them down into fast compiled DLL code in scipy.ndimage._ndimage.pyd, via functions in scipy.ndimage.filters module

        **.getDeriv()** Calculates first and second partial derivatives with respect to image grayscale brightness

        **.circFootprint()** Calculates pixel membership (within a square array) of pixels inside a circular footprint

        **.usan()** Calculates USAN response of each pixel in the image (see [SB1995]_ [SB1997]_ for details) 

    **cleanedges**
        Contains several functions to refine and remove noise from the raw response returned by edge detection algorithms 

        **.roundAngle()** Rounds edge normal direction angle to one of four major directions: horizontal, vertical, diagonal (positive slope) or diagonal (negative slope)

        **.nonMaxSuppEdge()** Suppresses edge responses from pixels which are not local maxima on a line segment along their local edge normal direction

        **.hystThresh()** Implements Canny style dual level hysteresis thresholding

        **.listEdge()** Assigns pixels with non-zero edge response into discrete groups based upon contiguity and also provides a list of member pixels of each group

        **.minMembership()** Rejects edges that consist of fewer than a minimum number of contiguous pixels

    **example**
        Demonstrates proper usage of the functions in the features module

        **.demo()** Runs the algorithms in the features module on a couple of standard test images (or user-supplied image files, if desired) using default parameter settings

External Requirements
---------------------

In addition to the Python interpreter and its constituent libraries, four external packages, which are not necessarily included in the Python standard libraries, are also required in order to use this package:

    - numpy_
    - scipy_
    - matplotlib_ (necessary for demo script only)
    - PIL_ (necessary for demo script only)

At least some of these packages, such as numpy and scipy, include portions which are not native Python and therefore have to be compiled, meaning that your system must have compilers installed in order to run them.  A popular and somewhat easier alternative is to download `Enthought Canopy Express`_ (formerly EPD Free), the free version of a Python-based GUI scientific analysis environment which also includes a Python distribution containing precompiled versions of all of the packages above.  As of this writing, a version of the PIL package does not yet exist for Python 3.x, which effectively means that both the Enthought Python Distribution is currently limited to Python 2.7.3, and also that this package has not yet been tested on versions of Python that are later than this, either.

.. _numpy:                     http://www.numpy.org/
.. _scipy:                     http://www.scipy.org/
.. _matplotlib:                http://matplotlib.org/
.. _PIL:                       http://www.pythonware.com/products/pil/

Install
-------

The ftdetect package is distributed using standard Python distutils.  After downloading and unzipping the package, open a command line shell, change directory to the package folder that contains setup.py, and simply type::

    python setup.py install
    
Example
-------

A demo script has been provided, with a couple of standard test images, to help illustrate usage.  To run it from the command line, type::

    python -c "import ftdetect.example; ftdetect.example.demo()"

To run it from an interactive Python session (IDLE, IPython, Enthought Canopy GUI, etc.), type::

    import ftdetect.example
    ftdetect.example.demo()

The demo script is also capable of running on user-selected figures, like so::

    ftdetect.example.demo(['filea.jpg', 'fileb.gif', 'filec.png'])

however, execution speed may vary widely.  The SUSAN algorithms in particular are prone to running very slowly on some images, particularly those with large dimensions, textured surfaces, or complicated fine structure.  If that happens, choose another image which is both smaller and doesn't contain as much fine detail.  Another more advanced option is to vary the algorithm input parameters so as to decrease the sensitivity, however, the code provided in the example module, being conceived primarily as instructional in nature, doesn't bother to expose this level of control to the end-user--you'll have to access the algorithms directly from the features module itself in that case. 

References
----------

.. [WB1995] Han Wang and Michael Brady, "Real-time corner detection algorithm for motion estimation", Image and Vision Computing 13(9): 695-703 (Nov 1995). doi_: `10.1016/0262-8856(95)98864-P  <http://dx.doi.org/10.1016/0262-8856(95)98864-P>`_

.. [SB1995] S. M. Smith and J. M. Brady, `"SUSAN--A New Approach to Low Level Image Processing" <http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.24.2763>`_, Technical Report TR95SMSIc (1995).

.. [SB1997] S. M. Smith and J. M. Brady, "SUSAN--A New Approach to Low Level Image Processing", International Journal of Computer Vision 23(1): 45-78 (May 1997). doi_: `10.1023/A:1007963824710 <http://dx.doi.org/10.1023/A:1007963824710>`_

.. _doi: http://en.wikipedia.org/wiki/Digital_object_identifier

Package Maintenance Information
-------------------------------

*Version*: 1.0.1

*Date*: 2018-01-18

*URL*: https://github.com/stachyra/ftdetect

*Author*: Andrew L. Stachyra

*Contact*: andrewlstachyra@gmail.com
