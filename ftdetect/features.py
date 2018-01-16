"""
features.py

Provides top-level function calls for implementing various types of
image feature detectors

AUTHOR: Andrew L. Stachyra
DATE: 5/17/2013
"""

import numpy as np
from scipy.ndimage import filters
# Package internal import syntax changes in 3.x, attempt to forestall breakage
try:
    from . import masks
except:
    import masks
try:
    from . import cleanedges
except:
    import cleanedges

def wangbrady(image, S=0.1, T1=500, T2=2000, weights=[3./16, 10./16, 3./16],
              mfsize=10):

    """
    Implements Wang-Brady style corner detection, based upon the
    description provided in "Real-time corner detection algorithm for
    motion estimation", Han Wang and Michael Brady, Image and Vision
    Computing 13(9): 695-703 (Nov 1995).

    Parameters
    ----------

    image: array_like

        2D array containing grayscale image data.  Only grayscale pixel
        values are supported--cannot handle 3-channel color.

    S: scalar, optional

        Image curvature parameter introduced in eq. 8 and 9 of the
        reference article.  Defaults to 0.1, as suggested by article
        authors.

    T1, T2: scalar, optional

        User-defined thresholds arising in eq. 9 of the reference
        article.  Defaults to (500, 2000), as recommended by article
        authors.

    weights: array_like

        1D array with 3 elements signifying type of derivative mask to
        use (e.g., Sobel, Prewitt, Scharr, etc.)  Defaults to
        [3./16, 10./16, 3./16], corresponding to Scharr.  Values should
        be normalized so that they sum to one; see masks.getDeriv for
        further details.

    mfsize: scalar, optional

        Size value to be passed through down to
        scipy.ndimage.filters.maximum_filter().  Effectively sets the
        maximum permitted density scale for corners, as multiple
        corner-reponsive pixels which are closer together than this
        distance from one another will tend to result in only the pixel
        with the largest corner response being accepted.  See
        scipy.ndimage.filters.maximum_filter() for further details.

    Returns
    -------

    corner: ndarray

        The corner response of the image as calculated by the Wang-Brady
        corner detection algorithm.

    """

    # Substitute very small number for zero, to avoid divide-by-zero error
    def removeZeros(dummy, subvalue = 1e-9):
        idx = (dummy == 0)
        dummy[idx] = 1e-9
        return dummy

    # Because we are calculating dxdy cross derivatives, weights in this case
    # must be normalized to have correct sizing relative to other derivatives
    if sum(weights) != 1:
        raise ValueError('Sum of weights must be normalized to 1, due to ' +
                         'use of cross derivatives')

    # Calculate first and second derivatives
    dx = masks.getDeriv(image, weights=weights, axes=[0])
    dy = masks.getDeriv(image, weights=weights, axes=[1])
    dxx = masks.getDeriv(image, weights=weights, axes=[0, 0])
    dyy = masks.getDeriv(image, weights=weights, axes=[1, 1])
    dxy = masks.getDeriv(image, weights=weights, axes=[0, 1])

    # Calculate image gradient squared and tangent second derivative
    grdsqd = removeZeros(dx*dx + dy*dy)
    numerator = removeZeros(dy*dy*dxx - 2*dx*dy*dxy + dx*dx*dyy)
    dtdt = np.divide(numerator, grdsqd)

    # Apply eq. 9 from reference article, together with requirement that
    # results should all be a local maxima, to obtain final corner response
    gamma = dtdt * dtdt - S * grdsqd 
    gmax = filters.maximum_filter(gamma, size=mfsize)
    corner = np.where((gamma == gmax) & (grdsqd > T1) & (gamma > T2), gamma, 0)

    return corner

def susanEdge(image, radius=3.4, fptype=bool, t=25, gfrac=None, cgthresh=None, 
              nshades=256, tlo=0.1, thi=0.3, minmem=1):

    """
    Implements SUSAN edge detector algorithm as described in "SUSAN--A
    New Approach to Low Level Image Processing", S. M. Smith and J. M.
    Brady, Technical Report TR95SMSIc (1995), available at:
    http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.24.2763 .
    Alternatively, there is also a slightly abridged version of the
    same reference, with identical title and authors, available at
    International Journal of Computer Vision 23(1): 45-78 (1997).

    image: array_like

        Array containing grayscale image data.  Only grayscale pixel
        values are supported--cannot handle 3-channel color.

    radius: scalar, optional

        Circular footprint radius, in units of pixels; passed down
        as input to masks.circFootprint().  Default is 3.4, as
        recommended in reference.

    fptype: data-type, optional

        Data type of circular footprint; passed to masks.circFootprint().
        Default is bool, since the version of the algorithm originally
        described in the reference did not define the algorithm behavior
        for any other type of footprint (we have trivially extended it
        to cover float type footprints as well, however there is not
        much observable different in the results, as it only affects the
        weighting of the pixels at the edge of the footprint, not the
        central region).

    t: scalar, optional

        Threshold value for color difference required to exclude/include
        a pixel in the USAN region described in the reference article.
        Default is 25 grayscale levels, as suggested by article authors,
        and assumes a grayscale image with a total of 256 distinct
        shades.  As described in eq. 4, this is technically a "soft"
        threshold, rather than a hard cutoff, and moreover it's also
        bidirectional; i.e., a setting of t=25 actually means +/- 25.

    gfrac: scalar, optional

        Fraction of maximum number of USAN pixels to assign to the "g"
        parameter in eq. 3 of the reference article.  Default setting
        of gfrac=None triggers lower level masks.usan() function to
        select a context-appropriate default value (for edge detection
        specifically, default is gfrac=0.75) as recommended by
        reference article authors.

    cgthresh: scalar, optional

        Threshold value of USAN center of gravity, in units of pixels,
        which will lead to a shift between one underlying set of
        approximations/assumptions vs. another.  Default setting of
        cgthresh=None triggers lower level masks.usan() function to
        select a context-appropriate default value (for edge detection
        specifically, default is cgthresh=0.3*radius) as recommended
        by reference article authors.  See masks.usan() source code
        and reference article for further (extremely esoteric) details.

    nshades: scalar, optional

        Total number of distinct integer grayscale levels available in
        image format.  Defaults to 256 (i.e., 2**8), appropriate for an
        8 bit image format such as jpeg.  For image formats with higher
        grayscale color resolution, e.g. such as 12-bit or 16-bit, set
        nshades=4096 or nshades=65536 (i.e., 2**12 or 2**16).

    tlo, thi: scalar, optional

        Low and high thresholds for Canny-style hysteresis thresholding.
        Default values are 0.1 and 0.3, and represent the two thresholds
        as fractions of the maximum edge response for the entire image.
        Set tlo=thi in order to turn off dual thresholding and revert
        effectively to a single threshold scheme.  Set tlo=thi=0 to turn
        off thresholding altogether.  All values must lie on interval
        0 <= tlo <= thi <= 1.

    minmem: scalar, optional

        Specifies minimum pixel membership to be required in each valid
        edge.  E.g., minmem=4 will delete extremely short junk edges
        consisting of only 1, 2, or 3 adjacent pixels, but will keep
        edges with 4 or more pixels.  This is sometimes useful when
        processing an image with texture, e.g., such as a brick wall,
        where small segments of the texture itself, such as the mortar
        between the bricks, may occasionally be misinterpreted by the
        algorithm as extremely short macroscopic edges.  Default is
        set to minmem=1, which effectively leaves this section of the
        processing turned off in normal usage, due to the fact that it
        was not part of the SUSAN algorithm as originally defined by
        the reference article authors.  

    Returns
    -------

    edge: ndarray

        The edge response of the image as calculated by the SUSAN edge
        detection algorithm.
        
    """

    # Get raw edge response; many edges may be wider than one pixel
    thickedge = masks.usan(image, mode='Edge', radius=radius, fptype=fptype,
                           t=t, gfrac=gfrac, nshades=nshades)
    
    # Get angle (in degrees) of normal vector perpendicular to each edge
    edgedir = masks.usan(image, mode='EdgeDir', radius=radius, fptype=fptype,
                         t=t, gfrac=gfrac, cgthresh=cgthresh, nshades=nshades)

    # Round normal vector to one of four cardinal directions: left-right,
    # up-down, diagonal (positive slope), diagonal (negative slope)
    rounddir = cleanedges.roundAngle(edgedir)

    # Thin edges to single pixel width by accepting only those whose USAN
    # response is at a local maximum relative to the local edge gradient
    maxedge = cleanedges.nonMaxSuppEdge(thickedge, rounddir)

    # Apply Canny-style hysteresis thresholding
    htedge = cleanedges.hystThresh(maxedge, tlo, thi)

    # Delete edges consisting of fewer than minmem connected pixels
    edge = cleanedges.minMembership(htedge, minmem=minmem)

    return edge

def susanCorner(image, radius=3.4, fptype=bool, t=25, gfrac=None,
                cgthresh=None, nshades=256):

    """
    Implements SUSAN corner detector algorithm as described in "SUSAN--A
    New Approach to Low Level Image Processing", S. M. Smith and J. M.
    Brady, Technical Report TR95SMSIc (1995), available at:
    http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.24.2763 .
    Alternatively, there is also a slightly abridged version of the
    same reference, with identical title and authors, available at
    International Journal of Computer Vision 23(1): 45-78 (1997).

    image: array_like

        Array containing grayscale image data.  Only grayscale pixel
        values are supported--cannot handle 3-channel color.

    radius: scalar, optional

        Circular footprint radius, in units of pixels; passed down
        as input to masks.circFootprint().  Default is 3.4, as
        recommended in reference.

    fptype: data-type, optional

        Data type of circular footprint; passed to masks.circFootprint().
        Default is bool, since the version of the algorithm originally
        described in the reference did not define the algorithm behavior
        for any other type of footprint (we have trivially extended it
        to cover float type footprints as well, however there is not
        much observable different in the results, as it only affects the
        weighting of the pixels at the edge of the footprint, not the
        central region).

    t: scalar, optional

        Threshold value for color difference required to exclude/include
        a pixel in the USAN region described in the reference article.
        Default is 25 grayscale levels, as suggested by article authors,
        and assumes a grayscale image with a total of 256 distinct
        shades.  As described in eq. 4, this is technically a "soft"
        threshold, rather than a hard cutoff, and moreover it's also
        bidirectional; i.e., a setting of t=25 actually means +/- 25.

    gfrac: scalar, optional

        Fraction of maximum number of USAN pixels to assign to the "g"
        parameter in eq. 3 of the reference article.  Default setting
        of gfrac=None triggers lower level functions to select a
        context-appropriate default value (for corner detection
        specifically, default is gfrac=0.5) as recommended by
        reference article authors.

    cgthresh: scalar, optional

        Threshold value of USAN center of gravity, in units of pixels,
        which will lead to a shift between one underlying set of
        approximations/assumptions vs. another.  Default setting of
        cgthresh=None triggers lower level masks.usan() function to
        select a context-appropriate default value (for corner detection
        specifically, default is cgthresh=0.45*radius) as recommended
        by reference article authors.  See masks.usan() source code
        and reference article for further (extremely esoteric) details.
        For corner detection in particular (less so with edge detection)
        we recommend testing a variety of settings both above and below
        the default as this parameter seems to have a particularly
        strong effect in some cases on signal vs. noise.

    nshades: scalar, optional

        Total number of distinct integer grayscale levels available in
        image format.  Defaults to 256 (i.e., 2**8), appropriate for an
        8 bit image format such as jpeg.  For image formats with higher
        grayscale color resolution, e.g. such as 12-bit or 16-bit, set
        nshades=4096 or nshades=65536 (i.e., 2**12 or 2**16).

    Returns
    -------

    corner: ndarray

        The corner response of the image as calculated by the SUSAN corner
        detection algorithm.
        
    """

    # Get raw corner response; many corners may consist of small clusters of
    # adjacent responsive pixels
    rawcorner = masks.usan(image, mode='Corner', radius=radius,
                           fptype=fptype, t=t, gfrac=gfrac,
                           cgthresh=cgthresh, nshades=nshades)

    # Find maximum corner response within circular USAN footprint (but force
    # footprint type to be bool in this case regardless of user-selected
    # input fptype, because float would make no sense in this context)
    fp = masks.circFootprint(radius=radius, dtype=bool)
    rawmax = filters.maximum_filter(rawcorner, footprint=fp)

    # True corners are those where response is both locally maximum as well
    # as non-zero
    corner = np.where(rawcorner == rawmax, rawcorner, 0)

    return corner
