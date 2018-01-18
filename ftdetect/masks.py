"""
masks.py

Provides functions which create specialized footprints or masks which are
subsequently fed as inputs to functions in scipy.ndimage.filters module.

AUTHOR: Andrew L. Stachyra
DATE: 5/17/2013
"""

import numpy as np
from scipy.ndimage import filters
import math

def getDeriv(image, weights=[3./16, 10./16, 3./16], axes=[0], mode="reflect",
             cval=0.0):

    """
    Calculates a first or second derivative on a multi-dimensional image
    array by the method of finite differences using a 3X3 stencil.
    Images in principle need not necessarily be 2D; i.e., 3D tomographic
    images should work also, however, "caveat emptor" as this latter
    functionality has not yet actually been tested fully.
    
    Parameters
    ----------

    image: array_like

        Array containing grayscale image data.  Only grayscale pixel
        values are supported--cannot handle 3-channel color.

    weights: array, optional

        1D sequence of three numbers representing the type of finite
        differences derivative (Prewitt, Sobel, Scharr, etc.) to compute.
        Defaults to [3./16, 10./16, 3./16], i.e., Scharr type.  It is
        recommended that this sequence should be normalized so that all
        components sum to 1.  If not, the function will still return a
        result, however, cross derivative (dxdy type) results will be
        scaled incorrectly with respect to 1st derivatives and non-cross
        2nd derivatives (e.g., dxdx, dydy).

    axes: scalar or array_like, optional

        Either a single value (1st derivative case) or two values (2nd
        derivative) indicating axes along which derivatives are to be
        taken.  Examples:

            axes=0         1st derivative, x-axis
            axes=[0]       Also indicates 1st derivative, x-axis
            axes=(1, 1)    2nd derivative, y-axis (i.e, dydy type)
            axes=[0, 2]    2nd derivative, x- and z-axes (i.e., dxdz,
                               assuming a tomographic style image with
                               three axes)

    mode: ('reflect', 'constant', 'nearest', 'mirror', 'wrap')

        Controls how edge pixels in the input image are treated.  See
        scipy.ndimage.filters.correlate1d() for details.

    cval: scalar, optional

        Only meaningful if mode is set to 'constant'.  See
        scipy.ndimage.filters.correlate1d() for details.

    Returns
    -------

    output: ndarray

        An estimate of first or second partial derivative with respect
        to image brightness at each pixel.

    """

    """Check and/or condition the input variables"""
    # Force treatment as float numpy array to avoid rounding errors later
    image = np.asarray(image, dtype=float)

    wmsg = 'weights input variable must be an array or list with ' + \
           'exactly three elements'
    try:
        nw = len(weights) # Fails if weights is not iterable type
    except:
        raise TypeError(wmsg)
    if nw != 3: # If weights is iterable, but still not correct length...
        raise ValueError(wmsg)

    """Set appropriate weights, and slightly reconfigure axes specification"""
    try: # Assume axes input value is iterable
        nx = len(axes) # Will raise a TypeError if axes is not iterable
    except TypeError:
        # First derivative
        wght = [-0.5, 0, 0.5]
        myaxes = [axes] # Force myaxes to be iterable list containing one item
        nx = 0

    # Skip the rest, if axes input value was scalar (i.e., not iterable)
    if nx == 0:
        pass
    # Alternative first derivative, if axes input is iterable
    elif nx == 1:
        wght = [-0.5, 0, 0.5]
        myaxes = axes
    elif nx == 2:
        # Second derivative, along same axis twice
        if axes[0] == axes[1]:
            wght = [1.0, -2.0, 1.0]
            myaxes = [axes[0]]
        # Second derivative, along orthogonal axes
        else:
            wght = [-0.5, 0, 0.5]
            myaxes = axes
    else:
        raise ValueError('Too many axes: 3rd derivatives and higher are ' +
                         'not yet supported')
    
    """Compute the derivative!!!"""
    for ii in myaxes:
        # Use fast compiled code from scipy.ndimage._nd_image.pyd
        output = filters.correlate1d(image, wght, ii, mode=mode, cval=cval)

    """Apply smoothing weights (Prewitt, Sobel, Scharr, or whatever the
    user has selected) to all remaining axes"""
    # Get a list of all other axes.  For 2D images, this will produce either
    # a null list (in the dxdy case) or at most one other axis.  For 3D
    # images (e.g., such as a tomographic image), there will be either one
    # or two other axes.
    otheraxes = [ii for ii in range(image.ndim) if ii not in myaxes]
    for ii in otheraxes:
        output = filters.correlate1d(output, weights, ii, mode=mode, cval=cval)

    return output

def circFootprint(radius, method='Area', npoints=10, dtype=bool):

    """
    Generates a circular footprint based on the input radius.  Intended
    for use with usan() function.

    Parameters
    ----------

    radius: scalar

        Radius of the circular footprint, in pixels.

    method: ('Center', 'center', 'Area', 'area'), optional

        Method by which to compute the footprint.  If method=='Center'
        or method=='center', each pixel is tested for membership in
        the footprint based upon whether a single point at the center
        of the pixel falls within the radius.  Depending upon the
        dtype selected, each pixel will assume either of two values:
        (True, False), (0, 1), or (0., 1.).  If method=='Area' or
        method=='area', a square subgrid of size npoints X npoints
        is superimposed upon each pixel, and membership is determined
        by the total number of subgrid points (representing the fraction
        of pixel area) that falls within the radius.  Depending upon the
        dtype selected, each pixel will assume values of either (True,
        False), (0, 1), or a sliding scale value between 0. and 1.

    npoints: scalar, optional

        Number of points to use in subgrid when method='Area' or
        method='area' is selected.  See method input variable above
        for further discussion.

    dtype: data-type, optional

        Data type to use for output.  Note that float is only really
        meaningful if method='Area' or method='area' is selected.  If
        dtype is set to float but method='Center' or method='center',
        then the pixels in the footprint will be assigned floating
        point values of 0. or 1., but will be unable to assume any
        values in between.  See method input variable above for further
        discussion.

    Returns
    -------

    footprint: ndarray

        A square array defining a circular footprint.  Values of zero
        indicate a pixel is not within the footprint radius, non-zero
        values indicate either absolute membership (bool or int) or
        degree of partial membership (float).

    """

    # Determine whether each test pixel falls within the circular mask based
    # on whether the pixel's center falls within the radius
    if method == 'Center' or method == 'center':
        halfext = int(math.floor(radius))
        ones = np.ones((2*halfext+1, 2*halfext+1), dtype=dtype)
        zeros = np.zeros((2*halfext+1, 2*halfext+1), dtype=dtype)
        # Make a square trial grid just large enough to contain radius
        v, h = np.ogrid[-halfext:(halfext+1), -halfext:(halfext+1)]
        # footprint consists of any pixel within the radius
        footprint = np.where(v**2 + h**2 <= radius**2, ones, zeros)
    # Determine each pixel's membership in circular mask based on total
    # percentage of pixel area that falls within the radius
    elif method == 'Area' or method == 'area':
        step = 1./npoints
        # Create a subgrid of size (npoints, npoints) within each pixel
        v, h = np.ogrid[(-0.5+step/2):(0.5+step/2):step,
                        (-0.5+step/2):(0.5+step/2):step]
        halfext = int(math.ceil(radius-0.5))
        fpfloat = np.zeros((2*halfext+1, 2*halfext+1), dtype=float)
        # Loop through each pixel in an implicit trial grid
        for ii in range(-halfext, (halfext+1)):
            for ij in range(-halfext, (halfext+1)):
                # Values of True signify points within the footprint radius
                subgrid = ((v-ii)**2 + (h-ij)**2 <= radius**2)
                # Total area of (ii,ij)'th pixel is proportional to total
                # fraction of True values in the subgrid
                fpfloat[(ii+halfext),
                        (ij+halfext)] = float(sum(sum(subgrid)))/(npoints**2)        
        ones = np.ones((2*halfext+1, 2*halfext+1), dtype=dtype)
        zeros = np.zeros((2*halfext+1, 2*halfext+1), dtype=dtype)
        # For dtypes that aren't capable of representing fpfloat properly,
        # create a footprint by rounding the values in fpfloat up or down...
        if dtype == bool or dtype == int:
            footprint = np.where(fpfloat >= 0.5, ones, zeros)
        # ...but otherwise, just use fpfloat directly
        else:
            footprint = fpfloat.astype(dtype)
        # If trial grid accidentally contains a one pixel wide perimeter band
        # which doesn't fall within the circular footprint, then trim it off
        if not footprint[0,halfext]:
            footprint = footprint[1:(2*halfext),1:(2*halfext)]       
    else:
        raise ValueError('Method ' + str(method) + ' not supported')

    return footprint

def usan(image, mode='Edge', radius=3.4, fptype=bool, t=25, gfrac=None,
         cgthresh=None, nshades=256):
    
    """
    Calculates raw edge or corner response, based upon an algorithm
    described in: "SUSAN--A New Approach to Low Level Image Processing",
    S. M. Smith and J. M. Brady, Technical Report TR95SMSIc (1995),
    available at:
    http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.24.2763 .
    Alternatively, there is also a slightly abridged version of the
    same reference, with identical title and authors, available at
    International Journal of Computer Vision, 23(1), 45-78 (1997).

    Parameters
    ----------

    image: array_like

        Array containing grayscale image data.  Only grayscale pixel
        values are supported--cannot handle 3-channel color.

    mode: ('Edge', 'edge', 'EdgeDir', 'edgedir', 'Corner', 'corner'),
              optional

        Chooses usage mode.  If mode=='Edge' or mode=='edge' or mode==
        'Corner' or mode=='corner' then the output will be either an
        edge or corner response.  If mode=='EdgeDir' or mode=='edgedir'
        then the output gives an estimate of the angle (in degrees)
        of the edge normal unit vector.  In general, a value for the
        edge normal angle will be generated for all pixels, however it
        isn't usually meaningful unless the edge response is non-zero.

    radius: scalar, optional

        Circular footprint radius, in units of pixels; passed through
        as input to circFootprint().  Default is 3.4, as recommended
        in reference.

    fptype: data-type, optional

        Data type of circular footprint; passed to circFootprint().
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
        parameter in eq. 3 of the reference article.  Adhering to
        recommendation of article authors, it defaults to 0.75 if user
        specifies mode='Edge' or mode='edge', and 0.5 if user specifies
        mode='Corner' or mode='corner'.

    cgthesh: scaler, optional

        Threshold value of USAN center of gravity, in units of pixels,
        which will lead to a shift between one underlying set of
        approximations/assumptions vs. another.  Defaults to 0.3*radius
        (i.e., 1.02 pixels, assuming a default USAN footprint radius of
        3.4 pixels, which is consistent with author's nominal suggested
        setting of about 1 pixel) if mode='Edge' or mode= 'edge', and
        defaults to 0.45*radius if mode='Corner' or mode='corner' (the
        article authors did not suggest a default setting for corner
        mode, so this setting is based on trial-and-error).  See
        reference article for more details.

    nshades: scalar, optional

        Total number of distinct integer grayscale levels available in
        image format.  Defaults to 256 (i.e., 2**8), appropriate for an
        8 bit image format such as jpeg.  For image formats with higher
        grayscale color resolution, e.g. such as 12-bit or 16-bit, set
        nshades=4096 or nshades=65536 (i.e., 2**12 or 2**16).

    Returns
    -------

    R or edgedir: ndarray

        Context sensitive value, depending on the value of the mode input
        variable.  If mode='Edge' or mode='edge' or mode='Corner' or
        mode='corner' then the return array holds the edge or corner
        response as described by eq. 3 in the reference article.  If
        mode='EdgeDir' or mode='edgedir' then the return array holds
        the raw estimated edge normal direction.  In general this value
        will only be meaningful if the edge response of the same pixel is
        nonzero.

    """

    """ Condition input variables """
    # Make sure that methods of type ndarray are available for use later
    image = np.asarray(image)
    
    # Assign default values suggested in reference, if user doesn't override
    if gfrac is None:
        if mode == 'Edge' or mode == 'edge':
            gfrac = 0.75
        elif mode == 'Corner' or mode == 'corner':
            gfrac = 0.5

    # Assign default value based on reference for 'EdgeDir' mode, and based
    # on trial-and-error experimentation, for 'Corner' mode
    if cgthresh is None:
        if mode == 'EdgeDir' or mode == 'edgedir':
            cgthresh = 0.3 * radius
        elif mode == 'Corner' or mode == 'corner':
            cgthresh = 0.45 * radius

    """ Create a lookup table, as recommended in the reference """
    idx = np.arange(nshades)
    explookup = np.zeros(len(idx))
    for ii in range(len(idx)):
        # From eq. 4 in the reference
        explookup[ii] = math.exp(-(idx[ii].astype(float)/t)**6)

    """ Set up USAN circular footprint and several related variables """
    # Get the circular USAN mask
    fp = circFootprint(radius, dtype=fptype)
    # For the dtype=bool case, the areawgt variable essentially turns into
    # a flattened array of ones, but for the dtype=float case, pixels near
    # the edge of the circular footprint will be reweighted according to what
    # fraction of their error falls within the footprint radius
    areawgt = fp[(fp != 0)]
    # Force fp to be all zeros and ones, to comport with expectations of
    # scipy.ndimage.filters.generic_filter()
    fp = fp.astype(bool).astype(int)
    # Define arrays consisting of horizontal and vertical offsets between
    # the nucleus and each of the surrounding pixels in the circular mask
    halfext = (fp.shape[1] - 1)/2
    xdiff, ydiff = np.mgrid[-halfext:(halfext+1), -halfext:(halfext+1)]
    xdiff = xdiff[(fp != 0)]
    ydiff = ydiff[(fp != 0)]    

    """ Define a function which will be called iteratively upon every pixel
    in the image, using scipy.ndimage.filters.generic_filter() """
    def filterfunc(maskout, areawgt, xdiff, ydiff, radius, mode, t, explookup,
                   gfrac, cgthresh, npoints=10):

        """
        Internal function to usan() which gets passed down through to
        scipy.ndimage.filters.generic_filters() to perform the actual
        work of filtering.  Gets called once for each pixel in image.

        maskout: flattened ndarray

            Contains values from surrounding pixels which fell within
            footprint at each pixel.

        areawgt: flattened ndarray

            Same size as maskout; if fptype=float in usan() input, gives
            fraction of each pixel area which fell within the circular
            footprint radius.  Otherwise, if fptype=bool or fptype=int,
            it's simply an array of ones.
            
        xdiff, ydiff: flattened ndarray

            (x, y) offset between each pixel and nucleus (center pixel
            of circular footprint).

        radius, mode, t, gfrac, cgthresh

            Straight pass-through of inputs to usan() function.

        explookup: ndarray

            1D array containing lookup table for eq. 4 in reference
            article.  Size should match nshades input argument to usan()
            function.

        npoints: scalar, optional

            Specifies number of subpixel lattice points to use per pixel
            when computing USAN contiguity if mode='Corner' or
            mode='corner'.

        Returns
        -------

            R or edgedir: scalar

            Context sensitive value, depending on the value of the mode
            input variable.  If mode='Edge' or mode='edge' or mode='Corner'
            or mode='corner' then the return value holds the edge or corner
            response as described by eq. 3 in the reference article.  If
            mode='EdgeDir' or mode='edgedir' then the return value holds
            the raw estimated edge normal direction.  In general this value
            will only be meaningful if the edge response of the same pixel
            is nonzero.
        
        """

        """ Condition inputs and pre-compute expressions which are required
        in multiple locations below """
        # Total number of pixels in mask
        ntot = len(maskout) - 1
        # Index and intensity of center pixel (i.e., the nucleus)
        ctridx = ntot//2
        nucleus = maskout[ctridx]
        # Delete data of center pixel in family of arrays with same dimension
        maskout = np.delete(maskout, ctridx)
        areawgt = np.delete(areawgt, ctridx)
        xdiff = np.delete(xdiff, ctridx)
        ydiff = np.delete(ydiff, ctridx)
        # Calculate color/grayscale shade difference between nucleus and all
        # other surrounding pixels in the footprint.  Cast type back to int
        # in order to index lookup table--will definitely be handed off as
        # a float otherwise (see explanatory note below for reason behind
        # convoluted type casting flip-flop in the first place).
        graydiff = np.abs(maskout-nucleus*np.ones(len(maskout))).astype(int)
        # Calculate c as described in eq. 4 in the reference.
        c = explookup[graydiff]
        # Reduces to eq. 2 in reference, if areawgt values are all 1
        n = (areawgt * c).sum()
        # Total number of pixels in circular mask
        nmax = areawgt.sum().astype(float)

        """ Compute an appropriate response function for each usage mode """
        if mode == 'Edge' or mode == 'edge':
            # Eq. 3 in reference
            R = gfrac*nmax - n
            if R<0: R=0.
            return R
        elif mode == 'EdgeDir' or mode == 'edgedir':
            denom = (areawgt * c).sum()
            # Usual case
            if denom:
                # Calculate center of gravity, using eq. 5 in reference
                xcg = ((xdiff * areawgt * c).sum()) / denom
                ycg = ((ydiff * areawgt * c).sum()) / denom
            # Divide-by-zero case, which can arise when a single noisy
            # pixels is surrounded by many others of dissimilar brightness
            else:
                xcg, ycg = 0, 0
            cgdist = math.sqrt(xcg**2 + ycg**2)
            # The so-called "inter-pixel" case mentioned in the reference
            if n >= (2*radius) and cgdist >= cgthresh:
                # Compute angle associated with edge normal direction unit
                # vector (i.e., the actual unit vector itself, had we needed
                # to calculate it explicitly in the code, would have been
                # (cos(edgedir), sin(edgedir))).  Due to the way the USAN
                # concept is defined by the reference authors, the edge
                # direction is NOT a gradient pointing from lighter to darker
                # (or vice versa) regions as is commonly the case with other
                # types of edge finding algorithms.  Rather, it always points
                # perpendicularly away from the edge, no matter which side of
                # the edge (lighter or darker) the pixel is on.  Thus, as the
                # USAN circular mask moves across an edge, the edgedir as it
                # is defined in the inter-pixel case usually tends to flip
                # very suddenly by 180 degrees.  For edgedir values falling
                # between 90 and 270 degrees, we will subtract 180 degrees
                # at a later stage of processing in order to map them onto
                # the interval -90 to +90 degrees, which is all that we have
                # available anyway for the intra-pixel case (see below). 
                edgedir = math.atan2(ycg, xcg) * 180 / math.pi
            # The "intra-pixel" case; see reference for description
            else:
                xvar = (xdiff * xdiff * areawgt * c).sum()    # Eq. 6
                yvar = (ydiff * ydiff * areawgt * c).sum()    # Eq. 7
                xycovar = (xdiff * ydiff * areawgt * c).sum() # Eq. 8               
                # Compute edge normal direction.  The xvar and yvar quantities
                # are essentially weighted variances, and are therefore
                # positive definite.  If the (x, y) content of the USAN
                # is positively covariant (i.e., lies along a positively
                # sloped line) then atan2(yvar, xvar)*180/pi gives an angle
                # parallel to the edge and between 0 and 90 degrees, while 
                # (atan2(yvar, xvar)*180/pi - 90) gives the desired edge
                # normal direction (i.e., an angle perpendicular to the edge
                # and guaranteed to lie between -90 and 0 degrees).  On the
                # other hand, if the USAN is negatively covariant, then
                # atan2(yvar, xvar)*180/pi instead gives an angle which is
                # mirror flipped about the y-axis compared to the true edge
                # line.  I.e., say the true edge lies parallel to some angle
                # which we shall define as (90 + theta), with
                # 90 >= theta >= 0, (by definition therefore giving the edge
                # line a negative slope) then atan2(yvar, xvar)*pi/2
                # returns the value (90 - theta), which is mirror-flipped
                # about the y-axis relative to the true value of
                # (90 + theta).  As a consequence of the way that we defined
                # the true edge parallel direction (90 + theta), theta itself
                # turns out to be just the edge normal direction that we had
                # wanted to find, and thus, solving for theta, we get
                # theta = -(atan2(yvar, xvar)*180/pi - 90).  I.e., its's the
                # same as for the positive covariance case, except for a sign
                # flip.  Note however that there is one key difference
                # between this case and the so-called the "inter-pixel" case
                # above: the angles in this "intra-pixel" case are
                # pre-constrained to fall only between -90 and +90 degrees,
                # whereas "inter-pixel" angles may fall anywhere from -180
                # to +180.
                edgedir = math.atan2(yvar, xvar) * 180 / math.pi - 90 
                if xycovar < 0:
                    edgedir = -edgedir
            return edgedir
        elif mode == 'Corner' or mode == 'corner':
            # Eq. 3, but with an alternative default setting (as compared to
            # 'Edge' mode) for the gfrac parameter
            R = gfrac*nmax - n
            if R<0: R=0.
            # Do false corner suppression, but only if there appears to be
            # a genuine non-zero corner response
            if R>0:
                denom = (areawgt * c).sum()
                # Usual case
                if denom:
                    # Calculate center of gravity, using eq. 5 in reference
                    xcg = ((xdiff * areawgt * c).sum()) / denom
                    ycg = ((ydiff * areawgt * c).sum()) / denom
                # Divide-by-zero case, which can arise when a single noisy
                # pixels is surrounded by many others of dissimilar brightness
                else:
                    xcg, ycg = 0, 0
                cgdist = math.sqrt(xcg**2 + ycg**2)
                # False corner check #1: CG is too close to nucleus 
                if cgdist < cgthresh:
                    R=0
                else:
                    # CG vector direction
                    theta = math.atan2(ycg.astype(float), xcg.astype(float))
                    # Calculate nearest pixel locations of a bunch of sub-
                    # pixel-spaced points on a line along the CG direction
                    for ii in np.arange(0, radius, 1./npoints):
                        xtest = int(round(ii * math.cos(theta)))
                        ytest = int(round(ii * math.sin(theta)))
                        for ij in range(len(xdiff)):
                            # Find corresponding index ij in footprint data
                            if xtest == xdiff[ij] and ytest == ydiff[ij]:
                                # False corner check #2: non-contiguous USAN
                                if areawgt[ij] == 1 and graydiff[ij] > t:
                                    R=0
                                    break
                        if R == 0:
                            break
            return R
        else:
            raise ValueError('Mode ' + str(mode) + ' not recognized')
    
    """ Finally, perform the USAN filter operation! """
    # Note that image must be cast as float in order to force
    # filters.generic_filter() to return the output type as float (which
    # is what we usually want).  However, the image itself is intrinisically
    # type int, and the journal reference recommends using a lookup table to
    # compare the relatively limited number (nshades=256) of different
    # possible pixel grayscale color values against one another.  This means
    # that as soon as program control drops down into the filterfunc()
    # (defined above), the color-related variables must be cast right back
    # to int again in order to be able to index the lookup table properly.
    # It's convoluted!
    extraarg = (areawgt, xdiff, ydiff, radius, mode, t, explookup, gfrac,
                cgthresh)
    return filters.generic_filter(image.astype(float), filterfunc,
                                  footprint=fp, extra_arguments=extraarg)
