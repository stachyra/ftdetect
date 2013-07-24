"""
cleanedges.py

Provides functions which enhance/refine/clean up the basic edge detector
output response.

AUTHOR: Andrew L. Stachyra
DATE: 5/17/2013
"""

import math
import numpy as np

def roundAngle(edgedir, units='Degrees'):

    """
    Rounds edge normal vector direction angles to one of four allowed
    values: 0, 45, 90 or 135 degrees, indicating that normal vector
    points horizontal, vertical, or diagonal with either positive or
    negative slope.  Input values which lie within +/- 22.5 degrees of
    an antipode (180, 225, 270, 315, -180, -135, -90, -45) are corrected
    by +/- 180 degrees as necessary.

    Parameters
    ----------

    edgedir: ndarray

        2D array containing estimated edge normal direction vector (e.g.,
        from a grayscale brightness gradient or similar) for each pixel
        in a 2D image.

    units: ('Degrees', 'degrees', 'Radians', 'radians'), optional

        Units of input and output.  Defaults to 'Degrees'.

    Returns
    -------

    rounddir: ndarray

        2D array of directions rounded to either 0, 45, 90 or 135
        degrees (or 0, pi/4, pi/2, and 3pi/4 radians, if units='Radians').
    
    """

    # Calculate degrees to radians conversion factor, if necessary
    if units == 'Degrees' or units == 'degrees':
        k = 1
    elif units == 'Radians' or units == 'radians':
        k = math.pi/180
    else:
        raise ValueError('Units value ' + units + ' not recognized')

    # Initialize output
    rounddir = np.zeros(shape=edgedir.shape)
    
    # Loop over four major compass directions: E/W, NE/SW, N/S, and NW/SE
    for ii in range(4):
        idx = np.zeros(shape=edgedir.shape).astype(bool)
        # Loop within each set of antipodes to group them together
        for ij in range(-1,2):
            iteridx = np.logical_and((edgedir >= k * (ij*180 + ii*45 - 22.5)),
                                     (edgedir < k * (ij*180 + ii*45 + 22.5)))
            idx = np.logical_or(idx, iteridx)
        # Round all angles lying within +/- 22.5 degrees of this value (or of
        # its antipode) to be equal to this value
        rounddir[idx] = k * ii * 45

    return rounddir

def nonMaxSuppEdge(inedge, rounddir, units='Degrees'):

    """
    Erodes "thick edge" raw edge detection response by deleting those
    responses from pixels which are not at a local maximum relative to
    the two neighbors to either side of them along the edge normal
    direction.

    Parameters
    ----------

    inedge: ndarray

        2D array containing raw edge response from an edge detection
        algorithm.  Larger values are presumed to correspond to sharper,
        more distinct edges (e.g., bright areas immediately adjacent to
        dark, or vice versa).

    rounddir: ndarray (must be same size/shape as inedge)

        2D array of edge normal vector angles rounded to either 0, 45,
        90 or 135 degrees (or equivalent in radians), output by
        roundAngle() function.

    units: ('Degrees', 'degrees', 'Radians', 'radians'), optional

        Units of rounddir.  Defaults to 'Degrees'.

    Returns
    -------

    outedge: ndarray

        2D array of "thinned" edge pixels, with non-maximal responses
        deleted.

    """

    # Calculate degrees to radians conversion factor, if necessary
    if units == 'Degrees' or units == 'degrees':
        k = 1
    elif units == 'Radians' or units == 'radians':
        k = math.pi/180
    else:
        raise ValueError('Units value ' + units + ' not recognized')

    # Get image size
    xdim, ydim = inedge.shape

    # Initialize output
    outedge = inedge

    for ii in range(xdim):
        for ij in range(ydim):
            # Edge normal vector points east-west
            if rounddir[ii][ij] == k * 0:
                if ii == 0: # Special case: test pixel is at lower x edge
                    a = False
                else: # Usual case: interior pixel wrt x dimension
                    a = (inedge[ii][ij] < inedge[ii-1][ij])
                if ii == (xdim-1): # Special case: upper x edge
                    b = False
                else: # Usual case: interior pixel wrt x dimension
                    b = (inedge[ii][ij] < inedge[ii+1][ij])
            # Edge normal points along positive slope diagonal (SW/NE)
            elif rounddir[ii][ij] == k * 45:
                if ii == 0 or ij == 0:
                    a = False
                else:
                    a = (inedge[ii][ij] < inedge[ii-1][ij-1])
                if ii == (xdim-1) or ij == (ydim-1):
                    b = False
                else:
                    b = (inedge[ii][ij] < inedge[ii+1][ij+1])
            # Edge normal points north-south
            elif rounddir[ii][ij] == k * 90:
                if ij == 0:
                    a = False
                else:
                    a = (inedge[ii][ij] < inedge[ii][ij-1])
                if ij == (ydim-1):
                    b = False
                else:
                    b = (inedge[ii][ij] < inedge[ii][ij+1])
            # Edge normal points along negative slope diagonal (NW/SE)
            elif rounddir[ii][ij] == k * 135:
                if ii == 0 or ij == (ydim-1):
                    a = False
                else:
                    a = (inedge[ii][ij] < inedge[ii-1][ij+1])
                if ii == (xdim-1) or ij == 0:
                    b = False
                else:
                    b = (inedge[ii][ij] < inedge[ii+1][ij-1])
            else:
                errmsg = 'Input either contains unrounded direction ' + \
                         'value (' + str(rounddir[ii][ij]) + \
                         ') at location [' + str(ii) + '][' + str(ij) + \
                         '] or else units (Degrees/' + \
                         'Radians) has been set incorrectly'
                raise ValueError(errmsg)
            # Test whether or not this particular pixel should be suppressed
            if a or b:
                outedge[ii][ij] = 0

    return outedge

def hystThresh(inedge, tlo=0.1, thi=0.3):

    """
    Performs Canny style "hysteresis" or dual-valued response
    thresholding for edge detection algorithms.  Pixels with edge
    responses that fall above the upper threshold (call them "group A"
    for brevity) are automatically included in the output, while pixels
    that fall below the lower threshold ("group C") are automatically
    excluded.  Pixels whose edge response falls between the two
    thresholds ("group B") will be upgraded to the group A list if they
    lie either within an 8 pixel surrounding neighborhood of a pixel
    that was already in group A from the start, or alternatively, if
    they lie within an 8 pixel neighborhood of another group B pixel
    that had previously been upgraded to group A.

    Parameters
    ----------

    inedge: ndarray

        2D array of "thin" edges, e.g. as returned by nonMaxSuppEdge().

    tlo: scalar, optional

        Lower threshold value, as a fraction of maximum edge response
        (must be chosen between 0 and 1, and less than or equal to thi).
        Defaults to 0.1.

    thi: scalar, optional

        Upper threshold value, as a fraction of maximum edge response
        (must be chosen between 0 and 1, and greater than or equal to
        tlo).  Defaults to 0.3.

    Returns
    -------

    outedge: ndarray

        2D array of edge pixels with noise/intermittent responses
        hopefully removed or substantially deleted.

    """

    # Catch bad inputs
    if tlo > thi:
        raise ValueError('tlo must be set less than or equal to thi')
    if tlo < 0 or tlo > 1 or thi < 0 or thi > 1:
        raise ValueError('tlo and/or thi values of less than 0 or ' + \
                         'greater than 1 are not meaningful')

    # 8 pixel neighborhood surrounding [0, 0]
    x = [-1,  0,  1, -1,  1, -1,  0,  1]
    y = [-1, -1, -1,  0,  0,  1,  1,  1]

    # Get dimensions
    xdim, ydim = inedge.shape

    # Pixels above the high threshold are automatically treated as true edges
    outedge = np.where(inedge > (thi * inedge.max()), inedge, 0)
    # Pixels above the low threshold will be added later if they prove to be
    # adjacent to another pixel which has been included in the outedge list
    lopass = np.where(inedge > (tlo * inedge.max()), inedge, 0)

    # Create an initial list of known edge pixels around which to search for
    # adjacent pixels which passed the low threshold but failed the upper one
    prevx, prevy = outedge.nonzero()

    # If the previous round of testing found no new pixels to transfer from
    # the lopass list to the edge list, then end the search--we're done
    while len(prevx) != 0:
        newx, newy = [], []
        # Loop over new edge pixels discovered on previous iteration
        for ii in range(len(prevx)):
            # Loop through 8 pixel neighborhood
            for ij in range(len(x)):
                xidx = prevx[ii] + x[ij]
                yidx = prevy[ii] + y[ij]
                # If pixel index falls within image boundary...
                if xidx >= 0 and xidx < xdim and yidx >= 0 and yidx < ydim:
                    # ...and pixel is on the lopass list but has not yet been
                    # added to the edge list...
                    if lopass[xidx][yidx] and not outedge[xidx][yidx]:
                        # Transfer to edge list
                        outedge[xidx][yidx] = lopass[xidx][yidx]
                        # Keep track of indices for next loop iteration
                        newx.append(xidx)
                        newy.append(yidx)
        # Update for next iteration
        prevx = newx
        prevy = newy

    return outedge

def listEdge(edge):
    
    """
    Given a 2D array of edge detection responses, groups edge pixels
    which are "geometrically connected" to one another (i.e., adjacent
    to other pixels that also have a non-zero edge detector response)
    together into discrete clusters.  Outputs a list providing
    membership of each cluster (i.e., the positional indices of
    constituent pixels) as well as the total number of pixels in each
    cluster.

    Parameters
    ----------

    edge: ndarray

        2D array containing edge response (possibly already pre-
        processed) from an edge detection algorithm.

    Returns
    -------

    vidx, hidx: ndarray

        Pair of 2D ragged-end arrays giving pixel membership within each
        edge cluster.  First axis loops over clusters and second index
        loops over individual pixel membership within each cluster; e.g.,
        (vidx[ii][ij], hidx[ii][ij]) will give the indices into the input
        edge array of the ij'th pixel within the ii'th cluster; i.e.
        edge[vidx[ii][ij]][hidx[ii][ij]].  Since each cluster is, in
        general, a different size than the others, the dimensions along
        the ij direction in general will be uneven (i.e., "ragged-ended").

    nmem: ndarray

        1D array giving number of member pixels within each cluster.
        
    """
    
    # 8 pixel neighborhood surrounding [0, 0]
    x = np.asarray([-1,  0,  1, -1,  1, -1,  0,  1], dtype=int)
    y = np.asarray([-1, -1, -1,  0,  0,  1,  1,  1], dtype=int)

    # Initialize output variables
    vidx, hidx = [], []
    nmem = np.asarray([], dtype=int)

    # Initial master list of all edge pixels
    v, h = edge.nonzero()
    # A value of True here indicates a pixel on the initial master list which
    # has not yet been transfered to its position in the output list.
    unfound = np.ones(len(v), dtype=bool)
    # These variables are used to build/aggregate member pixels together into
    # a self-contained edge in which all member pixels are nearest-neighbors
    # of one another, but are not directly adjacent to pixels of other edges
    vtmp, htmp = np.asarray([], dtype=int), np.asarray([], dtype=int)
    # "True" indicates a pixel in the vtmp/htmp list which hasn't yet had its
    # 8-pixel neighborhood traversed to search for other nearest neighbors
    untrav = np.asarray([], dtype=bool)
    # Find indices of all pixels still remaining on the initial master list
    ufidx = unfound.nonzero()[0]
    while len(ufidx):
        # Take the first pixel still remaining on the master list and add
        # it to the vtmp/htmp list...
        vtmp = np.append(vtmp, v[ufidx[0]])
        htmp = np.append(htmp, h[ufidx[0]])
        # ...mark its own 8 pixel neighborhood as not yet traversed...
        untrav = np.append(untrav, True)
        # ...and cross it off the initial master list
        unfound[ufidx[0]] = False
        # Traverse neighborhoods of pixels on the vtmp/htmp list which have
        # not been traversed yet
        utidx = untrav.nonzero()[0]
        while len(utidx):
            # Generate a list of neighbor pixels (some of these will not be
            # valid indices into inedge if vtmp/htmp falls at edge of array)
            vtest = vtmp[utidx[0]] + x
            htest = htmp[utidx[0]] + y
            for ii in range(len(vtest)):
                # Find nearest neighbor pixels which are valid edge pixels
                # from the initial master list, and also haven't been crossed
                # off the master list yet
                addidx = np.where((v == vtest[ii]) & (h == htest[ii]) & \
                                  unfound)[0]
                if len(addidx):
                    # Add any new finds to vtmp/htmp, mark the new pixel's
                    # neighborhood as untraversed, and cross it off master list
                    vtmp = np.append(vtmp, v[addidx])
                    htmp = np.append(htmp, h[addidx])
                    untrav = np.append(untrav, True)
                    unfound[addidx] = False
            # Mark the pixel on the vtmp/htmp list that we have just completed
            # as fully traversed, and recompute the untraversed list
            untrav[utidx[0]] = False
            utidx = untrav.nonzero()[0]
        # Reaching this point means that there are no pixels remaining in the
        # current vtmp/htmp list whose neighbor pixels haven't been explored.
        # Thus, this edge is complete--add it to the final output list and
        # re-zero the vtmp/htmp list to look for other unconnected edges.
        vidx.append(vtmp)
        hidx.append(htmp)
        nmem = np.append(nmem, len(vtmp))
        vtmp, htmp = np.asarray([], dtype=int), np.asarray([], dtype=int)
        untrav = np.asarray([], dtype=bool)
        ufidx = unfound.nonzero()[0]

    # Cast as numpy arrays to maintain consistency with nmem output, which
    # must be numpy array rather than list in order to use argsort() method
    vidx = np.asarray(vidx)
    hidx = np.asarray(hidx)
    
    # Sort the list of discontinuous/discrete edges in reverse order of size
    sidx = nmem.argsort()[::-1]
    nmem = nmem[sidx]
    vidx = vidx[sidx]
    hidx = hidx[sidx]

    return vidx, hidx, nmem
            
def minMembership(inedge, minmem):

    """
    Given a 2D array of edge detection responses, calls listEdge() to
    group edge pixels into geometrically isolated clusters, and then
    deletes clusters containing fewer than minmem pixels.

    Parameters
    ----------

    inedge: ndarray

        2D array containing edge response (possibly already pre-
        processed) from an edge detection algorithm.

    minmem: scalar

        Minimum number of pixels required in each cluster.  Clusters
        containing fewer pixels than this threshold are deleted from
        the output.

    Returns
    -------

    outedge: ndarray

        2D array containing the cleaned edge response after having small
        clusters removed.

    """

    if minmem <= 1:
        return inedge

    # Get a master list of discrete, unconnected edges, containing indices
    # of the member pixels (vidx, hidx) comprising each edge, as well as
    # total number of member pixels (nmem) within each edge 
    vidx, hidx, nmem = listEdge(inedge)

    outedge = np.zeros(shape=inedge.shape)
    
    for ii in range(len(nmem)):
        # For edges with a sufficiently large number of members...
        if nmem[ii] >= minmem:
            for ij in range(len(vidx[ii])):
                # ...copy their values to the output variable
                outedge[vidx[ii][ij]][hidx[ii][ij]] = \
                    inedge[vidx[ii][ij]][hidx[ii][ij]]

    return outedge
