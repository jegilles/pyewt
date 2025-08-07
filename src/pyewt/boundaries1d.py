import numpy as np
import scipy as scipy
import matplotlib.pyplot as plt
from scipy.special import ive
from scipy.special import erfinv, erf
from sklearn.cluster import KMeans
from numpy.polynomial import polynomial as P
from scipy import signal
from .usefullfunc import round_away


def boundaries_detect(f,params):

    """ Detect meaningful 1D boundaries

    Parameters
    ----------
    f : nparray
        vector containing the spectrum
    params : Dictionary
        must be set properly accordingly to the options described
        above 
        
    Returns
    -------
    bounds - nparray
        Updated list of boundaries
    presig - nparray
        Preprocessed spectrum used for the detection

    Notes
    -----
    This function complete the list of detected boundaries
    to make sure the minimum amount of modes is NT. If that 
    minimum is already reached then the function returns the 
    original set of boundaries. If completion is needed, then 
    the high frequencies support is evenly subdivided to provide
    the extra boundaries.

    If params["log"] is set to True, then the detection is performed
    on the logarithm of f, otherwise it will be on f itself.

    Two types of preprocessing are available to "clean" the spectrum:
    
    - A global trend remover method is set via params["globtrend"]:
        - "none" : does nothing, returns f
        - "powerlaw" : uses a power law to estimate the trend, returns 
            f - powerlaw(f)
        - "polylaw": uses a polynomial interpolation of degree specified
            by params["degree"] to estimate the trend, returns
            f - polynomial(f)
        - "morpho" : uses morphological operators, returns
            f - (opening(f)+closing(f))/2
        - "tophat" : uses morphological operators, returns
            f - opening(f)
        - "opening" : returns
            opening(f)

    - A regularization method is set via params["reg"]:
        - "none" : does nothing, returns f
        - "gaussian" : convolve f with a Gaussian of length and standard
            deviation given by params["lengthfilter"] and 
            params["sigmafilter"], respectively
        - "average": convolve f with a constant filter of length given 
            by params["lengthfilter"]
        - "closing" : applies the morphological closing operator of length 
            given by params["lengthfilter"]

    The wanted boundary detection method must be set in params["detect"].
    The available options are:
    - "localmax" : select the mid-point between consecutive maxima.
        params["N"] must be set to the expected number of modes.
    - "localmaxmin" : select the lowest minima between consecutive 
        maxima of the preprocessed spectrum.
        params["N"] must be set to the expected number of modes.
    - "localmaxminf" : select the lowest minima between consecutive 
        maxima where the minima are detected on the original spectrum
        instead of the preprocessed one.
        params["N"] must be set to the expected number of modes.
    - "adaptivereg" : this method re-adjust a set of provided initial
        boundaries based on the actual lowest minima of the preprocessed
        spectrum
        params["InitBounds"] must be set with a vector of initial 
        boundaries
    - "adaptive" : this method re-adjust a set of provided initial
        boundaries based on the actual lowest minima of the original
        spectrum
        params["InitBounds"] must be set with a vector of initial 
        boundaries
    - "scalespace" : uses the scale-space detection technique to 
        automatically find the boundaries (including their number)
        params["typeDetect"] must be set to the wanted classification
        method. The available options are:
        - "otsu" : uses Otsu's technique
        - "halfnormal" : uses a half-normal law to model the problem
        - "empiricallaw" : uses the data itself to build a model of
        the problem
        - "mean" : threshold is fixed as the mean of the data
        - "kmeans" : uses kmeans to classify

    For the automatic detection methods, a completion step is available 
    is the number of detected modes is lower than an expected value params["N]. 
    If needed, the last high frequency support is evenly splitted. This 
    step is performed if params["Completion"] is set to True.

    Author: Jerome Gilles
    Institution: San Diego State University
    Version: 1.0 (12/18/2024)
    """

    # apply the log if needed
    if params["log"]:
        f = np.log(f)

    # remove the global trend if needed
    presig = removetrend(f,params)

    # apply regularization if needed
    presig = specreg(presig,params)

    # use the selected detection method
    if params["detect"] == "locmax":
        bounds = localmax(presig, params["N"])
        L = 0
    elif params["detect"] == "locmaxmin":
        bounds = localmaxmin(presig, params["N"],presig)
        L = 0
    elif params["detect"] == "locmaxminf":
        bounds = localmaxmin(presig, params["N"],f)
        L = 0
    elif params["detect"] == "adaptivereg":
        bounds = adaptive_bounds_adapt(presig, params["InitBounds"])
        L = 0
    elif params["detect"] == "adaptive":
        bounds = adaptive_bounds_adapt(f, params["InitBounds"])
        L = 0
    else:
        bounds, plane, L, th = gss_boundariesdetect(presig, params["typeDetect"])
    
    if params["Completion"]:
        bounds = boundaries_completion(f, bounds, params["N"])

    return bounds, presig

def boundaries_completion(f, bounds, NT):

    """ Complete set of given boundaries
    
    Parameters
    ----------
    f : nparray
        vector containing the spectrum
    bounds - nparray
        list of previously detected boundaries
    N : integer
        number of wanted supports

    Returns
    -------
    bounds - nparray
        Updated list of boundaries

    Notes
    -----
    This function complete the list of detected boundaries
    to make sure the minimum amount of modes is NT. If that 
    minimum is already reached then the function returns the 
    original set of boundaries. If completion is needed, then 
    the high frequencies support is evenly subdivided to provide
    the extra boundaries.

    Author: Jerome Gilles
    Institution: San Diego State University
    Version: 1.0 (12/18/2024)
    """

    if np.size(bounds) < NT-1:
        Nd = NT-np.size(bounds) - 1

        deltaw = (np.size(f) - bounds[-1])/(Nd+1)

        for k in range(0,Nd):
            extrabound = bounds[-1]+deltaw
            bounds = np.hstack((bounds, extrabound))

    return bounds

def localmax(f, N):

    """ Find mid-point between maxima

    Parameters
    ----------
    f : nparray
        vector containing the spectrum
    N : integer
        number of wanted supports

    Returns
    -------
    bounds - nparray
        list of N-1 boundaries

    Notes
    -----
    This function detects the position of the N-1 boundaries 
    which define N supports. These positions correspond to the
    mid-point between consecutive local maxima.

    Author: Jerome Gilles
    Institution: San Diego State University
    Version: 1.0 (12/17/2024)
    """

    N = N-1
    locmax = np.zeros(np.size(f))
    
    # detect local maxima
    for i in range(1,np.size(f)-1):
        if (f[i-1] < f[i]) and (f[i] > f[i+1]):
            locmax[i] = f[i]

    # we keep the N-th largest maxima and their index
    lmax = np.sort(locmax)
    Imax = np.argsort(locmax)
    lmax = lmax[::-1]
    Imax = Imax[::-1]

    if np.size(lmax) > N:
        Imax = np.sort(Imax[0:N])
    else:
        Imax = np.sort(Imax)
        N = np.size(Imax)

    # we find the index of the middle point between 
    # consecutive maxima
    bounds = np.zeros(N)
    for i in range(0,N):
        if i == 0:
            a = 0
        else:
            a = Imax[i-1]

        bounds[i] = (a+Imax[i])/2

    return bounds

def localmaxmin(f, N, f2):

    """ Find lowest minima between maxima

    Parameters
    ----------
    f : nparray
        vector containing the preprocessed spectrum
    N : integer
        number of wanted supports
    f2 : nparray
        vector containing the original spectrum used 
        to find the minima

    Returns
    -------
    bounds - nparray
        list of N-1 boundaries

    Notes
    -----
    This function detects the position of the N-1 boundaries 
    which define N supports. These positions correspond to the
    lowest minima between consecutive local maxima. 

    Author: Jerome Gilles
    Institution: San Diego State University
    Version: 1.0 (12/17/2024)
    """

    locmax = np.zeros(np.size(f))
    locmin = np.ones(np.size(f2)) * np.max(f2)
    # detect local maxima and minima
    Nm = 1
    for i in range(1,np.size(f)-1):
        if (f[i-1] < f[i]) and (f[i] > f[i+1]):
            locmax[i] = f[i]
            Nm = Nm + 1

        if (f2[i-1] > f2[i]) and (f2[i] < f2[i+1]):
            locmin[i] = f2[i]
    
    # check if we have enough maxima for the expected
    # amount of supports, if not we fix the number of 
    # supports
    if N > Nm:
        N = Nm

    # we keep the N-th largest maxima and their index
    if N != -1:
        N = N-1
        lmax = np.sort(locmax)
        Imax = np.argsort(locmax)
        lmax = lmax[::-1]
        Imax = Imax[::-1]

        if np.size(lmax) > N:
            Imax = np.sort(Imax[0:N])
        else:
            Imax = np.sort(Imax)
            N = np.size(lmax)

        # we detect the lowest minima between consecutive maxima
        bounds = np.zeros(N)
        for i in range(0,N):
            if i == 0:
                a = 0
            else:
                a = Imax[i-1]

            lmin = np.sort(locmin[a:Imax[i]])
            ind = np.argsort(locmin[a:Imax[i]])
            tmp = lmin[0]
        
            n = 0
            if n < np.size(lmin):
                n = 1
                while (n <= np.size(lmin)) and (tmp == lmin[n]):
                    n = n+1
            
            nn = np.ceil((n-1)/2)
            nn = int(nn)
            
            bounds[i] = a + ind[nn] - 1
    else:
        k = 0
        bounds = np.zeros(1)
        for i in range(0,np.size(locmin)):
            if locmin[i] < np.max(f2):               
                if i != 0:
                    bounds = np.hstack((bounds,i-1))

                k = k+1
        bounds = bounds[1:]

    return bounds

def adaptive_bounds_adapt(f,params):

    """ Adapt given boundaries to actual lowest minima
    
    Parameters
    ----------
    f : nparray
        vector containing the spectrum used to find the minima
    params : Dictionary
        must have params["degree"] set to the wanted polynomial degree    
    
    Returns
    -------
    bounds - nparray
        list of updated boundaries

    Notes
    -----
    This function adapt an initial set of boundaries to the studied signal.
    First it computes some neighborhood from the initial boundaries then it
    detects the global minima in each neighborhood.
    The local minima are computed from f. 

    Author: Jerome Gilles
    Institution: San Diego State University
    Version: 1.0 (12/18/2024)
    """

    bounds = np.zeros(np.size(params["InitBounds"]))

    # initialize boundaries in indices space + endpoints
    spect_bounds = np.hstack((0, params["InitBounds"], np.size(f)-1))
    
    for i in range(1,np.size(params["InitBounds"])+1):
        # detect an asymetric epsilon-neighborhood
        neighb_low = int(round_away(spect_bounds[i]-round_away(np.abs(spect_bounds[i]-spect_bounds[i-1])/2)))
        neighb_hi = int(round_away(spect_bounds[i]+round_away(np.abs(spect_bounds[i+1]-spect_bounds[i])/2)))

        imini = np.argmin(f[neighb_low:neighb_hi])
        bounds[i-1] = imini + neighb_low - 1

    return bounds

def plotboundaries(f,bounds,title, SamplingRate = -1, logtag = 0, Cpx = 0):

    """ Plot 1D boundaries

    Parameters
    ----------
    f : nparray
        input vector which contains the histogram
    bounds : nparray
        list of indices of the position of the detected local minima
    title : string
        title to be added to the plot

    Notes
    -----
    Plot the position of the meaningful minima on top of the original histogram.

    Author: Jerome Gilles
    Institution: San Diego State University
    Version: 1.0 (12/18/2024)
    """

    # Manage if we deal with half the specrtum (real case) or full spectrum
    if Cpx == 0:
        freq=np.linspace(0,np.size(f)-1,np.size(f)) * 0.5 / np.size(f)
        freq = 2 * np.pi * freq
    else:
        freq=np.linspace(0,np.size(f)-1,np.size(f)) / np.size(f)
        freq = (2 * freq - 1) * np.pi


    # adapt the plot wrt the provided sampling rate if any
    if SamplingRate != -1:
        freq = freq * SamplingRate / (2* np.pi)
        bounds = bounds * SamplingRate / (2* np.pi)

    # plot the log of the spectrum
    if logtag != 0:
        f = np.log(f)
    

    figb, axb = plt.subplots(figsize=(12,9))
    axb.plot(freq,f, linewidth=0.7)
    for i in range(0,np.size(bounds)):
        axb.axvline(x=bounds[i], color='r', linestyle='--')
    
    axb.set_title(title)
    plt.show()

    return

# The next functions are for the scale-space method

def plangaussianscalespace(f,n=3,t=0.16):

    """ Build the scale-space plane containing the detected local minima across the scales

    Parameters
    ----------
    f : nparray
        input vector
    n : integer
        size of the gaussian kernel (default = 3)
    t : real number
        initial scale (default = 0.16)

    Returns
    -------
    plane - 2D sparse lil_matrix (each column corresponds to a scale)
        1 for detected local minima, 0 otherwise

    Notes
    -----
    This function return a sparse lil_matrix where entries equal to 1 correspond to the position 
    of the detected lowest local minima across the different scales. The scale-space is built
    using the discrete Gaussian kernel based on the modified Bessel function of the first kind.

    Author: Jerome Gilles
    Institution: San Diego State University
    Version: 1.0 (12/18/2024)
    """
    
    Niter=np.ceil(np.size(f)/n).astype(int)  # number of iteration through the scales
    ker=ive(np.linspace(-n,n,2*n+1),t)   # discrete Gaussian kernel via Bessel function
    plane = scipy.sparse.lil_matrix((np.size(f),Niter+1),dtype = np.int8)

    bounds = localmaxmin2(f)
    plane[bounds,0] = 1

    for i in range(0,Niter):
        f = np.convolve(f,ker,'same')
        bounds = localmaxmin2(f)
        plane[bounds,i+1] = 1
        if np.size(bounds) == 0:
            break

    return plane

def localmaxmin2(f):

    """ Detect the lowest local minima between two consecutive local maxima

    Parameters
    ----------
    f : nparray
        input vector

    Returns
    -------
    bounds - nparray
        list of indices of the position of the detected local minima

    Notes
    -----
    This function return an array bounds containing the indices of the lowest local minima
    between two consecutive local maxima in a vector f

    Author: Jerome Gilles
    Institution: San Diego State University
    Version: 1.0 (12/18/2024)
    """

    locmax = np.zeros(np.size(f))
    locmin = np.max(f) * np.ones(np.size(f))

    # find the local maximum
    for i in range(1, np.size(f)-1):
        if (f[i-1] < f[i]) and (f[i] > f[i+1]):
            locmax[i] = f[i]
    
    i = 1
    Nmin = 0
    while i < (np.size(f)-2):
        if (f[i-1] > f[i]) and (f[i] < f[i+1]):
            locmin[i] = f[i]
            i = i+1
            Nmin = Nmin + 1
        elif (f[i-1] > f[i]) and (f[i] == f[i+1]):
            i0 = i
            while (i < np.size(f)-2) and (f[i] == f[i+1]):
                i = i+1
            if f[i] < f[i+1]:   # end of flat minimum
                locmin[int(round_away((i0+i)/2))] = f[int(round_away((i0+i)/2))]
                Nmin = Nmin + 1
            i = i+1
        else:
            i = i+1

    bounds = np.zeros(Nmin)
    nb = 0
    for i in range(0,np.size(locmin)-1):
        if locmin[i] < np.max(f):
            bounds[nb] = i
            nb = nb + 1

    return bounds

def meaningfulscalespace(f,plane,type):

    """ Extract meaningful boundaries from the scale-space plane with the selected method

    Parameters
    ----------
    f : nparray
        input vector
    plane : lil_matrix
        scale-space plane representation
    type: string
        method to be used: "otsu", "halfnormal", "empiricallaw", "mean", "kmeans"

    Returns
    -------
    bounds - nparray
        list of indices of the position of the detected local minima
    L - nparray
        vector containing the length of the scale-space curves
    th - number
        detected scale threshold

    Notes
    -----
    This function extracts the meaningful minima which will correspond to segmenting the 
    histogram f based on its scale-space representation in plane. It returns an array bounds 
    containing the indices of the boundaries, the set of length 
    of the scale-space curves L, and the detected threshold th.

    Author: Jerome Gilles
    Institution: San Diego State University
    Version: 1.0 (12/18/2024)
    """

    # Compute the scale length of each curve in plane
    L, ind = lengthscalecurve(plane)

    # Detect the meaningful minima with the selected method
    if type.lower() == "otsu":
        bounds, th = otsumethod(L,ind)
    elif type.lower() == "halfnormal":
        bounds, th = halfnormallaw(L,ind,plane.shape[1])
    elif type.lower() == "empiricallaw":
        bounds, th = empiricallaw(L,ind)
    elif type.lower() == "mean":
        bounds, th = meanth(L,ind)
    else:
        bounds, th = kmeansdetect(L,ind)

    # Postprocessing: manage curves originating from several minima
    bounds = removemerge(f, plane, bounds, th)

    return bounds, L, th

def lengthscalecurve(plane):

    """ Compute the length of curves in the scale-space plane

    Parameters
    ----------
    plane : lil_matrix
        scale-space plane representation

    Returns
    -------
    Length - nparray
        list of the length of each curve
    Indices - nparray
        list of indices of the position of the original local minima

    Notes
    -----
    This function returns a vector containing the length of each scale-space
    curves (in terms of scale lifespan) and a vector containing the indices 
    corresponding to the position of the original local minima (i.e. scale 0)

    Author: Jerome Gilles
    Institution: San Diego State University
    Version: 1.0 (12/18/2024)
    """

    nr = plane.shape[0]
    nc = plane.shape[1]

    Ncurve = 0
    for i in range(0,nr):
        if plane[i,0] == 1:
            Ncurve = Ncurve + 1

    Length = np.ones(Ncurve)
    Indices = np.zeros(Ncurve)
    ic = 0

    for i in range(0,nr):
        if plane[i,0] == 1:
            Indices[ic] = i
            i0 = i
            j0 = 1
            stop = 0
            if i0 == 0:
                while stop == 0:
                    if plane[i0,j0] == 1:
                        Length[ic] = Length[ic] +1
                        j0 = j0 + 1
                        if j0 > nc-1:
                            stop = 1
                    elif plane[i0+1,j0] == 1:
                        Length[ic] = Length[ic] +1
                        j0 = j0 + 1
                        i0 = i0 + 1
                        if (i0 == 0) or (j0 > nc-1):
                            stop = 1
                    else:
                        stop = 1
                
                ic = ic + 1
            elif i0 == nr-1:
                while stop == 0:
                    if plane[i0,j0] == 1:
                        Length[ic] = Length[ic] +1
                        j0 = j0 + 1
                        if j0 > nc-1:
                            stop = 1
                    elif plane[i0-1,j0] == 1:
                        Length[ic] = Length[ic] +1
                        j0 = j0 + 1
                        i0 = i0 - 1
                        if (i0 == nr-1) or (j0 > nc-1):
                            stop = 1
                    else:
                        stop = 1
                
                ic = ic + 1
            else:
                while stop == 0:
                    if plane[i0,j0] == 1:
                        Length[ic] = Length[ic] +1
                        j0 = j0 + 1
                        if j0 > nc-1:
                            stop = 1
                    elif plane[i0-1,j0] == 1:
                        Length[ic] = Length[ic] +1
                        j0 = j0 + 1
                        i0 = i0 - 1
                        if (i0 == 0) or (j0 > nc-1):
                            stop = 1
                    elif plane[i0+1,j0] == 1:
                        Length[ic] = Length[ic] +1
                        j0 = j0 + 1
                        i0 = i0 + 1
                        if (i0 == nr-1) or (j0 > nc-1):
                            stop = 1
                    else:
                        stop = 1
                
                ic = ic + 1

    return Length.astype(int), Indices.astype(int)

def otsumethod(L, ind):

    """ Detect meaningful minima using Otsu's method

    Parameters
    ----------
    L : nparray
        vector of the length of the minima curves
    ind: nparray
        vector containing the indices of the position of the original
        minima

    Returns
    -------
    bounds - nparray
        list of the indices of the position of the meaningful minima
    th - number
        detected scale threshold

    Notes
    -----
    This function classifies the set of minima curve lengths stored
    in L into two classes by using Otsu's method. It returns the 
    meaninful ones and the detected threshold.

    Author: Jerome Gilles
    Institution: San Diego State University
    Version: 1.0 (12/18/2024)
    """
    histo, be = np.histogram(L,np.max(L))
    Nt = histo.sum()
    histo = histo/Nt

    muT = 0.0
    for i in range(0,np.size(histo)):
        muT = muT + (i+1) * histo[i]

    sigbcv = np.zeros(np.size(histo)-1)

    for k in range(0,np.size(sigbcv)):
        wb = 0.0
        mu = 0.0
        for i in range(0,k+1):
            wb = wb + histo[i]
            mu = mu + (i+1) * histo[i]

        wf = 1 - wb
        if (wb != 0) and (wf != 0):
            mub = mu / wb
            muf = (muT - mu) / wf
            sigbcv[k] = wb * wf * (mub - muf) ** 2
        else:
            sigbcv[k] = 0

    th = maxcheckplateau(sigbcv) + 1

    Lb = np.ones(np.size(L))
    for i in range(0,np.size(L)):
        if L[i] < th:
            Lb[i] = 0

    bounds = ind[np.where(Lb==1)[0]]

    return bounds, th

def maxcheckplateau(L):

    """ Check plateau of maxima
    
    Parameters:
    -----------
    - L: nparray
        vector containing the indices of detected minima

    Returns:
    -th: center of the plateau. If no plateau, returns 1.
    
    Notes
    -----
    Check if maxima form a plateau. For those who do, it returns 
    the position in the center of the plateau.

    Author: Jerome Gilles
    Institution: San Diego State University
    Version: 1.0 (12/18/2024)
    """

    m = np.max(L)
    th = np.argmax(L)

    if np.size(th) == 0:
        # takes care if there's no maximum
        th = 1
    else:
        if (th < np.size(L)) and (L[th+1] == m):  # there is a plateau
            thb = th
            while (L[thb+1] == m) and (thb < np.size(L)-2):
                thb = thb + 1

            th = np.floor((th+thb)/2)

    return th

def halfnormallaw(L, ind, Lmax):

    """ Detect meaningful minima using the epsilon-meaningful method
    (half-normal law)

    Parameters
    ----------
    L : nparray
        vector of the length of the minima curves
    ind: nparray
        vector containing the indices of the position of the original
        minima
    Lmax: number
        maximum possible length of a minima curve (i.e. number of 
        columns in the scale-space plane)

    Returns
    -------
    bounds - nparray
        list of the indices of the position of the meaningful minima
    th - number
        detected scale threshold

    Notes
    -----
    This function classifies the set of minima curve lengths stored
    in L into the ones which are epsilon-meaningful for an half-normal 
    law fitted to the data. It returns the meaninful ones and the detected 
    threshold.

    Author: Jerome Gilles
    Institution: San Diego State University
    Version: 1.0 (12/18/2024)
    """

    # estimate sigma
    sigma = np.sqrt(np.pi/2) * np.mean(L)

    # compute the threshold
    th = np.sqrt(2) * sigma * erfinv(erf(Lmax / (np.sqrt(2) * sigma)) - 1/np.size(L))

    # keep the meaningful minima
    Lth = L
    for i in range(0,np.size(L)):
        if L[i] <= th:
            Lth[i] = 0

    bounds = ind[np.where(Lth != 0)[0]]

    return bounds, th

def empiricallaw(L,ind):

    """ Detect meaningful minima using the epsilon-meaningful method
    (empirical law)

    Parameters
    ----------
    L : nparray
        vector of the length of the minima curves
    ind: nparray
        vector containing the indices of the position of the original
        minima

    Returns
    -------
    bounds - nparray
        list of the indices of the position of the meaningful minima
    th - number
        detected scale threshold

    Notes
    -----
    This function classifies the set of minima curve lengths stored
    in L into the ones which are epsilon-meaningful for an empirical 
    law fitted. It returns the meaninful ones and the detected threshold.

    Author: Jerome Gilles
    Institution: San Diego State University
    Version: 1.0 (12/18/2024)
    """

    histo, be = np.histogram(L,np.max(L))
    chisto = np.cumsum(histo / np.sum(histo))
    
    th = np.where(chisto > (1-1/np.size(L)))[0][0]

    Lth = np.ones(np.size(L))
    for i in range(0,np.size(L)):
        if L[i] < th:
            Lth[i] = 0

    bounds = ind[np.where(Lth == 1)[0]]

    return bounds, th

def meanth(L, ind):

    """ Detect meaningful minima using a mean threshold

    Parameters
    ----------
    L : nparray
        vector of the length of the minima curves
    ind: nparray
        vector containing the indices of the position of the original
        minima

    Returns
    -------
    bounds - nparray
        list of the indices of the position of the meaningful minima
    th - number
        detected scale threshold

    Notes
    -----
    This function classifies the set of minima curve lengths stored
    in L into the ones which meaningful based on a threshold computed
    as the mean of L. It returns the meaninful ones and the detected threshold.

    Author: Jerome Gilles
    Institution: San Diego State University
    Version: 1.0 (12/18/2024)
    """

    th = np.ceil(np.mean(L))

    Lth = np.ones(np.size(L))
    for i in range(0,np.size(L)):
        if L[i] < th:
            Lth[i] = 0

    bounds = ind[np.where(Lth == 1)[0]]

    return bounds, th

def kmeansdetect(L, ind):

    """ Detect meaningful minima using kmeans

    Parameters
    ----------
    L : nparray
        vector of the length of the minima curves
    ind: nparray
        vector containing the indices of the position of the original
        minima

    Returns
    -------
    bounds - nparray
        list of the indices of the position of the meaningful minima
    th - number
        detected scale threshold

    Notes
    -----
    This function classifies the set of minima curve lengths stored
    in L into the ones which meaningful based on a kmeans clustering. 
    It returns the meaninful ones and the detected threshold.

    Author: Jerome Gilles
    Institution: San Diego State University
    Version: 1.0 (12/18/2024)
    """

    LL = np.zeros([np.size(L), 2])
    LL[:,1] = L
    km_model = KMeans(n_clusters=2, n_init="auto").fit(LL)
    clusters = km_model.fit_predict(LL)

    Lmax = np.where(L == np.max(L))[0][0]
    cln = clusters[Lmax]

    bounds = ind[np.where(clusters == cln)[0]]
    th = np.min(L[np.where(clusters == cln)[0]])

    
    return bounds, th

def removemerge(f, plane, bounds, th):

    """ Detect the meaningful minima using kmeans

    Parameters
    ----------
    f : nparray
        histogram to be segmented
    plane : lil_matrix
        scale-space plane representation
    bounds - 1D array
        list of the indices of the position of the meaningful minima
    th - number
        detected scale threshold

    Returns
    -------
    bounds - nparray
        updated list of the indices of the position of the meaningful minima

    Notes
    -----
    This function manage local minima which merge at some point in the
    scale-space plane according to the following rules:
        - if the mergin occur before the scale th then we keep only one minima
        (the lowest one) as they are not individually meaningful
        - if the mergin occur after the scale th then we consider that each 
        initial minima is meaningful and we keep them

    Author: Jerome Gilles
    Institution: San Diego State University
    Version: 1.0 (12/12/2024)
    """

    tagplane = scipy.sparse.lil_matrix(plane.shape,dtype = np.int8)
    indrem = np.zeros(np.size(bounds))

    # tag the first curve
    tag = bounds[0]
    stop = 0
    i = tag
    j = 0
    while stop != 1:
        tagplane[i,j] = tag
        if i > 0:
            if plane[i-1,j+1] == 1:
                i = i-1
                j = j+1
            elif plane[i,j+1] == 1:
                j = j+1
            elif plane[i+1,j+1] == 1:
                i = i+1
                j = j+1
            else:
                stop = 1
        else:
            if plane[i,j+1] == 1:
                j = j+1
            elif plane[i+1,j+1] == 1:
                i = i+1
                j = j+1
            else:
                stop = 1
        
        if (j > th) or (j == plane.shape[1]-2):
            stop = 1

    # we address the other curves
    for k in range(1,np.size(bounds)):
        tag = bounds[k]
        i = tag
        j = 0
        stop = 0
        retag = 0

        while stop != 1:
            tagplane[i,j] = tag
            if i >1:
                if plane[i-1,j+1] == 1:
                    if (tagplane[i-1,j+1] == bounds[k-1]) and (retag == 0):
                        if f[bounds[k-1]] < f[bounds[k]]:
                            indrem[k] = 1
                            stop =1
                        else:
                            indrem[k-1] = 1
                            retag = 1
                    i = i-1
                    j = j+1
                elif plane[i,j+1] == 1:
                    if (tagplane[i,j+1] == bounds[k-1]) and (retag == 0):
                        if f[bounds[k-1]] < f[bounds[k]]:
                            indrem[k] = 1
                            stop =1
                        else:
                            indrem[k-1] = 1
                            retag = 1
                    j = j+1
                elif plane[i+1,j+1] == 1:
                    i = i+1
                    j = j+1
                else:
                    stop = 1
            else:
                if plane[i,j+1] == 1:
                    if (tagplane[i,j+1] == bounds[k-1]) and (retag == 0):
                        if f[bounds[k-1]] < f[bounds[k]]:
                            indrem[k] = 1
                            stop =1
                        else:
                            indrem[k-1] = 1
                            retag = 1
                    j = j+1
                elif plane[i+1,j+1] == 1:
                    i = i+1
                    j = j+1
                else:
                    stop = 1
            
            if (j >  th) or (j == plane.shape[1]-2):
                stop = 1
        
    bounds = bounds[np.where(indrem == 0)[0]]

    return bounds

def gss_boundariesdetect(f,type):

    """ Extract meaningful boundaries by scale-space method

    Parameters
    ----------
    f : 1D array
        input vector
    type: string
        method to be used: "otsu", "halfnormal", "empiricallaw", "mean", "kmeans"

    Returns
    -------
    bounds - nparray
        list of indices of the position of the detected local minima
    plane : lil_matrix
        scale-space plane representation
    L - nparray
        vector containing the length of the scale-space curves
    th - number
        detected scale threshold

    Notes
    -----
    This function builds a scale-space representation of the provided histogram and then
    extract the meaningful minima which will correspond to segmenting the histogram. It 
    returns an array bounds containing the indices of the boundaries, the set of length 
    of the scale-space curves, and the detected threshold.

    Author: Jerome Gilles
    Institution: San Diego State University
    Version: 1.0 (12/11/2024)
    """

    # Build the scale-space plane
    plane = plangaussianscalespace(f)

    # Extract the meaningful minima, i.e. boundaries
    bounds, L, th = meaningfulscalespace(f,plane,type)

    return bounds, plane, L, th

def plotplane(plane, hideaxes = False):

    """ Plot the scale-space plane

    Parameters:
    -----------
    - plane: lil_matrix sparse matrix
        matrix containing the scale-space plane
    - hideaxes: False (default) or True
        if False, the axes will be plotted, if True , they will be hidden

    Notes
    -----
    This function plots the scale-space plane containing the position of the local
    minima through the scales

    Author: Jerome Gilles
    Institution: San Diego State University
    Version: 1.0 (12/11/2024)
    """

    figp, axp = plt.subplots(figsize=(10,10))
    axp.matshow(plane.transpose().toarray(),cmap=plt.cm.gray_r)
    axp.set_title(r"Scale-space plane")
    axp.set_ylabel('Scales')
    if hideaxes == True:
        plt.axis('off')
    plt.show()

# =================================================================
#                     TREND PREPROCESSING
# =================================================================
def removetrend(f, params):

    """ Remove the global trend

    Parameters
    ----------
    h : nparray
        vector containing the histogram
    params : Dictionary
        must have params["globtrend"] to be set to one of the methods 
        listed in the notes below.
        must have params["degree"] set to the wanted polynomial degree
        if the polylaw method is selected.

    Returns
    -------
    presig - nparray
        the processed histogram

    Notes
    -----
    This function removes the global trend from f using the selected
    method sets in params["globtrend"]. The available methods are:
    - "none" : does nothing, returns f
    - "powerlaw" : uses a power law to estimate the trend, returns 
                f - powerlaw(f)
    - "polylaw": uses a polynomial interpolation of degree specified
        by params["degree"] to estimate the trend, returns
                f - polynomial(f)
    - "morpho" : uses morphological operators, returns
                f - (opening(f)+closing(f))/2
    - "tophat" : uses morphological operators, returns
                f - opening(f)
    - "opening" : returns
                opening(f)

    Author: Jerome Gilles
    Institution: San Diego State University
    Version: 1.0 (12/16/2024)
    """

    if params["globtrend"] == "powerlaw":
        presig = powerlawtrend(f)
    elif params["globtrend"] == "polylaw":
        presig = polytrend(f,params)
    elif params["globtrend"] == "morpho":
        presig = morphotrend(f)
    elif params["globtrend"] == "tophat":
        presig = tophattrend(f)
    elif params["globtrend"] == "opening":
        presig = openingtrend(f)
    else:
        presig = f

    return presig

def powerlawtrend(h):

    """ Remove the global trend using a power law estimator

    Parameters
    ----------
    h : nparray
        vector containing the histogram

    Returns
    -------
    presig - nparray
        the processed histogram

    Notes
    -----
    This function fits a power law to the provided histogram 
    h. It then return the original histogram minus the power 
    law.

    Author: Jerome Gilles
    Institution: San Diego State University
    Version: 1.0 (12/16/2024)
    """
    
    h = h / np.max(h)
    w = np.linspace(1,np.size(h),np.size(h))
    lw = np.log(w)

    s = np.sum(lw * np.log(h))/np.sum(lw * lw)
    law = np.power(w,s)

    presig = h-law

    return presig

def polytrend(h,params):

    """ Remove global trend using a polynomial estimator

    Parameters
    ----------
    h : nparray
        vector containing the histogram
    params : Dictionary
        must have params["degree"] set to the wanted polynomial degree

    Returns
    -------
    presig - nparray
        the processed histogram

    Notes
    -----
    This function fits a polynomial of degree specified by params to the 
    provided histogram h. It then return the original histogram minus the 
    polynomial estimator.

    Author: Jerome Gilles
    Institution: San Diego State University
    Version: 1.0 (12/16/2024)
    """

    w = np.linspace(1,np.size(h),np.size(h))

    C = P.polyfit(w,h, params["degree"])
    plaw = P.polyval(w,C,tensor=False)

    presig = h - plaw

    return presig

def morphotrend(h):

    """ Remove global trend using morphological opening + closing

    Parameters
    ----------
    h : nparray
        vector containing the histogram

    Returns
    -------
    presig - nparray
        the processed histogram
    
    Notes
    -----
    This function estimates the trend as half the sum of the morphological
    opening + closing of h. It then return the original histogram minus the 
    trend.

    Author: Jerome Gilles
    Institution: San Diego State University
    Version: 1.0 (12/16/2024)
    """


    # We detect first the size of the structural element
    # as the smallest distance between two consecutive maxima +1
    
    locmax = np.zeros(np.size(h))

    # detect local maxima
    for i in range(1,np.size(h)-1):
        if (h[i-1] < h[i]) and (h[i] > h[i+1]):
            locmax[i] = h[i]

    sizeel = np.size(h)
    n = 0
    nn = 0
    while (n < (np.size(locmax))):
        if (locmax[n] != 0):
            if sizeel > (n-nn):
                sizeel = n-nn
            
            nn =n
            n = n+1
        
        n = n+1

    print(sizeel)

    presig = h - (morphoopening1D(h,sizeel+1) + morphoclosing1D(h,sizeel+1))/2

    return presig

def tophattrend(h):

    """ Remove global trend using morphological Top-Hat

    Parameters
    ----------
    h : nparray
        vector containing the histogram

    Returns
    -------
    presig - nparray
        the processed histogram

    Notes
    -----
    This function estimates the trend using the Top Hat morphological operator
    i.e. h-opening. It then return the original histogram minus the 
    trend.

    Author: Jerome Gilles
    Institution: San Diego State University
    Version: 1.0 (12/17/2024)
    """


    # We detect first the size of the structural element
    # as the smallest distance between two consecutive maxima +1
    
    locmax = np.zeros(np.size(h))

    # detect local maxima
    for i in range(1,np.size(h)-1):
        if (h[i-1] < h[i]) and (h[i] > h[i+1]):
            locmax[i] = h[i]

    sizeel = np.size(h)
    n = 0
    nn = 0
    while (n < (np.size(locmax))):
        if (locmax[n] != 0):
            if sizeel > (n-nn):
                sizeel = n-nn
            
            nn =n
            n = n+1
        
        n = n+1

    print(sizeel)

    presig = h - morphoopening1D(h,sizeel+1)

    return presig

def openingtrend(h):

    """ Remove global trend using morphological opening

    Parameters
    ----------
    h : nparray
        vector containing the histogram

    Returns
    -------
    presig - nparray
        the processed histogram

    Notes
    -----
    This function estimates the trend using the opening morphological operator. 
    It then return the opened histogram.

    Author: Jerome Gilles
    Institution: San Diego State University
    Version: 1.0 (12/17/2024)
    """


    # We detect first the size of the structural element
    # as the smallest distance between two consecutive maxima +1
    
    # Mirror the histogram
    n0 = np.size(h)-1
    hr = h[::-1]
    hr = hr[:np.size(hr)-1]
    h = np.hstack((hr,h))

    locmax = np.zeros(np.size(h))

    # detect local maxima
    for i in range(1,np.size(h)-1):
        if (h[i-1] < h[i]) and (h[i] > h[i+1]):
            locmax[i] = h[i]

    sizeel = np.size(h)
    n = 0
    nn = 0
    while (n < (np.size(locmax))):
        if (locmax[n] != 0):
            if sizeel > (n-nn):
                sizeel = n-nn
            
            nn = n
            n = n+1
        
        n = n+1

    presig = morphoopening1D(h,sizeel+1)
    presig = presig[n0:]

    return presig

def morphoerosion1D(f,sizeel):

    """ 1D Morphological Erosion

    Parameters
    ----------
    f : nparray
        vector containing the histogram
    sizeel : integer
        size of the structural element

    Returns
    -------
    s - nparray
        the eroded histogram

    Notes
    -----
    This function applies the 1D morphological erosion to f with a structural 
    element of size sizeel. It returns the eroded version of f.

    Author: Jerome Gilles
    Institution: San Diego State University
    Version: 1.0 (12/16/2024)
    """

    s = np.zeros(np.size(f))

    for x in range(0,np.size(f)):
        if x < (sizeel+1):
            a = 0
        else:
            a = x - sizeel

        if x >= (np.size(f)-sizeel):
            b = np.size(f) + 1
        else:
            b = x + sizeel + 1

        s[x] = np.min(f[a:b])

    return s
    
def morphodilation1D(f,sizeel):

    """ 1D Morphological Dilation

    Parameters
    ----------
    f : nparray
        vector containing the histogram
    sizeel : integer
        size of the structural element

    Returns
    -------
    s - nparray
        the dilated histogram

    Notes
    -----
    This function applies the 1D morphological dilation to f with a structural 
    element of size sizeel. It returns the dilated version of f.

    Author: Jerome Gilles
    Institution: San Diego State University
    Version: 1.0 (12/16/2024)
    """

    s = np.zeros(np.size(f))

    for x in range(0,np.size(f)):
        if x < (sizeel+1):
            a = 0
        else:
            a = x - sizeel

        if x >= (np.size(f)-sizeel):
            b = np.size(f) + 1
        else:
            b = x + sizeel + 1

        s[x] = np.max(f[a:b])

    return s

def morphoopening1D(f,sizeel):

    """ 1D Morphological Opening

    Parameters
    ----------
    f : nparray
        vector containing the histogram
    sizeel : integer
        size of the structural element

    Returns
    -------
    ouv - nparray
        the opened histogram

    Notes
    -----
    This function applies the 1D morphological opening to f with a structural 
    element of size sizeel (i.e. opening = dilation(erosion)). It returns the 
    opened version of f.
        
    Author: Jerome Gilles
    Institution: San Diego State University
    Version: 1.0 (12/16/2024)
    """

    ouv = morphodilation1D(morphoerosion1D(f,sizeel),sizeel)

    return ouv

def morphoclosing1D(f,sizeel):

    """ 1D Morphological Closing

    Parameters
    ----------
    f : nparray
        vector containing the histogram
    sizeel : integer
        size of the structural element

    Returns
    -------
    clo - nparray
        the closed histogram

    Notes
    -----
    This function applies the 1D morphological closing to f with a structural 
    element of size sizeel (i.e. opening = erosion(dilation). It returns the 
    closed version of f.

    Author: Jerome Gilles
    Institution: San Diego State University
    Version: 1.0 (12/16/2024)
    """
    
    clo = morphoerosion1D(morphodilation1D(f,sizeel),sizeel)

    return clo

# =================================================================
#                     SPECTRUM REGULARIZATION
# =================================================================
def specreg(f, params):

    """ Spectrum regularization

    Parameters
    ----------
    f : nparray
        vector containing the spectrum
    params : Dictionary
        must have params["reg"] to be set to one of the methods 
        listed above
        must have params["lengthfilter"] to be set for all methods
        must have params["sigmafilter"] to be set to the standard
        deviation for the Gaussian method

    Returns
    -------
    freg - nparray
        the processed histogram

    Notes
    -----
    This function regularizes f using the selected
    method sets in params["reg"]. The available methods are:
    - "none" : does nothing, returns f
    - "gaussian" : convolve f with a Gaussian of length and standard
        deviation given by params["lengthfilter"] and 
        params["sigmafilter"], respectively
    - "average": convolve f with a constant filter of length given 
        by params["lengthfilter"]
    - "closing" : applies the morphological closing operator of length 
        given by params["lengthfilter"]

    Author: Jerome Gilles
    Institution: San Diego State University
    Version: 1.0 (12/17/2024)
    """

    if params["reg"] == "gaussian":
        filter = signal.windows.gaussian(params["lengthfilter"],std=params["sigmafilter"])
        filter = filter/np.sum(filter)
        freg = np.convolve(f,filter,'same')
    elif params["reg"] == "average":
        filter = np.ones(params["lengthfilter"])
        filter = filter/np.sum(filter)
        freg = np.convolve(f,filter,'same')
    elif params["reg"] == "closing":
        freg = morphoclosing1D(f,params["lengthfilter"])
    else:
        freg = f

    return freg

