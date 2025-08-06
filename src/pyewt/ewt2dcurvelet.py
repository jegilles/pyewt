import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse , Arc

from .pseudopolarfft import PPFFT
from .usefullfunc import round_away , beta
from .boundaries1d import boundaries_detect , removetrend , specreg, gss_boundariesdetect

def ewt2d_curvelet(f,params):

    """ 2D empirical curvelet transform

    Parameters
    ----------
    - f: ndarray
        input image (must be square)
    - params: Dictionary
        must be set properly accordingly to the options described
        in the Notes section

    Returns
    -------
    - ewtc: list
        set of curvelet coefficients.
    - mfb: list
        set of curvelet filters that were used to perform the forward transform.
    - Bw: set of detected scales (see Notes below)
    - Bt: set of detected angles (see Notes below)

    Notes
    -----
    This function performs the empirical curvelet transform on the square image f.
    Three options are available (set in params["option"] as 1, 2 or 3):
        1) scales and angles are detected independently
        2) scales are detected first then the angular sector within each scale ring
        3) angles are detected first then the scales within each angular sector
    Note that the scaling function is always detect first before performing the above
    options.

    The options for detecting the scales and angles are set in params as:
    If params["log"] is set to True, then the detection is performed on the logarithm 
    of the spectrum of f.
    
    SCALES:
    Two types of preprocessing are available to "clean" the spectrum f:
    
    A) A global trend remover method is set via params["globtrend"]:
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

    B) A regularization method is set via params["reg"]:
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
    is the number of detected modes is lower than an expected value params["N"]. 
    If needed, the last high frequency support is evenly splitted. This 
    step is performed if params["Completion"] is set to True.
    
    ANGLES:
    Two types of preprocessing are available to "clean" the spectrum f:
    
    A) A global trend remover method is set via params["curvpreproc"]:
        - "none" : does nothing, returns f
        - "powerlaw" : uses a power law to estimate the trend, returns 
                f - powerlaw(f)
        - "polylaw": uses a polynomial interpolation of degree specified
            by params["curvdegree"] to estimate the trend, returns
                f - polynomial(f)
        - "morpho" : uses morphological operators, returns
                f - (opening(f)+closing(f))/2
        - "tophat" : uses morphological operators, returns
                f - opening(f)
        - "opening" : returns
                opening(f)

    B) A regularization method is set via params["curvreg"]:
        - "none" : does nothing, returns f
        - "gaussian" : convolve f with a Gaussian of length and standard
            deviation given by params["curvlengthfilter"] and 
            params["sigmafilter"], respectively
        - "average": convolve f with a constant filter of length given 
            by params["curvlengthfilter"]
        - "closing" : applies the morphological closing operator of length 
            given by params["curvlengthfilter"]

    The wanted boundary detection method must be set in params["curvmethod"].
    The available options are:
    - "localmax" : select the mid-point between consecutive maxima.
        params["curvN"] must be set to the expected number of modes.
    - "localmaxmin" : select the lowest minima between consecutive 
        maxima of the preprocessed spectrum.
        params["curvN"] must be set to the expected number of modes.
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

    The ewtc and mfb are organized as follows: 1) (ewtc{1} are the lowpass 
    coefficient (scaling function). For option 1 and 2, next are mfb{s}{t} are the 
    coefficients where s corresponds to the scales and t to the direction; while for 
    option 3 they are ordered as mfb{t}{s}. For mfb, all filters have been shifted 
    back to have the (0,0) frequency in the upper left corner.

    For option 1 and 2, Bw is an array containing the set of scale boundaries, for 
    option 3 it is a list where Bw[0] is the radius of the scaling function, the
    remaining Bw[k] are ndarray containing the list of scale radius per each angular 
    sector.

    For option 1 and 3, Bt is an array contianing the set of angles boundaries, for 
    option 2 it is a list where each Bt[k] is an ndarray containing the detected angles
    within each scale ring.

    Author: Jerome Gilles
    Institution: San Diego State University
    Version: 1.0 (01/30/2025)  
    """

    # check if the input image is of proper complex type to
    # guarantee machine precision in the transform. If it is
    # not then we cast it to the appropriate type.
    if f.dtype != np.complex128:
        f = f.astype(np.complex128)

    [H, W] = np.shape(f)

    # We compute the pseudo polar Fourier transform
    pseudofft = PPFFT(f)

    if params["option"] == 2:
        # Option 2: scales first then angular sectors
        # we start with the scales
        meanfft = np.fft.fftshift(np.sum(np.abs(pseudofft),axis=1))

        # Detect the boundaries
        boundaries, presig = boundaries_detect(np.abs(meanfft[0:int(round_away((np.size(meanfft))/2))]),params)
        if boundaries[0] == 0:
            boundaries = boundaries[1:]
        Bw = boundaries * np.pi / round_away(np.size(meanfft)/2)

        # Detect angles for each scale
        Bt = []
        for s in np.arange(np.size(Bw)-1):
            meanfft = np.sum(np.abs(pseudofft[boundaries[s]:boundaries[s+1],:]),axis=0)
            # detect the theta boundaries
            bounds = ewt_angles_detect(meanfft,params)
            Bt.append(bounds * np.pi / np.size(meanfft) - 3 * np.pi /4)
        meanfft = np.sum(np.abs(pseudofft[boundaries[-1]:,:]),axis=0)
        bounds = ewt_angles_detect(meanfft,params)
        Bt.append(bounds * np.pi / np.size(meanfft) - 3 * np.pi /4)

        # Build the filter bank
        mfb = ewt2d_curvelet_filterbank(Bw,Bt,W,H,params["option"])
    elif params["option"] == 3:
        # Option 3: angular sectors first then scales

        # find the first scale
        meanfft = np.fft.fftshift(np.sum(np.abs(pseudofft),axis=1))
        LL = round_away(np.size(meanfft)/2)
        boundariesW, presig = boundaries_detect(np.abs(meanfft[0:int(round_away((np.size(meanfft))/2))]),params)
        if boundariesW[0] == 0:
            boundariesW = boundariesW[1:]
        Bw1 = boundariesW[0] * np.pi / LL

        # compute the mean spectrum wrt the magnitude frequency to find the angles
        meanfft = np.sum(np.abs(pseudofft[int(np.floor(np.shape(pseudofft)[0]/2))+boundariesW[0]:,:]),axis=0)
        # detect the boundaries
        boundaries = ewt_angles_detect(meanfft,params)
        Bt = boundaries * np.pi / np.size(meanfft) - 3 * np.pi /4

        Bw = []
        Bw.append(Bw1)
        # we detect the scales per each angular sector
        for t in np.arange(np.size(boundaries)-1):
            # average the spectrum on the current angular sector
            meanfft = np.sum(np.abs(pseudofft[int(np.floor(np.shape(pseudofft)[0]/2))+boundariesW[0]:,boundaries[t]:boundaries[t+1]]),axis=1)
            # detect the boundaries
            bounds, presig = boundaries_detect(meanfft,params)
            Bw.append((boundariesW[0]+bounds) * np.pi / LL)

        # last one
        meanfft = np.sum(np.abs(pseudofft[int(np.floor(np.shape(pseudofft)[0]/2))+boundariesW[0]:,boundaries[-1]:]),axis=1) + np.sum(np.abs(pseudofft[int(np.floor(np.shape(pseudofft)[0]/2))+boundariesW[0]:,0:boundaries[0]]),axis=1)

        # detect boundaries (note in Matlab the 'closing' regularization is forced...)
        bounds, presig = boundaries_detect(meanfft,params)
        Bw.append((boundariesW[0]+bounds) * np.pi / LL)

        # Build the filter bank
        mfb = ewt2d_curvelet_filterbank(Bw,Bt,W,H,params["option"])
    else:
        # Option 1: scales and angles are detected independently
        # we start with the scales
        meanfft = np.fft.fftshift(np.sum(np.abs(pseudofft),axis=1))

        # Detect the boundaries
        boundaries, presig = boundaries_detect(np.abs(meanfft[0:int(round_away((np.size(meanfft))/2))]),params)
        if boundaries[0] == 0:
            boundaries = boundaries[1:]
        Bw = boundaries * np.pi / round_away(np.size(meanfft)/2)

        # next the angles
        meanfft = np.sum(np.abs(pseudofft),axis=0)

        # Detect the boundaries
        boundaries = ewt_angles_detect(meanfft,params)
        Bt = boundaries * np.pi / np.size(meanfft) - 3 * np.pi / 4

        # Build the filter bank
        mfb = ewt2d_curvelet_filterbank(Bw,Bt,W,H,params["option"])

    # we filter to extract each subband
    ff = np.fft.fft2(f)

    ewtc = []
    # scaling function first
    tmp = np.fft.ifft2(np.multiply(np.conjugate(mfb[0]),ff))
    ewtc.append(tmp)
    # angular sectors next
    for k in np.arange(len(mfb)-1):
        ewtc.append([])
        for l in np.arange(len(mfb[k+1])):
            tmp = np.fft.ifft2(np.multiply(np.conjugate(mfb[k+1][l]),ff))
            ewtc[k+1].append(tmp)

    return ewtc, mfb, Bw, Bt

def iewt2d_curvelet(ewtc, mfb):

    """ Inverse 2D empirical curvelet transform

    Parameters
    ----------
    - ewtc: list
        set of curvelet coefficients.
    - mfb: list
        set of curvelet filters that were used to perform the forward transform.

    Returns
    -------
    - rec: ndarray
        reconstructed image (note it is of complex type)

    Notes
    -----
    Uses the formulation of the dual filter bank to perform the inverse transform.
    
    Author: Jerome Gilles
    Institution: San Diego State University
    Version: 1.0 (01/30/2025)  
    """

    # scaling function first
    rec = np.fft.ifft2(np.multiply(mfb[0],np.fft.fft2(ewtc[0])))
    # angular sectors next
    for k in np.arange(len(mfb)-1):
        for l in np.arange(len(mfb[k+1])):
            rec = rec + np.fft.ifft2(np.multiply(mfb[k+1][l],np.fft.fft2(ewtc[k+1][l])))

    return rec

# =============================================================
#                   ANGLE DETECTION ROUTINES
# =============================================================

def ewt_angles_detect(f,params):

    """ Detect the angular boundaries

    Parameters
    ----------
    - f: ndarray
        1D spectrum on which to perform the detection
    - params: Dictionary
        must be set properly accordingly to the options described
        in the Notes section

    Returns
    -------
    - boundaries: ndarray
        set of detected angles

    Notes
    -----
    If params["log"] is set to True, then the detection is performed
    on the logarithm of the spectrum of f.

    Two types of preprocessing are available to "clean" the spectrum f:
    
    A) A global trend remover method is set via params["curvpreproc"]:
        - "none" : does nothing, returns f
        - "powerlaw" : uses a power law to estimate the trend, returns 
                f - powerlaw(f)
        - "polylaw": uses a polynomial interpolation of degree specified
            by params["curvdegree"] to estimate the trend, returns
                f - polynomial(f)
        - "morpho" : uses morphological operators, returns
                f - (opening(f)+closing(f))/2
        - "tophat" : uses morphological operators, returns
                f - opening(f)
        - "opening" : returns
                opening(f)

    B) A regularization method is set via params["curvreg"]:
        - "none" : does nothing, returns f
        - "gaussian" : convolve f with a Gaussian of length and standard
            deviation given by params["curvlengthfilter"] and 
            params["sigmafilter"], respectively
        - "average": convolve f with a constant filter of length given 
            by params["curvlengthfilter"]
        - "closing" : applies the morphological closing operator of length 
            given by params["curvlengthfilter"]

    The wanted boundary detection method must be set in params["curvmethod"].
    The available options are:
    - "localmax" : select the mid-point between consecutive maxima.
        params["curvN"] must be set to the expected number of modes.
    - "localmaxmin" : select the lowest minima between consecutive 
        maxima of the preprocessed spectrum.
        params["curvN"] must be set to the expected number of modes.
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

    Author: Jerome Gilles
    Institution: San Diego State University
    Version: 1.0 (01/30/2025)  
    """

    # set the proper options for the angles
    params2 = params
    params2["globtrend"] = params["curvpreproc"]
    params2["reg"] = params["curvreg"]
    params2["degree"] = params["curvdegree"]
    params2["lengthfilter"] = params["curvlengthfilter"]
    params2["sigmafilter"] = params["curvsigmafilter"]

    presig = removetrend(f,params2) # remove trend in the angular spectrum
    presig = specreg(presig,params) # regularize the angular spectrum

    if params["curvmethod"] == "locmax":
        boundaries = ewt_angles_locmax(presig,params["curvN"])
    elif params["curvmethod"] == "locmaxmin":
        boundaries = ewt_angles_locmaxmin(presig,params["curvN"])
    else:
        boundaries, plane, L, th = gss_boundariesdetect(presig, params["typeDetect"])


    return boundaries

def ewt_angles_locmax(f,N):

    """ Detect the midpoint between N largest maxima

    Parameter
    ---------
    - f: ndarray
        1D spectrum on which to perform the detection
    - N: maximal number of largest maxima

    Returns
    - bounds: ndarray
        set of detected midpoints

    Notes
    -----
    The minima are return as indices (not frequency).
    
    Author: Jerome Gilles
    Institution: San Diego State University
    Version: 1.0 (01/30/2025)  
    """

    locmax = np.zeros(np.size(f))

    # detect local maxima
    for i in np.arange(np.size(f)-2):
        if (f[i] < f[i+1]) and (f[i+1] > f[i+2]):
            locmax[i+1] = f[i+1]

    # check the endpoints (we work on the torus)
    if (f[-1] < f[0]) and (f[0] > f[1]):
        locmax[0] = f[0]

    if (f[np.size(f)-2] < f[-1]) and (f[-1] > f[0]):
        locmax[-1] = f[-1]

    # keep the N-th highest maxima and their indices
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
        if i == N-1:
            bounds[i] = (Imax[i]+Imax[0]+np.size(f))/2
            if bounds[i] > np.size(f):
                bounds[i] = bounds[i]-np.size(f)
        else:
            bounds[i] = (Imax[i]+Imax[i+1])/2

    bounds = np.sort(bounds)

    return bounds

def ewt_angles_locmaxmin(f,N):

    """ Detect the lowest minima between N largest maxima

    Parameter
    ---------
    - f: ndarray
        1D spectrum on which to perform the detection
    - N: maximal number of largest maxima

    Returns
    - bounds: ndarray
        set of detected minima

    Notes
    -----
    The minima are return as indices (not frequency).
    
    Author: Jerome Gilles
    Institution: San Diego State University
    Version: 1.0 (01/30/2025)  
    """

    locmax = np.zeros(np.size(f))
    locmin = np.max(f) * np.ones(np.size(f))

    # we dectect the local maxima and minima
    for i in np.arange(np.size(f)-2):
        if (f[i] < f[i+1]) and (f[i+1] > f[i+2]):
            locmax[i+1] = f[i+1]

        if (f[i] > f[i+1]) and (f[i+1] <= f[i+2]):
            locmin[i+1] = f[i+1]

    # check the endpoints (we work on the torus)
    if (f[-1] < f[0]) and (f[0] > f[1]):
        locmax[0] = f[0]
    if (f[-1] > f[0]) and (f[0] <= f[1]):
        locmin[0] = f[0]

    if (f[np.size(f)-2] < f[-1]) and (f[-1] > f[0]):
        locmax[-1] = f[-1]
    if (f[np.size(f)-2] > f[-1]) and (f[-1] <= f[0]):
        locmin[-1] = f[-1]

    # keep the N-th highest maxima and their indices
    lmax = np.sort(locmax)
    Imax = np.argsort(locmax)
    lmax = lmax[::-1]
    Imax = Imax[::-1]

    if np.size(lmax) > N:
        Imax = np.sort(Imax[0:N])
    else:
        Imax = np.sort(Imax)
        N = np.size(Imax)    

    # we detect the lowest minima between two consecutive maxima
    bounds = np.zeros(N)
    for i in np.arange(N):
        if i == N-1:
            lmin = np.sort(np.hstack((locmin[Imax[i]:],locmin[0:Imax[0]])))
            ind = np.argsort(np.hstack((locmin[Imax[i]:],locmin[0:Imax[0]])))
            tmp = lmin[0]
            n = 0 
            if n < np.size(lmin):
                n = 1
                while (n <= np.size(lmin)) and (tmp == lmin[n]):
                    n = n+1

            bounds[i] = Imax[i] + ind[int(np.ceil(n/2))-1]  # maybe -2?
            if bounds[i] > np.size(f):
                bounds[i] = bounds[i] - np.size(f)
        else:
            lmin = np.sort(locmin[Imax[i]:Imax[i+1]])
            ind = np.argsort(locmin[Imax[i]:Imax[i+1]])
            tmp = lmin[0]
            n = 0 
            if n < np.size(lmin):
                n = 1
                while (n <= np.size(lmin)) and (tmp == lmin[n]):
                    n = n+1
            bounds[i] = Imax[i] + ind[int(np.ceil(n/2))-1]  # maybe -2?
            
    return np.sort(bounds)

# =============================================================
#             ROUTINES TO BUILD CURVELET FILTERS
# =============================================================
def ewt2d_curvelet_filterbank(Bw,Bt,W,H,option=1):

    """ Build the empirical curvelet filter bank

    Parameters
    ----------
    - Bw: ndarray
        list of radius boundaries (scales) normalized in [0,pi)
    - Bt: ndarray
        list of angular boundaries normalized in [-3pi/4,pi/4]
    - W, H: width and height of the image
    - option: 1 (default), 2, or 3
        wanted curvelet option used to detect Bw and Bt

    Returns
    -------
    - mfb: list
        set of filters
    
    Notes
    -----
    The filter bank is organized as follows: 1) (mfb{1} is the lowpass 
    filter (scaling function). For option 1 and 2, next are mfb{s}{t} are the bandpass 
    filters where s corresponds to the scales and t to the direction; while for 
    option 3 they are ordered as mfb{t}{s}. All filters have been shifted back to 
    have the (0,0) frequency in the upper left corner.
    
    Author: Jerome Gilles
    Institution: San Diego State University
    Version: 1.0 (01/30/2025)  
    """

    if np.remainder(H,2) == 0:
        extH=1
        H=H+1
    else:
        extH=0

    if np.remainder(W,2) == 0:
        extW=1
        W=W+1
    else:
        extW=0

    if option == 1:
        # scales and angular sector are detected independently

        # we compute gammaw
        gammaw = np.pi
        Npic = np.size(Bw)
        for k in np.arange(Npic-1):
            r = (Bw[k+1]-Bw[k])/(Bw[k+1]+Bw[k])
            if r < gammaw:
                gammaw = r
        r = (np.pi-Bw[-1])/(np.pi+Bw[-1])
        if r < gammaw:
            gammaw = r
        gammaw = (1-1/np.max((W,H))) * gammaw
        if gammaw > Bw[0]:
            gammaw = (1-1/np.max((W,H))) * Bw[0]

        # we compute DTheta
        DTheta = 2 * np.pi
        Npic = np.size(Bt)
        for k in np.arange(Npic-1):
            r = (Bt[k+1]-Bt[k])/2
            if r < DTheta:
                DTheta = r

        r = (Bt[0]+np.pi-Bt[-1])/2
        if r < DTheta:
            DTheta = r
        DTheta = (1-1/np.max((H,W))) * DTheta

        # we prepare the polar grid
        theta, radius = ewt2d_create_polar_grid(W,H)

        # now we build the filters
        mfb = []
        # we start with the scaling function
        tmp = ewt2d_curvelet_scaling(radius,Bw[0],gammaw,W,H)
        mfb.append(tmp)

        # generate each angular sector except the last ones
        for s in np.arange(np.size(Bw)-1):
            mfb.append([])
            for t in np.arange(np.size(Bt)-1):
                mfb[s+1].append(ewt2d_angular_sector(theta, radius, Bt[t], Bt[t+1], Bw[s], Bw[s+1], gammaw, DTheta))
            mfb[s+1].append(ewt2d_angular_sector(theta, radius, Bt[-1], Bt[0]+np.pi, Bw[s], Bw[s+1], gammaw, DTheta))

        # generate the last ones
        mfb.append([])
        for t in np.arange(np.size(Bt)-1):
            mfb[-1].append(ewt2d_angular_sector(theta, radius, Bt[t], Bt[t+1], Bw[-1], 2*np.pi, gammaw, DTheta))
            
        mfb[-1].append(ewt2d_angular_sector(theta, radius, Bt[-1], Bt[0]+np.pi, Bw[-1], 2*np.pi, gammaw, DTheta))
    elif option == 2:
        # scales are detected first then angular sectors per each scale

        # we compute gammaw
        gammaw = np.pi
        Npic = np.size(Bw)
        for k in np.arange(Npic-1):
            r = (Bw[k+1]-Bw[k])/(Bw[k+1]+Bw[k])
            if r < gammaw:
                gammaw = r
        r = (np.pi-Bw[-1])/(np.pi+Bw[-1])
        if r < gammaw:
            gammaw = r
        gammaw = (1-1/np.max((W,H))) * gammaw

        # we compute DTheta
        DTheta = 2 * np.pi * np.ones(len(Bw))
        for s in np.arange(len(Bw)):
            Npic = np.size(Bt[s])
            for k in np.arange(Npic-1):
                r = (Bt[s][k+1]-Bt[s][k])/2
                if r < DTheta[s]:
                    DTheta[s] = r

            r = (Bt[s][0]+np.pi-Bt[s][-1])/2
            if r < DTheta[s]:
                DTheta[s] = r
            DTheta[s] = (1-1/np.max((H,W))) * DTheta[s]

        # we prepare the polar grid
        theta, radius = ewt2d_create_polar_grid(W,H)

        # now we build the filters
        mfb = []
        # we start with the scaling function
        tmp = ewt2d_curvelet_scaling(radius,Bw[0],gammaw,W,H)
        mfb.append(tmp)
        # generate each angular sector except the last ones
        for s in np.arange(np.size(Bw)-1):
            mfb.append([])
            for t in np.arange(np.size(Bt[s])-1):
                mfb[s+1].append(ewt2d_angular_sector(theta, radius, Bt[s][t], Bt[s][t+1], Bw[s], Bw[s+1], gammaw, DTheta[s]))
            
            mfb[s+1].append(ewt2d_angular_sector(theta, radius, Bt[s][-1], Bt[s][0]+np.pi, Bw[s], Bw[s+1], gammaw, DTheta[s]))

        # generate the last ones
        mfb.append([])
        for t in np.arange(np.size(Bt[-1])-1):
            mfb[-1].append(ewt2d_angular_sector(theta, radius, Bt[-1][t], Bt[-1][t+1], Bw[-1], 2*np.pi, gammaw, DTheta[-1]))
            
        mfb[-1].append(ewt2d_angular_sector(theta, radius, Bt[-1][-1], Bt[-1][0]+np.pi, Bw[-1], 2*np.pi, gammaw, DTheta[-1]))
    else:
        # angular sectors are detected first then scales per angular sector

        # we compute DTheta
        DTheta = 2 * np.pi
        Npic = np.size(Bt)
        for k in np.arange(Npic-1):
            r = (Bt[k+1]-Bt[k])/2
            if r < DTheta:
                DTheta = r

        r = (Bt[0]+np.pi-Bt[-1])/2
        if r < DTheta:
            DTheta = r
        DTheta = (1-1/np.max((H,W))) * DTheta

        # we compute gammaw
        gammaw = Bw[0]/2
        for t in np.arange(np.size(Bt)-1):
            Npic = np.size(Bw[t+1])
            for k in np.arange(Npic-1):
                r = (Bw[t+1][k+1]-Bw[t+1][k])/(Bw[t+1][k+1]+Bw[t+1][k])
                if r < gammaw:
                    gammaw = r
            r = (np.pi-Bw[t+1][-1])/(np.pi+Bw[t+1][-1])
            if r < gammaw:
                gammaw = r
            gammaw = (1-1/np.max((W,H))) * gammaw

        # we prepare the polar grid
        theta, radius = ewt2d_create_polar_grid(W,H)

        # we build the filters
        mfb = []
        # we start with the scaling function
        tmp = ewt2d_curvelet_scaling(radius,Bw[0],gammaw,W,H)
        mfb.append(tmp)

        # generate each angular sector
        for t in np.arange(np.size(Bt)-1):
            mfb.append([])
            # generate the first scale
            mfb[t+1].append(ewt2d_angular_sector(theta, radius, Bt[t], Bt[t+1], Bw[0], Bw[t+1][0], gammaw, DTheta))
            # generate the other scales
            for s in np.arange(np.size(Bw[t+1])-1):
                mfb[t+1].append(ewt2d_angular_sector(theta, radius, Bt[t], Bt[t+1], Bw[t+1][s], Bw[t+1][s+1], gammaw, DTheta))
            mfb[t+1].append(ewt2d_angular_sector(theta, radius, Bt[t], Bt[t+1], Bw[t+1][-1], 2 * np.pi, gammaw, DTheta))

        # generate the last one
        mfb.append([])
        # generate the first scale
        mfb[-1].append(ewt2d_angular_sector(theta, radius, Bt[-1], Bt[0]+np.pi, Bw[0], Bw[-1][0], gammaw, DTheta))
        # generate the other scales
        for s in np.arange(np.size(Bw[-1])-1):
            mfb[-1].append(ewt2d_angular_sector(theta, radius, Bt[-1], Bt[0]+np.pi, Bw[-1][s], Bw[-1][s+1], gammaw, DTheta))
        mfb[t+1].append(ewt2d_angular_sector(theta, radius, Bt[-1], Bt[0]+np.pi, Bw[-1][-1], 2 * np.pi, gammaw, DTheta))

    # check if we need to cut it back to the original size
    if extH == 1:
        mfb[0] = mfb[0][0:np.shape(mfb[0])[0]-1,:]
        for s in np.arange(len(mfb)-1):
            for t in np.arange(len(mfb[s+1])):
                mfb[s+1][t] = mfb[s+1][t][0:np.shape(mfb[s+1][t])[0]-1,:]

    if extW == 1:
        mfb[0] = mfb[0][:,0:np.shape(mfb[0])[1]-1]
        for s in np.arange(len(mfb)-1):
            for t in np.arange(len(mfb[s+1])):
                mfb[s+1][t] = mfb[s+1][t][:,0:np.shape(mfb[s+1][t])[1]-1]

    # shift back the filter in the correct position for filtering
    mfb[0] = np.fft.ifftshift(mfb[0])
    for s in np.arange(len(mfb)-1):
        for t in np.arange(len(mfb[s+1])):
            mfb[s+1][t] = np.fft.ifftshift(mfb[s+1][t])

    return mfb

def ewt2d_create_polar_grid(W,H):

    """ Create the polar coordinates

    Parameters
    ----------
    - W, H: width and height of the image

    Returns
    -------
    - theta: ndarray
        Angles at each pixel of the polar grid corresponding to the image
    - radius: ndarray
        Radius at each pixel of the polar grid corresponding to the image

    Notes
    -----
    This function creates matrices containing the polar radius and angles (in 
    [-pi, pi]) for each pixel (assuming that the position (0,0) is at the center 
    of the image.
    
    Author: Jerome Gilles
    Institution: San Diego State University
    Version: 1.0 (01/30/2025)  
    """

    theta = np.zeros((H,W))
    radius = np.zeros((H,W))

    middleh = np.floor(H/2)
    middlew = np.floor(W/2)

    for i in np.arange(W):
        for j in np.arange(H):
            ri = (i-middlew) * np.pi / (middlew+1)
            rj = (j-middleh) * np.pi / (middleh+1)

            radius[j,i] = np.sqrt(np.square(ri)+np.square(rj))
            if (ri == 0) and (rj == 0):
                theta[j,i] = 0
            elif (ri == 0) and (rj < 0):
                theta[j,i] = np.atan(-np.inf)
            elif (ri == 0) and (rj > 0):
                theta[j,i] = np.atan(np.inf)
            else:
                theta[j,i] = np.atan(rj/ri)

            if (ri < 0):
                if rj <= 0:
                    theta[j,i] = theta[j,i] - np.pi
                else:
                    theta[j,i] = theta[j,i] + np.pi

                if theta[j,i] < -3 * np.pi /4:
                    theta[j,i] = theta[j,i] + 2 * np.pi
                
    return theta, radius

def ewt2d_curvelet_scaling(radius,w1,gamma,W,H):

    """ Build the scaling function in the Fourier domain

    Parameters
    ----------
    - radius: ndarray
        Radius at each pixel of the polar grid corresponding to the image
    - w1: radius of the scaling domain
    - gamma: ratio for the spread of the scale transition zone
    - W, H: width and height of the image

    Returns
    -------
    - yms: ndarray
        Matrix containing the scaling filter

    Notes
    -----
    Generate the 2D Littlewood-Paley wavelet in the Fourier domain associated 
    to the disk [0,w1] with transition ratio gamma.
    
    Author: Jerome Gilles
    Institution: San Diego State University
    Version: 1.0 (01/30/2025)  
    """

    an = 1/(2*gamma*w1)
    pbn = (1+gamma)*w1
    mbn = (1-gamma)*w1

    yms = np.zeros((H,W)).astype(np.complex128)

    for i in np.arange(W):
        for j in np.arange(H):
            if (radius[j,i] < mbn):
                yms[j,i] = 1
            elif (radius[j,i] >= mbn) and (radius[j,i] <= pbn):
                yms[j,i] = np.cos(np.pi * beta(an * (radius[j,i]-mbn))/2)

    return yms

def ewt2d_angular_sector(theta, radius, theta0, theta1, r0, r1, gammaw, Dtheta):

    """ Build curvelet filter on a given angular sector

    Parameters
    ----------
    - theta: ndarray
        Angles at each pixel of the polar grid corresponding to the image
    - radius: ndarray
        Radius at each pixel of the polar grid corresponding to the image
    - theta0, theta1: angles defining the angular edges (must have theta0 < theta1)
    - r0, r1, radius defining the edges of the angular sector (must have r0 < r1 <= pi)
    - gammaw: ratio for the spread of the scale transition zones
    - Dtheta: ratio for the spread of the angle transition zones

    Returns
    - angular: ndarray
        Matrix containing the curvelet filter

    Notes
    -----
    This function creates the curvelet filter in the Fourier domain ((0,0) frequency
    being in the center of the image) which has a support defined in polar coordinates 
    (r,angles) in 
        [(1-gammaw)r0,(1+gammaw)r1]x[theta0-Dtheta,theta1+Dtheta],
    and 
        [(1-gammaw)r0,(1+gammaw)r1]x[theta0-Dtheta+pi,theta1+Dtheta+pi],
    respectively.

    Author: Jerome Gilles
    Institution: San Diego State University
    Version: 1.0 (01/30/2025)  
    """


    (H, W) = np.shape(theta)

    wan = 1/(2*gammaw*r0)
    wam = 1/(2*gammaw*r1)
    wpbn = (1+gammaw)*r0
    wmbn = (1-gammaw)*r0
    wpbm = (1+gammaw)*r1
    wmbm = (1-gammaw)*r1

    an = 1/(2*Dtheta)
    pbn = theta0 + Dtheta
    mbn = theta0 - Dtheta
    pbm = theta1 + Dtheta
    mbm = theta1 - Dtheta

    angular = np.zeros((H,W)).astype(np.complex128)

    if r1 < np.pi:
        for i in np.arange(W):
            for j in np.arange(H):
                if (theta[j,i] > pbn) and (theta[j,i] < mbm):
                    if (radius[j,i] > wpbn) and (radius[j,i] < wmbm):
                        angular[j,i] = 1
                    elif (radius[j,i] >= wmbm) and (radius[j,i] <= wpbm):
                        angular[j,i] = np.cos(np.pi * beta(wam * (radius[j,i]-wmbm))/2)
                    elif (radius[j,i] >= wmbn) and (radius[j,i] <= wpbn):
                        angular[j,i] = np.sin(np.pi * beta(wan * (radius[j,i]-wmbn))/2)
                elif (theta[j,i] >= mbn) and (theta[j,i] <= pbn):
                    if (radius[j,i] > wpbn) and (radius[j,i] < wmbm):
                        angular[j,i] = np.sin(np.pi * beta(an * (theta[j,i]-mbn))/2)
                    elif (radius[j,i] > wmbm) and (radius[j,i] <= wpbm):
                        angular[j,i] = np.sin(np.pi * beta(an * (theta[j,i]-mbn))/2) * np.cos(np.pi * beta(wam * (radius[j,i]-wmbm))/2)
                    elif (radius[j,i] >= wmbn) and (radius[j,i] <= wpbn):
                        angular[j,i] = np.sin(np.pi * beta(an * (theta[j,i]-mbn))/2) * np.sin(np.pi * beta(wan * (radius[j,i]-wmbn))/2)
                elif (theta[j,i] >= mbm) and (theta[j,i] <= pbm):
                    if (radius[j,i] > wpbn) and (radius[j,i] < wmbm):
                        angular[j,i] = np.cos(np.pi * beta(an * (theta[j,i]-mbm))/2)
                    elif (radius[j,i] > wmbm) and (radius[j,i] <= wpbm):
                        angular[j,i] = np.cos(np.pi * beta(an * (theta[j,i]-mbm))/2) * np.cos(np.pi * beta(wam * (radius[j,i]-wmbm))/2)
                    elif (radius[j,i] >= wmbn) and (radius[j,i] <= wpbn):
                        angular[j,i] = np.cos(np.pi * beta(an * (theta[j,i]-mbm))/2) * np.sin(np.pi * beta(wan * (radius[j,i]-wmbn))/2)
    else:
        for i in np.arange(W):
            for j in np.arange(H):
                if (theta[j,i] > pbn) and (theta[j,i] <mbm):
                    if (radius[j,i] > wpbn):
                        angular[j,i] = 1
                    elif (radius[j,i] >= wmbn) and (radius[j,i] <= wpbn):
                        angular[j,i] = np.sin(np.pi * beta(wan * (radius[j,i]-wmbn))/2)
                elif (theta[j,i] >= mbn) and (theta[j,i] <= pbn):
                    if (radius[j,i] > wpbn):
                        angular[j,i] = np.sin(np.pi * beta(an * (theta[j,i]-mbn))/2)
                    elif (radius[j,i] >= wmbn) and (radius[j,i] <= wpbn):
                        angular[j,i] = np.sin(np.pi * beta(an * (theta[j,i]-mbn))/2) * np.sin(np.pi * beta(wan * (radius[j,i]-wmbn))/2)
                elif (theta[j,i] >= mbm) and (theta[j,i] <= pbm):
                    if (radius[j,i] > wpbn):
                        angular[j,i] = np.cos(np.pi * beta(an * (theta[j,i]-mbm))/2)
                    elif (radius[j,i] >= wmbn) and (radius[j,i] <= wpbn):
                        angular[j,i] = np.cos(np.pi * beta(an * (theta[j,i]-mbm))/2) * np.sin(np.pi * beta(wan * (radius[j,i]-wmbn))/2)

    angular = angular + np.fliplr(np.flipud(angular))

    return angular

# =============================================================
#                     PLOTTING ROUTINES
# =============================================================
def ewt2d_curvelet_plot_filters(mfb, option):

    """ Plot the empirical curvelet filters in the Fourier domain

    Parameters
    ----------
    - mfb: list
        The first element is the scaling filter. The next elements are themselves lists
        containing the curvelet filters
    - option: 1, 2 or 3
        Type of empirical curvelets that was used in the transform
    
    Author: Jerome Gilles
    Institution: San Diego State University
    Version: 1.0 (01/30/2025)  
    """

    # plot the scaling filter first
    fig = plt.figure(figsize=(3.5,4))
    plt.imshow(np.fft.fftshift(np.real(mfb[0])),cmap='gray',interpolation='none')
    plt.axis('off')
    plt.title('scale=0')
    plt.show()

    if option == 1:
        for s in np.arange(len(mfb)-1):
            # compute the structure of the grid
            if len(mfb[s+1]) <= 3:
                nc = len(mfb[s+1])
                nr = 1
                figheight = 3.5
                figwidth = nc * 3
            else:
                nc = 3
                nr = int(np.ceil(len(mfb[s+1])/nc))
                figheight = nr * 3.5
                figwidth = nc * 3

            fig, ax = plt.subplots(nr, nc, figsize=(figwidth,figheight))

            k = 0
            if nr == 1:
                for kc in np.arange(nc):
                    ax[kc].imshow(np.fft.fftshift(np.real(mfb[s+1][k])),cmap='gray',interpolation='none')
                    ax[kc].set_title("scale="+str(s+1)+" ; theta="+str(k))
                    ax[kc].axis('off')
                    k = k+1
                    if k >= len(mfb):
                        break
            else:
                for kr in np.arange(nr):
                    for kc in np.arange(nc):
                        ax[kr,kc].imshow(np.fft.fftshift(np.real(mfb[s+1][k])),cmap='gray',interpolation='none')
                        ax[kr,kc].set_title("scale="+str(s+1)+" ; theta="+str(k))
                        ax[kr,kc].axis('off')
                        k = k+1
                        if k >= len(mfb[s+1]):
                            break
                    if k >= len(mfb[s+1]):
                        break

            # Remove the empty plots
            for ax in ax.flat[len(mfb[s+1]):]:
                ax.remove()

            plt.tight_layout()
            plt.show()
    elif option == 2:
        for s in np.arange(len(mfb)-1):
            # compute the structure of the grid
            if len(mfb[s+1]) <= 3:
                nc = len(mfb[s+1])
                nr = 1
                figheight = 3.5
                figwidth = nc * 3
            else:
                nc = 3
                nr = int(np.ceil(len(mfb[s+1])/nc))
                figheight = nr * 3.5
                figwidth = nc * 3

            fig, ax = plt.subplots(nr, nc, figsize=(figwidth,figheight))

            k = 0
            if nr == 1:
                for kc in np.arange(nc):
                    ax[kc].imshow(np.fft.fftshift(np.real(mfb[s+1][k])),cmap='gray',interpolation='none')
                    ax[kc].set_title("scale="+str(s+1)+" ; theta="+str(k))
                    ax[kc].axis('off')
                    k = k+1
                    if k >= len(mfb[s+1]):
                        break
            else:
                for kr in np.arange(nr):
                    for kc in np.arange(nc):
                        ax[kr,kc].imshow(np.fft.fftshift(np.real(mfb[s+1][k])),cmap='gray',interpolation='none')
                        ax[kr,kc].set_title("scale="+str(s+1)+" ; theta="+str(k))
                        ax[kr,kc].axis('off')
                        k = k+1
                        if k >= len(mfb[s+1]):
                            break
                    if k >= len(mfb[s+1]):
                        break

            # Remove the empty plots
            for ax in ax.flat[len(mfb[s+1]):]:
                ax.remove()

            plt.tight_layout()
            plt.show()
    elif option == 3:
        for s in np.arange(len(mfb)-1):
            # compute the structure of the grid
            if len(mfb[s+1]) <= 3:
                nc = len(mfb[s+1])
                nr = 1
                figheight = 3.5
                figwidth = nc * 3
            else:
                nc = 3
                nr = int(np.ceil(len(mfb[s+1])/nc))
                figheight = nr * 3.5
                figwidth = nc * 3

            fig, ax = plt.subplots(nr, nc, figsize=(figwidth,figheight))

            k = 0
            if nr == 1:
                for kc in np.arange(nc):
                    ax[kc].imshow(np.fft.fftshift(np.real(mfb[s+1][k])),cmap='gray',interpolation='none')
                    ax[kc].set_title("theta="+str(s+1)+" ; scale="+str(k))
                    ax[kc].axis('off')
                    k = k+1
                    if k >= len(mfb[s+1]):
                        break
            else:
                for kr in np.arange(nr):
                    for kc in np.arange(nc):
                        ax[kr,kc].imshow(np.fft.fftshift(np.real(mfb[s+1][k])),cmap='gray',interpolation='none')
                        ax[kr,kc].set_title("theta="+str(s+1)+" ; scale="+str(k))
                        ax[kr,kc].axis('off')
                        k = k+1
                        if k >= len(mfb[s+1]):
                            break
                    if k >= len(mfb[s+1]):
                        break

            # Remove the empty plots
            for ax in ax.flat[len(mfb[s+1]):]:
                ax.remove()

            plt.tight_layout()
            plt.show()

    return

def plot_curvelet_sectors(f,BW,BT,option,logtag=True,title="Curvelet Fourier supports"):

    """ Plot the rings delineating the curvelet supports

    Parameters
    ----------
    - f: 2D nparray
        Input image
    - BW: ndarray
        Array that contain the detected radius.
    - BT: ndarray
        Array that contain the detected angle.
    - option: 1, 2 or 3
        Type of empirical curvelets that was used in the transform
    - logtag: True (default) or False
        Indicate if the logarithm of the spectrum should be used for the background.
    - title: string
        Title to be plotted on the figure. The default title is "Curvelet Fourier supports".

    Notes
    -----
    This function plots the position of the detected curvelet supports in the Fourier domain.
    
    Author: Jerome Gilles
    Institution: San Diego State University
    Version: 1.0 (01/30/2025)    
    """

    ft = np.fft.fftshift(np.abs(np.fft.fft2(f)))

    if logtag:
        ft = np.log(1+ft)
    
    figb, axb = plt.subplots(figsize=(10,7))
    axb.imshow(ft, cmap="gray", interpolation='none')

    (r,c) = np.shape(f)

    # find center coordinates
    po = [round_away(np.shape(ft)[0]/2), round_away(np.shape(ft)[1]/2)]

    if option ==1:
        #  draw the rings
        a = BW * np.shape(ft)[0] / np.pi
        b = BW * np.shape(ft)[1] / np.pi
        for k in np.arange(np.size(BW)):
            axb.add_patch(Ellipse((po[1],po[0]),b[k],a[k],color='red',fill=False))

        # draw the angular sector edges
        p0 = np.zeros(2)
        p1 = np.zeros(2)
        for n in np.arange(np.size(BT)):
            if np.abs(BT[n]) <= np.pi/4:
                p0[0] = po[0] * (1+BW[0]*np.cos(BT[n])/np.pi)
                p0[1] = po[1] * (1+BW[0]*np.sin(BT[n])/np.pi)
                p1[0] = c-1
                p1[1] = np.ceil((r+c*np.tan(BT[n]))/2)
                plt.plot([p1[0],p0[0]],[p1[1],p0[1]],color='red')
                p0[0] = po[0] * (1-BW[0]*np.cos(BT[n])/np.pi)
                p0[1] = po[1] * (1-BW[0]*np.sin(BT[n])/np.pi)
                p1[0] = 0
                p1[1] = np.ceil((r-c*np.tan(BT[n]))/2)
                plt.plot([p1[0],p0[0]],[p1[1],p0[1]],color='red')
            else:
                p0[0] = po[0] * (1-BW[0]*np.cos(BT[n])/np.pi)
                p0[1] = po[1] * (1-BW[0]*np.sin(BT[n])/np.pi)
                p1[0] = np.ceil((r+c/np.tan(BT[n]))/2)
                p1[1] = r-1
                plt.plot([p1[0],p0[0]],[p1[1],p0[1]],color='red')
                p0[0] = po[0] * (1+BW[0]*np.cos(BT[n])/np.pi)
                p0[1] = po[1] * (1+BW[0]*np.sin(BT[n])/np.pi)
                p1[0] = np.ceil((r-c/np.tan(BT[n]))/2)
                p1[1] = 0
                plt.plot([p1[0],p0[0]],[p1[1],p0[1]],color='red')

        plt.axis('off')
        plt.title(title)
        plt.show()
    elif option == 2:
        #  draw the rings
        a = BW * np.shape(ft)[0] / np.pi
        b = BW * np.shape(ft)[1] / np.pi
        for k in np.arange(np.size(BW)):
            axb.add_patch(Ellipse((po[1],po[0]),b[k],a[k],color='red',fill=False))

        # draw the angular sectors per ring
        p0 = np.zeros(2)
        p1 = np.zeros(2)
        for s in np.arange(np.size(BW)-1):
            for n in np.arange(np.size(BT[s])):
                if np.abs(BT[s][n]) <= np.pi/4:
                    p0[0] = po[0] + BW[s]*np.ceil(c/(2*np.pi))*np.cos(BT[s][n])
                    p0[1] = po[1] + BW[s]*np.ceil(c/(2*np.pi))*np.sin(BT[s][n])
                    p1[0] = po[0] + BW[s+1]*np.floor(c/(2*np.pi))*np.cos(BT[s][n])+1
                    p1[1] = po[0] + BW[s+1]*np.floor(c/(2*np.pi))*np.sin(BT[s][n])+1
                    plt.plot([p1[0],p0[0]],[p1[1],p0[1]],color='red')
                    p0[0] = po[0] - BW[s]*np.ceil(c/(2*np.pi))*np.cos(BT[s][n])
                    p0[1] = po[1] - BW[s]*np.ceil(c/(2*np.pi))*np.sin(BT[s][n])
                    p1[0] = po[0] - BW[s+1]*np.floor(c/(2*np.pi))*np.cos(BT[s][n])-1
                    p1[1] = po[0] - BW[s+1]*np.floor(c/(2*np.pi))*np.sin(BT[s][n])-1
                    plt.plot([p1[0],p0[0]],[p1[1],p0[1]],color='red')
                else:
                    p0[0] = po[0] - BW[s]*np.ceil(c/(2*np.pi))*np.cos(BT[s][n])
                    p0[1] = po[1] - BW[s]*np.ceil(c/(2*np.pi))*np.sin(BT[s][n])
                    p1[0] = po[0] - BW[s+1]*np.floor(c/(2*np.pi))*np.cos(BT[s][n])+1
                    p1[1] = po[0] - BW[s+1]*np.floor(c/(2*np.pi))*np.sin(BT[s][n])+1
                    plt.plot([p1[0],p0[0]],[p1[1],p0[1]],color='red')
                    p0[0] = po[0] + BW[s]*np.ceil(c/(2*np.pi))*np.cos(BT[s][n])
                    p0[1] = po[1] + BW[s]*np.ceil(c/(2*np.pi))*np.sin(BT[s][n])
                    p1[0] = po[0] + BW[s+1]*np.floor(c/(2*np.pi))*np.cos(BT[s][n])-1
                    p1[1] = po[0] + BW[s+1]*np.floor(c/(2*np.pi))*np.sin(BT[s][n])-1
                    plt.plot([p1[0],p0[0]],[p1[1],p0[1]],color='red')

        for n in np.arange(np.size(BT[-1])):
            if np.abs(BT[-1][n]) <= np.pi/4:
                p0[0] = po[0] + BW[-1]*np.ceil(c/(2*np.pi))*np.cos(BT[-1][n])+1
                p0[1] = po[1] + BW[-1]*np.ceil(c/(2*np.pi))*np.sin(BT[-1][n])+1
                p1[0] = c-1
                p1[1] = np.ceil((r+c*np.tan(BT[-1][n]))/2)
                plt.plot([p1[0],p0[0]],[p1[1],p0[1]],color='red')
                p0[0] = po[0] - BW[-1]*np.ceil(c/(2*np.pi))*np.cos(BT[-1][n])-1
                p0[1] = po[1] - BW[-1]*np.ceil(c/(2*np.pi))*np.sin(BT[-1][n])-1
                p1[0] = 0
                p1[1] = np.ceil((r-c*np.tan(BT[-1][n]))/2)
                plt.plot([p1[0],p0[0]],[p1[1],p0[1]],color='red')
            else:
                p0[0] = po[0] - BW[-1]*np.ceil(c/(2*np.pi))*np.cos(BT[-1][n])-1
                p0[1] = po[1] - BW[-1]*np.ceil(c/(2*np.pi))*np.sin(BT[-1][n])-1
                p1[0] = np.ceil((r+c/np.tan(BT[-1][n]))/2)
                p1[1] = r-1
                plt.plot([p1[0],p0[0]],[p1[1],p0[1]],color='red')
                p0[0] = po[0] + BW[-1]*np.ceil(c/(2*np.pi))*np.cos(BT[-1][n])+1
                p0[1] = po[1] + BW[-1]*np.ceil(c/(2*np.pi))*np.sin(BT[-1][n])+1
                p1[0] = np.ceil((r-c/np.tan(BT[-1][n]))/2)
                p1[1] = 0
                plt.plot([p1[0],p0[0]],[p1[1],p0[1]],color='red')            

        plt.axis('off')
        plt.title(title)
        plt.show()
    elif option == 3:
        #  draw the first ring
        a = BW[0] * np.shape(ft)[0] / np.pi
        b = BW[0] * np.shape(ft)[1] / np.pi
        axb.add_patch(Ellipse((po[1],po[0]),b,a,color='red',fill=False))

        # draw the angular sector edges
        p0 = np.zeros(2)
        p1 = np.zeros(2)
        for n in np.arange(np.size(BT)):
            if np.abs(BT[n]) <= np.pi/4:
                p0[0] = po[0] * (1+BW[0]*np.cos(BT[n])/np.pi)
                p0[1] = po[1] * (1+BW[0]*np.sin(BT[n])/np.pi)
                p1[0] = c-1
                p1[1] = np.ceil((r+c*np.tan(BT[n]))/2)
                plt.plot([p1[0],p0[0]],[p1[1],p0[1]],color='red')
                p0[0] = po[0] * (1-BW[0]*np.cos(BT[n])/np.pi)
                p0[1] = po[1] * (1-BW[0]*np.sin(BT[n])/np.pi)
                p1[0] = 0
                p1[1] = np.ceil((r-c*np.tan(BT[n]))/2)
                plt.plot([p1[0],p0[0]],[p1[1],p0[1]],color='red')
            else:
                p0[0] = po[0] * (1-BW[0]*np.cos(BT[n])/np.pi)
                p0[1] = po[1] * (1-BW[0]*np.sin(BT[n])/np.pi)
                p1[0] = np.ceil((r+c/np.tan(BT[n]))/2)
                p1[1] = r-1
                plt.plot([p1[0],p0[0]],[p1[1],p0[1]],color='red')
                p0[0] = po[0] * (1+BW[0]*np.cos(BT[n])/np.pi)
                p0[1] = po[1] * (1+BW[0]*np.sin(BT[n])/np.pi)
                p1[0] = np.ceil((r-c/np.tan(BT[n]))/2)
                p1[1] = 0
                plt.plot([p1[0],p0[0]],[p1[1],p0[1]],color='red')

        # draw the arcs within the angular sector
        for t in np.arange(np.size(BT)-1):
            for s in np.arange(np.size(BW[t+1])):
                a = BW[t+1][s] * np.shape(ft)[0] / np.pi
                b = BW[t+1][s] * np.shape(ft)[1] / np.pi
                axb.add_patch(Arc((po[1],po[0]),b,a,theta1=180*BT[t]/np.pi,theta2=180*BT[t+1]/np.pi,color='red',fill=False))
                axb.add_patch(Arc((po[1],po[0]),b,a,theta1=180*(BT[t]+np.pi)/np.pi,theta2=180*(BT[t+1]+np.pi)/np.pi,color='red',fill=False))
        # last sector
        for s in np.arange(np.size(BW[-1])):
            a = BW[-1][s] * np.shape(ft)[0] / np.pi
            b = BW[-1][s] * np.shape(ft)[1] / np.pi
            axb.add_patch(Arc((po[1],po[0]),b,a,theta1=180*BT[-1]/np.pi,theta2=180*(BT[0]+np.pi)/np.pi,color='red',fill=False))
            axb.add_patch(Arc((po[1],po[0]),b,a,theta1=180*(BT[-1]+np.pi)/np.pi,theta2=180*(BT[0]+2*np.pi)/np.pi,color='red',fill=False))


        plt.axis('off')
        plt.title(title)
        plt.show()

    return

def ewt2d_curvelet_plot_comp(ewt, option, energy=False):

    """ Plot the empirical curvelet components

    Parameters
    ----------
    - ewt: list
        The first element is the scaling coefficient. The next elements are themselves lists
        containing the curvelet coefficients
    - option: 1, 2 or 3
        Type of empirical curvelets that was used in the transform
    - energy: True or False (default)
        If True, the energy will be indicated above each image

    Author: Jerome Gilles
    Institution: San Diego State University
    Version: 1.0 (01/30/2025)  
    """

    # plot the scaling filter first
    fig = plt.figure(figsize=(3.5,4))
    plt.imshow(np.real(ewt[0]),cmap='gray',interpolation='none')
    plt.axis('off')
    if energy:
        plt.title('scale=0 ; energy='+str(np.round(LA.norm(ewt[0]),2)),wrap=True)
    else:
        plt.title('scale=0')
    plt.show()

    if option == 1:
        for s in np.arange(len(ewt)-1):
            # compute the structure of the grid
            if len(ewt[s+1]) <= 3:
                nc = len(ewt[s+1])
                nr = 1
                figheight = 3.5
                figwidth = nc * 3
            else:
                nc = 3
                nr = int(np.ceil(len(ewt[s+1])/nc))
                figheight = nr * 3.5
                figwidth = nc * 3

            fig, ax = plt.subplots(nr, nc, figsize=(figwidth,figheight))            
            # plot the coefficients
            k = 0
            if nr == 1:
                for kc in np.arange(nc):
                    ax[kc].imshow(np.real(ewt[s+1][k]),cmap='gray',interpolation='none')
                    if energy:
                        ax[kc].set_title("scale="+str(s+1)+" ; theta="+str(k)+"\n energy="+str(np.round(LA.norm(ewt[s+1][k]),2)))
                    else:
                        ax[kc].set_title("scale="+str(s+1)+" ; theta="+str(k))
                    ax[kc].axis('off')
                    k = k+1
                    if k >= len(ewt):
                        break
            else:
                for kr in np.arange(nr):
                    for kc in np.arange(nc):
                        ax[kr,kc].imshow(np.real(ewt[s+1][k]),cmap='gray',interpolation='none')
                        if energy:
                            ax[kr,kc].set_title("scale="+str(s+1)+" ; theta="+str(k)+"\n energy="+str(np.round(LA.norm(ewt[s+1][k]),2)))
                        else:
                            ax[kr,kc].set_title("scale="+str(s+1)+" ; theta="+str(k))
                        ax[kr,kc].axis('off')
                        k = k+1
                        if k >= len(ewt[s+1]):
                            break
                    if k >= len(ewt[s+1]):
                        break

            # Remove the empty plots
            for ax in ax.flat[len(ewt[s+1]):]:
                ax.remove()

            plt.tight_layout()
            plt.subplots_adjust(hspace=0.25)
            plt.show()
    elif option == 2:
        for s in np.arange(len(ewt)-1):
            # compute the structure of the grid
            if len(ewt[s+1]) <= 3:
                nc = len(ewt[s+1])
                nr = 1
                figheight = 3.5
                figwidth = nc * 3
            else:
                nc = 3
                nr = int(np.ceil(len(ewt[s+1])/nc))
                figheight = nr * 3.5
                figwidth = nc * 3

            fig, ax = plt.subplots(nr, nc, figsize=(figwidth,figheight))

            # plot the coefficients
            k = 0
            if nr == 1:
                for kc in np.arange(nc):
                    ax[kc].imshow(np.real(ewt[s+1][k]),cmap='gray',interpolation='none')
                    if energy:
                        ax[kc].set_title("scale="+str(s+1)+" ; theta="+str(k)+"\n energy="+str(np.round(LA.norm(ewt[s+1][k]),2)))
                    else:
                        ax[kc].set_title("scale="+str(s+1)+" ; theta="+str(k))
                    ax[kc].axis('off')
                    k = k+1
                    if k >= len(ewt[s+1]):
                        break
            else:
                for kr in np.arange(nr):
                    for kc in np.arange(nc):
                        ax[kr,kc].imshow(np.real(ewt[s+1][k]),cmap='gray',interpolation='none')
                        if energy:
                            ax[kr,kc].set_title("scale="+str(s+1)+" ; theta="+str(k)+"\n energy="+str(np.round(LA.norm(ewt[s+1][k]),2)))
                        else:
                            ax[kr,kc].set_title("scale="+str(s+1)+" ; theta="+str(k))
                        ax[kr,kc].axis('off')
                        k = k+1
                        if k >= len(ewt[s+1]):
                            break
                    if k >= len(ewt[s+1]):
                        break

            # Remove the empty plots
            for ax in ax.flat[len(ewt[s+1]):]:
                ax.remove()

            plt.tight_layout()
            plt.subplots_adjust(hspace=0.25)
            plt.show()
    elif option == 3:
        for s in np.arange(len(ewt)-1):
            # compute the structure of the grid
            if len(ewt[s+1]) <= 3:
                nc = len(ewt[s+1])
                nr = 1
                figheight = 3.5
                figwidth = nc * 3
            else:
                nc = 3
                nr = int(np.ceil(len(ewt[s+1])/nc))
                figheight = nr * 3.5
                figwidth = nc * 3

            fig, ax = plt.subplots(nr, nc, figsize=(figwidth,figheight))

            # plot the coefficients
            k = 0
            if nr == 1:
                for kc in np.arange(nc):
                    ax[kc].imshow(np.real(ewt[s+1][k]),cmap='gray',interpolation='none')
                    if energy:
                        ax[kc].set_title("theta="+str(s)+" ; scale="+str(k)+"\n energy="+str(np.round(LA.norm(ewt[s+1][k]),2)))
                    else:
                        ax[kc].set_title("theta="+str(s)+" ; scale="+str(k))
                    ax[kc].axis('off')
                    k = k+1
                    if k >= len(ewt[s+1]):
                        break
            else:
                for kr in np.arange(nr):
                    for kc in np.arange(nc):
                        ax[kr,kc].imshow(np.real(ewt[s+1][k]),cmap='gray',interpolation='none')
                        if energy:
                            ax[kr,kc].set_title("theta="+str(s)+" ; scale="+str(k)+"\n energy="+str(np.round(LA.norm(ewt[s+1][k]),2)))
                        else:
                            ax[kr,kc].set_title("theta="+str(s)+" ; scale="+str(k))
                        ax[kr,kc].axis('off')
                        k = k+1
                        if k >= len(ewt[s+1]):
                            break
                    if k >= len(ewt[s+1]):
                        break

            # Remove the empty plots
            for ax in ax.flat[len(ewt[s+1]):]:
                ax.remove()

            plt.tight_layout()
            plt.subplots_adjust(hspace=0.25)
            plt.show()

    return