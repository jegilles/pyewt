import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.signal import hilbert
from scipy.interpolate import CubicSpline

from .boundaries1d import boundaries_detect
from .usefullfunc import round_away , beta


# ===========================================================================
#                      1D EMPIRICAL WAVELET TRANSFORM
# ===========================================================================
def ewt1d(f, params):

    """ 1D Empirical Wavelet Transform

    Parameters
    ----------
    - f: ndarray
        input signal.
    - params: Dictionary
        must be set properly accordingly to the options described
        in the notes section below.

    Returns
    -------
    - ewt: list
        each element of that list is a ndarray containing the corresponding
        components of the EWT (note, they could be either real of complex depending
        on the used mother wavelet).
    - mfb: list
        each element of that list is a ndarray containing the different filters 
        built accordingly to the detected supports and the chosen mother wavelet.
    - boundaries: ndarray
        list of boundaries normalized in [0,pi) if f is real, or [0,2pi) if f is 
        complex (i.e. no fftshift has been applied).
    
    Notes
    -----
    This function performs the 1D Empirical Wavelet Transform (EWT) of the input signal f,
    using the parameters provided in params. It has three main steps: 1) detect the supports
    boundaries, 2) build the wavelet filter bank, 3) perform the transform itself. The options
    for steps 1) and 2) are described hereafter.

    STEP 1): boundary detection
    ---------------------------
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
    is the number of detected modes is lower than an expected value params["N"]. 
    If needed, the last high frequency support is evenly splitted. This 
    step is performed if params["Completion"] is set to True.

    STEP 2): construction of the wavelet filter bank
    The name of the wanted mother wavelet must be given in params["wavname"]. 
    The available options are:
        - "littlewood-paley"
        - "meyer"
        - "shannon"
        - "gabor1"
        - "gabor 2"


    Author: Jerome Gilles
    Institution: San Diego State University
    Version: 1.0 (01/07/2025)
    """

    # compute Fourier transform
    ff = np.fft.fft(f)

    # extract the boundaries
    if np.isrealobj(f) == True:
        boundaries, presig = boundaries_detect(np.abs(ff[0:int(round_away((np.size(ff))/2))]),params)

        if boundaries[0] == 0:
            boundaries = boundaries[1:]

        boundaries = boundaries * np.pi / round_away(np.size(ff)/2)
    else:
        boundaries, presig = boundaries_detect(np.abs(ff),params)
        boundaries = boundaries * 2 * np.pi / np.size(ff)

        # if a boundary is too close to pi, we remove it
        boundaries = boundaries[np.where(np.abs(boundaries-np.pi) >= (10 * np.pi / np.size(f)))]

    # We build the corresponding filter bank, available mother wavelets:
    # Littlewood-Paley (default), Shannon, Gabor1, Gabor2, Meyer
    if params["wavname"] == "shannon":
        mfb = shannon_filterbank(boundaries, np.size(ff), np.iscomplexobj(f))
    elif params["wavname"] == "gabor1":
        mfb = gabor1_filterbank(boundaries, np.size(ff), np.iscomplexobj(f))
    elif params["wavname"] == "gabor2":
        mfb = gabor2_filterbank(boundaries, np.size(ff), np.iscomplexobj(f))
    elif params["wavname"] == "meyer":
        mfb = meyer_filterbank(boundaries, np.size(ff), np.iscomplexobj(f))
    else:
        mfb = lp_filterbank(boundaries, np.size(ff), np.iscomplexobj(f))
    
    ewt = []
    if params['wavname'] == "littlewood-paley":
        if np.isrealobj(f) == True:
            for k in np.arange(np.shape(mfb)[0]):
                tmp = np.real(np.fft.ifft(np.multiply(np.conjugate(mfb[k]),ff)))
                ewt.append(tmp)
        else:
            for k in np.arange(np.shape(mfb)[0]):
                tmp = np.fft.ifft(np.multiply(np.conjugate(mfb[k]),ff))
                ewt.append(tmp)
    else:
        for k in np.arange(np.shape(mfb)[0]):
            tmp = np.fft.ifft(np.multiply(np.conjugate(mfb[k]),ff))
            ewt.append(tmp)

    return ewt, mfb, boundaries


# ===========================================================================
#                              MOTHER WAVELETS
# ===========================================================================
# Littlewood - Paley
def lp_filterbank(boundaries, N, cpx):

    """ Littlewood-Paley wavelet filters

    Parameters
    ----------
    - boundaries: ndarray containing the support boundaries to be used.
        It is expected that these boundaries lie in [0,pi) for a real case,
        and [0,2pi) for a complex case.
    - N: number of samples per filter.
        Should correspond to the length of the Fourier transform of the signal 
        used to detect the boundaries
    - cpx: True if the original signal was complex, False otherwise.

    Returns
    -------
    - mfb: list
        Each element of that list is a ndarray corresponding to each filter

    Notes
    -----
    This function builds the set of Littlewood-Paley filters corresponding to the given set
    of boundaries.

    Author: Jerome Gilles
    Institution: San Diego State University
    Version: 1.0 (01/07/2025)
    """

    Npic = np.size(boundaries)

    # create the frequency axis [0,pi)u[-pi,0)
    Mi = int(np.floor((N+1)/2))

    
    w = 2 * np.pi * np.arange(N) / N
    w[Mi:] = -2 * np.pi + w[Mi:]

    if cpx == False:
        # we compute gamma accordingly to the theory
        r = np.divide(np.hstack((boundaries[1:], np.pi)) - boundaries,np.hstack((boundaries[1:], np.pi)) + boundaries)
        gamma = np.min(r)
        gamma = (1-1/N) * gamma
        mfb = []
        # we start by generating the scaling function
        am = 1 / (2 * gamma * np.abs(boundaries[0]))
        pbm = (1 + gamma) * boundaries[0]
        mbm = (1 - gamma) * boundaries[0]

        tmp = np.zeros(N)
        for k in np.arange(N):
            if np.abs(w[k]) <= mbm:
                tmp[k] = 1
            elif (np.abs(w[k]) >= mbm) and (np.abs(w[k]) <= pbm):
                tmp[k] = np.cos(np.pi * beta(am * (np.abs(w[k]) - mbm))/2)
        mfb.append(tmp)

        # we generate the wavelets except the last one
        for l in np.arange(Npic-1):
            tmp = np.zeros(N)

            an = 1 / (2 * gamma * np.abs(boundaries[l]))
            pbn = (1 + gamma) * boundaries[l]
            mbn = (1 - gamma) * boundaries[l]

            am = 1 / (2 * gamma * np.abs(boundaries[l+1]))
            pbm = (1 + gamma) * boundaries[l+1]
            mbm = (1 - gamma) * boundaries[l+1]

            for k in np.arange(N):
                if (np.abs(w[k]) <= mbm) and (np.abs(w[k]) >= pbn):
                    tmp[k] = 1
                elif (np.abs(w[k]) >= mbm) and (np.abs(w[k]) <= pbm):
                    tmp[k] = np.cos(np.pi * beta(am * (np.abs(w[k]) - mbm))/2)
                elif (np.abs(w[k]) >= mbn) and (np.abs(w[k]) <= pbn):
                    tmp[k] = np.sin(np.pi * beta(an * (np.abs(w[k]) - mbn))/2)

            mfb.append(tmp)

        # last wavelet
        am = 1 / (2 * gamma * np.abs(boundaries[-1]))
        pbm = (1 + gamma) * boundaries[-1]
        mbm = (1 - gamma) * boundaries[-1]

        tmp = np.zeros(N)
        for k in np.arange(N):
            if np.abs(w[k]) >= pbm:
                tmp[k] = 1
            elif (np.abs(w[k]) >= mbm) and (np.abs(w[k]) <= pbm):
                tmp[k] = np.sin(np.pi * beta(am * (np.abs(w[k]) - mbm))/2)
        
        mfb.append(tmp)
    else:
        indgpi = np.argwhere(boundaries >= np.pi)
        boundaries[indgpi] = -2 * np.pi + boundaries[indgpi]
        
        # we compute gamma accordingly to the theory
        sb = np.hstack((boundaries[np.where(boundaries < 0)], boundaries[np.where(boundaries >= 0)]))
        r = np.divide(np.hstack((sb, np.pi)) - np.hstack((-np.pi, sb)),np.abs(np.hstack((sb, np.pi)) + np.hstack((-np.pi, sb))))
        gamma = np.min(r)
        gamma = (1-1/N) * gamma

        mfb = []

        # we start by generating the scaling function
        # (here we assume 0 cannot be a possible boundary - May change in the future)
        an = 1 / (2 * gamma * np.abs(boundaries[-1]))
        pbn = (1 + np.sign(boundaries[-1]) * gamma) * boundaries[-1]
        mbn = (1 - np.sign(boundaries[-1]) * gamma) * boundaries[-1]

        am = 1 / (2 * gamma * np.abs(boundaries[0]))
        pbm = (1 + np.sign(boundaries[0]) * gamma) * boundaries[0]
        mbm = (1 - np.sign(boundaries[0]) * gamma) * boundaries[0]

        tmp = np.zeros(N)
        for k in np.arange(N):
                if (w[k] <= mbm) and (w[k] >= pbn):
                    tmp[k] = 1
                elif (w[k] >= mbm) and (w[k] <= pbm):
                    tmp[k] = np.cos(np.pi * beta(am * (w[k] - mbm))/2)
                elif (w[k] >= mbn) and (w[k] <= pbn):
                    tmp[k] = np.sin(np.pi * beta(an * (w[k] - mbn))/2)
        mfb.append(tmp)

        # we generate the wavelets except the last one
        for l in np.arange(Npic-1):
            if boundaries[l] > boundaries[l+1]:
                # high pass filter
                tmp = np.zeros(N)

                an = 1 / (2 * gamma * np.abs(boundaries[l]))
                pbn = (1 + np.sign(boundaries[l]) * gamma) * boundaries[l]
                mbn = (1 - np.sign(boundaries[l]) * gamma) * boundaries[l]

                for k in np.arange(N):
                    if (w[k] >= pbn):
                        tmp[k] = 1
                    elif (w[k] >= mbn) and (w[k] <= pbn):
                        tmp[k] = np.sin(np.pi * beta(an * (w[k] - mbn))/2)

                mfb.append(tmp)

                tmp = np.zeros(N)

                am = 1 / (2 * gamma * np.abs(boundaries[l+1]))
                pbm = (1 + np.sign(boundaries[l+1]) * gamma) * boundaries[l+1]
                mbm = (1 - np.sign(boundaries[l+1]) * gamma) * boundaries[l+1]

                for k in np.arange(N):
                    if (w[k] <= mbm):
                        tmp[k] = 1
                    elif (w[k] >= mbm) and (w[k] <= pbm):
                        tmp[k] = np.cos(np.pi * beta(am * (w[k] - mbm))/2)

                mfb.append(tmp)
            else:
                tmp = np.zeros(N)

                an = 1 / (2 * gamma * np.abs(boundaries[l]))
                pbn = (1 + np.sign(boundaries[l]) * gamma) * boundaries[l]
                mbn = (1 - np.sign(boundaries[l]) * gamma) * boundaries[l]

                am = 1 / (2 * gamma * np.abs(boundaries[l+1]))
                pbm = (1 + np.sign(boundaries[l+1]) * gamma) * boundaries[l+1]
                mbm = (1 - np.sign(boundaries[l+1]) * gamma) * boundaries[l+1]

                for k in np.arange(N):
                    if (w[k] <= mbm) and (w[k] >= pbn):
                        tmp[k] = 1
                    elif (w[k] >= mbm) and (w[k] <= pbm):
                        tmp[k] = np.cos(np.pi * beta(am * (w[k] - mbm))/2)
                    elif (w[k] >= mbn) and (w[k] <= pbn):
                        tmp[k] = np.sin(np.pi * beta(an * (w[k] - mbn))/2)

                mfb.append(tmp)
            
    return mfb

# Shannon wavelet
def shannon_filterbank(boundaries, N, cpx):

    """ Shannon wavelet filters

    Parameters
    ----------
    - boundaries: ndarray containing the support boundaries to be used.
        It is expected that these boundaries lie in [0,pi) for a real case,
        and [0,2pi) for a complex case.
    - N: number of samples per filter.
        Should correspond to the length of the Fourier transform of the signal 
        used to detect the boundaries
    - cpx: True if the original signal was complex, False otherwise.

    Returns
    -------
    - mfb: list
        Each element of that list is a ndarray corresponding to each filter

    Notes
    -----
    This function builds the set of Shannon filters corresponding to the given set
    of boundaries.

    Author: Jerome Gilles
    Institution: San Diego State University
    Version: 1.0 (01/07/2025)
    """

    Npic = np.size(boundaries)

    # create the frequency axis [0,pi)u[-pi,0)
    Mi = int(np.floor(N/2)) + 1
    
    w = 2 * np.pi * np.arange(N) / N
    w[Mi:] = -2 * np.pi + w[Mi:]

    if cpx == False:
        mfb = []

        # scaling function
        tmp = np.zeros(N)

        for k in np.arange(N):
            if np.abs(w[k]) < boundaries[0]:
                tmp[k] = 1
        mfb.append(tmp)

        # each wavelet except the last one
        for l in np.arange(Npic-1):
            tmp = np.zeros(N).astype(complex)

            w1 = boundaries[l+1] - boundaries[l]
            for k in np.arange(N):
                if (boundaries[l] <= np.abs(w[k])) and (np.abs(w[k]) < boundaries[l+1]):
                    tmp[k] = np.exp(-1j * np.pi * (w[k]+np.sign(w[k]) * (boundaries[l+1] - 2 * boundaries[l]))/(2 * w1))/np.sqrt(w1)
            mfb.append(tmp)
                    
        # last one
        tmp = np.zeros(N).astype(complex)

        w1 = np.pi - boundaries[-1]
        for k in np.arange(N):
            if boundaries[-1] <= np.abs(w[k]):
                tmp[k] = np.exp(-1j * np.pi * (w[k]+np.sign(w[k]) * (np.pi - 2*boundaries[-1]))/(2 * w1))/np.sqrt(w1)
        mfb.append(tmp)
    else:
        indgpi = np.argwhere(boundaries >= np.pi)
        boundaries[indgpi] = -2 * np.pi + boundaries[indgpi]

        mfb = []

        # scaling function
        tmp = np.zeros(N)

        for k in np.arange(N):
            if (boundaries[-1] <= w[k]) and (w[k] < boundaries[0]):
                tmp[k] = 1
        mfb.append(tmp)

        # each wavelet (high frequencies in a separate way)
        for l in np.arange(Npic-1):
            if boundaries[l] > boundaries[l+1]:  # high pass filters
                #positive frequencies
                tmp = np.zeros(N).astype(complex)
                w1 = np.pi - boundaries[l]
                for k in np.arange(N):
                    if boundaries[l] <= w[k]:
                        tmp[k] = np.exp(-1j * np.pi * (w[k]+np.sign(w[k]) * (np.pi - 2 * boundaries[l]))/(2 * w1))/np.sqrt(w1)
                mfb.append(tmp)

                #negative frequencies
                tmp = np.zeros(N).astype(complex)
                w1 = boundaries[l+1] + np.pi
                for k in np.arange(N):
                    if w[k] < boundaries[l+1]:
                        tmp[k] = np.exp(-1j * np.pi * (w[k] - 2 * boundaries[l+1] - np.pi)/(2 * w1))/np.sqrt(w1)
                mfb.append(tmp)
            else:  # individual wavelets
                tmp = np.zeros(N).astype(complex)
                w1 = boundaries[l+1] - boundaries[l]

                for k in np.arange(N):
                    if (boundaries[l] <= w[k]) and (w[k] < boundaries[l+1]):
                        if boundaries[l] >0:
                            tmp[k] = np.exp(-1j * np.pi * (w[k]+(boundaries[l+1]-2*boundaries[l]))/(2*w1))/np.sqrt(w1)
                        else:
                            tmp[k] = np.exp(-1j * np.pi * (w[k]+(boundaries[l]-2*boundaries[l+1]))/(2*w1))/np.sqrt(w1)
                mfb.append(tmp)

    return mfb

# Meyer wavelet
def meyer_filterbank(boundaries, N, cpx):

    """ Meyer wavelet filters

    Parameters
    ----------
    - boundaries: ndarray containing the support boundaries to be used.
        It is expected that these boundaries lie in [0,pi) for a real case,
        and [0,2pi) for a complex case.
    - N: number of samples per filter.
        Should correspond to the length of the Fourier transform of the signal 
        used to detect the boundaries
    - cpx: True if the original signal was complex, False otherwise.

    Returns
    -------
    - mfb: list
        Each element of that list is a ndarray corresponding to each filter

    Notes
    -----
    This function builds the set of Meyer filters corresponding to the given set
    of boundaries.

    Author: Jerome Gilles
    Institution: San Diego State University
    Version: 1.0 (01/07/2025)
    """

    Npic = np.size(boundaries)

    # create the frequency axis [0,pi)u[-pi,0)
    Mi = int(np.floor(N/2)) + 1
    
    w = 2 * np.pi * np.arange(N) / N
    w[Mi:] = -2 * np.pi + w[Mi:]

    mfb = []

    if cpx == False:
        # scaling function
        tmp = np.zeros(N)
        wm = (boundaries[0] + boundaries[1])/2
        for k in np.arange(N):
            tmp[k] = np.cos(np.pi * beta(np.abs(w[k])/wm)/2)/np.sqrt(wm)

        mfb.append(tmp)

        # each wavelet except last one
        for l in np.arange(Npic-1):
            tmp = np.zeros(N).astype(complex)

            wn = (boundaries[l+1] + boundaries[l])/2
            if l == 0:
                wm = boundaries[0]/2
            else:
                wm = (boundaries[l-1] + boundaries[l])/2

            if l == (Npic-2):
                wp = (np.pi + boundaries[l+1])/2
            else:
                wp = (boundaries[l+1] + boundaries[l+2])/2

            for k in np.arange(N):
                if (np.abs(w[k]) >= wm) and (np.abs(w[k]) <= wn):
                    tmp[k] = np.sqrt(2/(wp-wm)) * np.exp(1j * np.pi * np.sign(w[k]) * (np.abs(w[k]) + (wp-4*wm)/3)/(wp-wm)) * np.sin(np.pi * beta((np.abs(w[k])-wm)/(wn-wm))/2)                
                elif (np.abs(w[k]) >= wn) and (np.abs(w[k]) <= wp):
                    tmp[k] = np.sqrt(2/(wp-wm)) * np.exp(1j * np.pi * np.sign(w[k]) * (np.abs(w[k]) + (wp-4*wm)/3)/(wp-wm)) * np.cos(np.pi * beta((np.abs(w[k])-wn)/(wp-wn))/2)

            mfb.append(tmp)

        # last one
        tmp = np.zeros(N).astype(complex)

        wn = (np.pi + boundaries[-1])/2
        wm = (boundaries[-1] + boundaries[Npic-2])/2

        for k in np.arange(N):
            if (wm <= np.abs(w[k])) and (np.abs(w[k]) <= wn):
                tmp[k] = np.sqrt(2/(wn-wm)) * np.exp(1j * np.pi * np.sign(w[k]) * (np.abs(w[k]) + (np.pi-4*wm)/3)/(np.pi-wm)) * np.sin(np.pi * beta((np.abs(w[k])-wm)/(wn-wm))/2)
            elif (np.abs(w[k]) >= wn):
                tmp[k] = np.sqrt(2/(wn-wm)) * np.exp(1j * np.pi * np.sign(w[k]) * (np.abs(w[k]) + (np.pi-4*wm)/3)/(np.pi-wm))
        mfb.append(tmp)
    else:
        indgpi = np.argwhere(boundaries >= np.pi)
        boundaries[indgpi] = -2 * np.pi + boundaries[indgpi]

        # scaling function
        tmp = np.zeros(N)

        wn = (boundaries[-1] + boundaries[0])/2
        wm = (boundaries[Npic-2] + boundaries[-1])/2
        wp = (boundaries[0] + boundaries[1])/2

        for k in np.arange(N):
            if (w[k] >= wm) and (w[k] <= wn):
                tmp[k] = np.sqrt(2/(wp-wm)) * np.sin(np.pi * beta((w[k]-wm)/(wn-wm))/2)
            elif (w[k] >= wn) and (w[k] <= wp):
                tmp[k] = np.sqrt(2/(wp-wm)) * np.cos(np.pi * beta((w[k]-wn)/(wp-wn))/2)
        mfb.append(tmp)
        
        # each wavelet (high frequencies in separate way)
        for l in np.arange(Npic-1):
            if boundaries[l] > boundaries[l+1]:  # high pass filter
                # positive frequencies
                tmp = np.zeros(N).astype(complex)

                wn = (np.pi + boundaries[l])/2
                if l == 1:
                    wm = (boundaries[-1] + boundaries[l])/2
                else:
                    wm = (boundaries[l] + boundaries[l-1])/2
                
                for k in np.arange(N):
                    if (wm <= w[k]) and (w[k] <= wn):
                        if w[k] >= 0:
                            tmp[k] = np.sqrt(2/(wn-wm)) * np.exp(1j * np.pi * (w[k] + (np.pi-4*wm)/3)/(np.pi-wm)) * np.sin(np.pi * beta((w[k]-wm)/(wn-wm))/2)
                        else:
                            tmp[k] = np.sqrt(2/(wn-wm)) * np.exp(1j * np.pi * (w[k] + (wm-4*np.pi)/3)/(np.pi-wm)) * np.sin(np.pi * beta((w[k]-wm)/(wn-wm))/2)
                    elif w[k] >= wn:
                        if w[k] >= 0:
                            tmp[k] = np.sqrt(2/(wn-wm)) * np.exp(1j * np.pi * (w[k] + (np.pi-4*wm)/3)/(np.pi-wm))
                        else:
                            tmp[k] = np.sqrt(2/(wn-wm)) * np.exp(1j * np.pi * (w[k] + (wm-4*np.pi)/3)/(np.pi-wm))
                mfb.append(tmp)
        
                # negative frequencies
                tmp = np.zeros(N).astype(complex)

                wn = (-np.pi + boundaries[l+1])/2
                if (l+1) == (Npic-1):
                    wp = (boundaries[0] + boundaries[l+1])/2
                else:
                    wp = (boundaries[l+1] + boundaries[l+2])/2
        
                for k in np.arange(N):
                    if (wn <= w[k]) and (w[k] <= wp):
                        if w[k] >= 0:
                            tmp[k] = np.sqrt(2/(wp-wn)) * np.exp(1j * np.pi * (w[k] + (wp+4*np.pi)/3)/(wp+np.pi)) * np.cos(np.pi * beta((w[k]-wn)/(wp-wn))/2)
                        else:
                            tmp[k] = np.sqrt(2/(wp-wn)) * np.exp(1j * np.pi * (w[k] - (4*wp+np.pi)/3)/(wp+np.pi)) * np.cos(np.pi * beta((w[k]-wn)/(wp-wn))/2)
                    elif w[k] <= wn:
                        if w[k] >= 0:
                            tmp[k] = np.sqrt(2/(wp-wn)) * np.exp(1j * np.pi * (w[k] + (wp+4*np.pi)/3)/(wp+np.pi))
                        else:
                            tmp[k] = np.sqrt(2/(wp-wn)) * np.exp(1j * np.pi * (w[k] - (4*wp+np.pi)/3)/(wp+np.pi))
                mfb.append(tmp)
            else: # individual wavelets
                tmp = np.zeros(N).astype(complex)

                wn = (boundaries[l+1] + boundaries[l])/2
                if (l-1) <= -1:
                    wm = (boundaries[0] + boundaries[-1])/2
                else:
                    if (boundaries[l-1] > 0) and (boundaries[l] < 0):
                        wm = (boundaries[l] - np.pi)/2
                    else:
                        wm = (boundaries[l] + boundaries[l-1])/2

                if (l+2) >= Npic-1:
                    wp = (boundaries[l+1] + boundaries[0])/2
                else:
                    if (boundaries[l+1] > 0) and (boundaries[l+2] < 0):
                        wp = (boundaries[l+1] + np.pi)/2
                    else:
                        wp = (boundaries[l+1] + boundaries[l+2])/2
                
                for k in np.arange(N):
                    if (w[k] >= wm) and (w[k] <= wn):
                        if np.sign(w[k]) >= 0:
                            tmp[k] = np.sqrt(2/(wp-wm)) * np.exp(1j * np.pi * (w[k] + (wp-4*wm)/3)/(wp-wm)) * np.sin(np.pi * beta((w[k]-wm)/(wn-wm))/2)
                        else:
                            tmp[k] = np.sqrt(2/(wp-wm)) * np.exp(1j * np.pi * (w[k] - (4*wp-wm)/3)/(wp-wm)) * np.sin(np.pi * beta((w[k]-wm)/(wn-wm))/2)
                    elif (w[k] >= wn) and (w[k] <= wp):
                        if np.sign(w[k]) >= 0:
                            tmp[k] = np.sqrt(2/(wp-wm)) * np.exp(1j * np.pi * (w[k] + (wp-4*wm)/3)/(wp-wm)) * np.cos(np.pi * beta((w[k]-wn)/(wp-wn))/2)
                        else:
                            tmp[k] = np.sqrt(2/(wp-wm)) * np.exp(1j * np.pi * (w[k] - (4*wp-wm)/3)/(wp-wm)) * np.cos(np.pi * beta((w[k]-wn)/(wp-wn))/2)
                mfb.append(tmp)
        
    return mfb

# Gabor 1 wavelet
def gabor1_filterbank(boundaries, N, cpx):

    """ Gabor wavelet filters - Option 1

    Parameters
    ----------
    - boundaries: ndarray containing the support boundaries to be used.
        It is expected that these boundaries lie in [0,pi) for a real case,
        and [0,2pi) for a complex case.
    - N: number of samples per filter.
        Should correspond to the length of the Fourier transform of the signal 
        used to detect the boundaries
    - cpx: True if the original signal was complex, False otherwise.

    Returns
    -------
    - mfb: list
        Each element of that list is a ndarray corresponding to each filter

    Notes
    -----
    This function builds the set of Gabor filters corresponding to the given set
    of boundaries. This option build the last filter as a full Gabor filter.

    Author: Jerome Gilles
    Institution: San Diego State University
    Version: 1.0 (01/07/2025)
    """

    Npic = np.size(boundaries)

    # create the frequency axis [0,pi)u[-pi,0)
    Mi = int(np.floor(N/2)) + 1
    
    w = 2 * np.pi * np.arange(N) / N
    w[Mi:] = -2 * np.pi + w[Mi:]

    mfb = []
    kk = 0
    if cpx == False:
        # scaling function
        tmp = np.zeros(N)
        om = 2 * boundaries[0]

        for k in np.arange(N):
            tmp[k] = np.exp(-np.pi * (2.5 * np.abs(w[k])/om)**2)/np.sqrt(om)
        mfb.append(tmp)

        # each wavelet except the last one
        for l in np.arange(Npic-1):
            tmp = np.zeros(N)

            om = boundaries[l+1]-boundaries[l]
            wm = (boundaries[l+1] + boundaries[l])/2

            for k in np.arange(N):
                tmp[k] = np.exp(-np.pi * (2.5 * (np.abs(w[k])-wm)/om)**2)/np.sqrt(om)

            mfb.append(tmp)
        
        # last one
        tmp = np.zeros(N)

        om = np.pi - boundaries[-1]
        wm = (boundaries[-1] + np.pi)/2

        for k in np.arange(N):
            tmp[k] = np.exp(-np.pi * (2.5 * (np.abs(w[k])-wm)/om)**2)/np.sqrt(om)

        mfb.append(tmp)
    else:
        indgpi = np.argwhere(boundaries >= np.pi)
        boundaries[indgpi] = -2 * np.pi + boundaries[indgpi]

        # scaling function
        tmp = np.zeros(N)

        om = boundaries[0] - boundaries[-1]
        wm = (boundaries[0] + boundaries[-1])/2

        for k in np.arange(N):
            tmp[k] = np.exp(-np.pi * (2.5 * (w[k] - wm)/om)**2)/np.sqrt(om)
        mfb.append(tmp)

        # each wavelet (high frequencies in separate way)
        for l in np.arange(Npic-1):
            if boundaries[l] > boundaries[l+1]:
                #positive frequencies
                tmp = np.zeros(N)

                om = np.pi - boundaries[l]
                wm = (boundaries[l] + np.pi)/2

                for k in np.arange(N):
                    tmp[k] = np.exp(-np.pi * (2.5 * (w[k] - wm)/om)**2)/np.sqrt(om)

                mfb.append(tmp)

                # negative frequencies
                tmp = np.zeros(N)

                om = np.abs(-np.pi - boundaries[l+1])
                wm = (boundaries[l+1] - np.pi)/2

                for k in np.arange(N):
                    tmp[k] = np.exp(-np.pi * (2.5 * (w[k] - wm)/om)**2)/np.sqrt(om)

                mfb.append(tmp)
            else:  # individual wavelets
                tmp = np.zeros(N)

                om = boundaries[l+1] - boundaries[l]
                wm = (boundaries[l] + boundaries[l+1])/2

                for k in np.arange(N):
                    tmp[k] = np.exp(-np.pi * (2.5 * (w[k] - wm)/om)**2)/np.sqrt(om)

                mfb.append(tmp)

    return mfb

# Gabor 2 wavelet
def gabor2_filterbank(boundaries, N, cpx):

    """ Gabor wavelet filters - option 2

    Parameters
    ----------
    - boundaries: ndarray containing the support boundaries to be used.
        It is expected that these boundaries lie in [0,pi) for a real case,
        and [0,2pi) for a complex case.
    - N: number of samples per filter.
        Should correspond to the length of the Fourier transform of the signal 
        used to detect the boundaries
    - cpx: True if the original signal was complex, False otherwise.

    Returns
    -------
    - mfb: list
        Each element of that list is a ndarray corresponding to each filter

    Notes
    -----
    This function builds the set of Gabor filters corresponding to the given set
    of boundaries. This option build the last filter as half a Gabor filter and the
    other half as a constant.

    Author: Jerome Gilles
    Institution: San Diego State University
    Version: 1.0 (01/07/2025)
    """

    Npic = np.size(boundaries)

    # create the frequency axis [0,pi)u[-pi,0)
    Mi = int(np.floor(N/2)) + 1
    
    w = 2 * np.pi * np.arange(N) / N
    w[Mi:] = -2 * np.pi + w[Mi:]

    mfb = []
    kk = 0
    if cpx == False:
        # scaling function
        tmp = np.zeros(N)
        om = 2 * boundaries[0]

        for k in np.arange(N):
            tmp[k] = np.exp(-np.pi * (2.5 * np.abs(w[k])/om)**2)/np.sqrt(om)
        mfb.append(tmp)

        # each wavelet except the last one
        for l in np.arange(Npic-1):
            tmp = np.zeros(N)

            om = boundaries[l+1]-boundaries[l]
            wm = (boundaries[l+1] + boundaries[l])/2

            for k in np.arange(N):
                tmp[k] = np.exp(-np.pi * (2.5 * (np.abs(w[k])-wm)/om)**2)/np.sqrt(om)

            mfb.append(tmp)
        
        # last one
        tmp = np.zeros(N)

        om = np.pi - boundaries[-1]
        wm = (boundaries[-1] + np.pi)/2

        for k in np.arange(N):
            if np.abs(w[k]) < wm:
                tmp[k] = np.exp(-np.pi * (2.5 * (np.abs(w[k])-wm)/om)**2)/np.sqrt(om)
            else:
                tmp[k] = 1/np.sqrt(om)

        mfb.append(tmp)
        
    else:
        indgpi = np.argwhere(boundaries >= np.pi)
        boundaries[indgpi] = -2 * np.pi + boundaries[indgpi]

        # scaling function
        tmp = np.zeros(N)

        om = boundaries[0] - boundaries[-1]
        wm = (boundaries[0] + boundaries[-1])/2

        for k in np.arange(N):
            tmp[k] = np.exp(-np.pi * (2.5 * (w[k] - wm)/om)**2)/np.sqrt(om)
        mfb.append(tmp)

        # each wavelet (high frequencies in separate way)
        for l in np.arange(Npic-1):
            if boundaries[l] > boundaries[l+1]:
                #positive frequencies
                tmp = np.zeros(N)

                om = np.pi - boundaries[l]
                wm = (boundaries[l] + np.pi)/2

                for k in np.arange(N):
                    if w[k] < wm:
                        tmp[k] = np.exp(-np.pi * (2.5 * (w[k] - wm)/om)**2)/np.sqrt(om)
                    else:
                        tmp[k] = 1/np.sqrt(om)

                mfb.append(tmp)

                # negative frequencies
                tmp = np.zeros(N)

                om = np.abs(np.pi + boundaries[l+1])
                wm = (boundaries[l+1] - np.pi)/2

                for k in np.arange(N):
                    if wm <= w[k]:
                        tmp[k] = np.exp(-np.pi * (2.5 * (w[k] - wm)/om)**2)/np.sqrt(om)
                    else:
                        tmp[k] = 1/np.sqrt(om)

                mfb.append(tmp)
            else:  # individual wavelets
                tmp = np.zeros(N)

                om = boundaries[l+1] - boundaries[l]
                wm = (boundaries[l] + boundaries[l+1])/2

                for k in np.arange(N):
                    tmp[k] = np.exp(-np.pi * (2.5 * (w[k] - wm)/om)**2)/np.sqrt(om)

                mfb.append(tmp)

    return mfb


# ===========================================================================
#                            AUXILIARY FUNCTIONS
# ===========================================================================
def dual_filterbank(mfb):

    """ Build the dual filterbank

    Parameters
    ----------
    - mfb: list
        List containing the original empirical wavelet filters.

    Returns
    -------
    - dualmfb: list
        List which elements are the dual filters of each filter of the input 
        filterbank.

    Notes
    -----
    This function builds the dual filterbank used for the reconstruction.

    Author: Jerome Gilles
    Institution: San Diego State University
    Version: 1.0 (01/07/2025)
    """

    sum = np.square(np.abs(mfb[0]))
    for k in np.arange(1,np.shape(mfb)[0]):
        sum = sum + np.square(np.abs(mfb[k]))

    dualmfb = []
    for k in np.arange(np.shape(mfb)[0]):
        dualmfb.append(np.divide(mfb[k],sum))

    return dualmfb

def iewt1d(ewt, mfb, cpx):

    """ Inverse 1D Empirical Wavelet Transform

    Parameters
    ----------
    -ewt: List
        Each element in that list is a ndarray containing the empirical wavelet 
        components.
    -mfb: List
        Each element in that list is a ndarray containing the empirical wavelet
        filters used to extract ewt
    -cpx: True is the original signal was complex, False otherwise

    Returns
    -------
    - rec ndarray
        Reconstructed signal
    
    Notes
    -----
    This function performs the inverse empirical wavelet transform from 
    the empirical components stored in ewt.


    Author: Jerome Gilles
    Institution: San Diego State University
    Version: 1.0 (01/07/2025)
    """

    # compute the dual filter bank
    mfb = dual_filterbank(mfb)

    # perform the adjoint operator to reconstruct the signal
    rec = np.zeros(np.shape(ewt)[1])
    for k in np.arange(np.shape(ewt)[0]):
        if cpx == False:
            rec = rec + np.real(np.fft.ifft(np.multiply(np.fft.fft(ewt[k]),mfb[k])))
        else:
            rec = rec + np.fft.ifft(np.multiply(np.fft.fft(ewt[k]),mfb[k]))

    return rec


# ===========================================================================
#                          FUNCTIONS TO PLOT
# ===========================================================================
def plot_filterBank1d(mfb,spectrum=[],Grouped = False,SamplingRate=2*np.pi):

    """ Plot 1D empirical wavelet filterbank

    Parameters
    ----------
    - mfb: list 
        contains the 1D numpy arrays corresponding to each filter
    - spectrum: [] (default) or 1D array of the spectrum 
        we assume it has not been fftshift.
    - Grouped: False or True
        if False, each filter is plotted on a different subplot, otherwise 
        they are all plotted on the same figure.
    - SamplingRate: Sampling frequency
        if known we can indicate the used sampling frequency. If unknown, 
        set it to 2*np.pi (default) to have a normalized frequancy axis,
        i.e. [-pi,pi)

    Notes
    -----
    This function plots the filter bank on top of the spectrum (if provided).
    If the filters are complex, it plots the magnitude.


    Author: Jerome Gilles
    Institution: San Diego State University
    Version: 1.0 (01/07/2025)
    """

    Mi = int(np.floor((np.shape(mfb)[1]+1)/2))
    w = 2 * np.pi * np.arange(np.shape(mfb)[1]) / np.shape(mfb)[1]
    w[Mi:] = -2 * np.pi + w[Mi:]
    w = np.fft.fftshift(w)

    if SamplingRate != 2*np.pi:
         w = w * SamplingRate / (2*np.pi)

    if np.size(spectrum) != 0:
        spectrum = np.fft.fftshift(spectrum / np.max(spectrum))


    if Grouped:
        plt.figure(figsize=(12,8))
        if np.size(spectrum) != 0:
            plt.plot(w,spectrum,linewidth=1)
        for k in np.arange(np.shape(mfb)[0]):
            if np.isrealobj(mfb[k]):
                plt.plot(w,np.fft.fftshift(mfb[k]),'r',linewidth=1)
            else:
                plt.plot(w,np.fft.fftshift(np.abs(mfb[k])),'r',linewidth=1)
        plt.title("Filter Bank")
        plt.show()
    else:
        # compute the structure of the grid
        nc = int(np.ceil(np.sqrt(len(mfb))))
        nr = int(np.ceil(len(mfb)/nc))

        figmfb, axmfb = plt.subplots(nr,nc, figsize=(12,8))
        k = 0

        if nr == 1:
            for kc in np.arange(nc):
                if np.isrealobj(mfb[k]):
                    if np.size(spectrum) != 0:
                        print('boo!')
                        axmfb[kc].plot(w,spectrum,linewidth=1)
                    axmfb[kc].plot(w,np.fft.fftshift(mfb[k]),'r',linewidth=1)
                    axmfb[kc].set_title("Filter "+str(k))
                else:
                    if np.size(spectrum) != 0:
                        axmfb[kc].plot(w,spectrum,linewidth=1)
                    axmfb[kc].plot(w,np.fft.fftshift(np.abs(mfb[k])),'r',linewidth=1)
                    axmfb[kc].set_title("Filter "+str(k))
                k = k+1
                if k >= len(mfb):
                    break

        else:
            for kr in np.arange(nr):
                for kc in np.arange(nc):
                    if np.isrealobj(mfb[k]) == True:
                        if np.size(spectrum) != 0:
                            axmfb[kr,kc].plot(w,spectrum,linewidth=1)
                        axmfb[kr,kc].plot(w,np.fft.fftshift(mfb[k]),'r',linewidth=1)
                        axmfb[kr,kc].set_title("Filter "+str(k))
                    else:
                        if np.size(spectrum) != 0:
                            axmfb[kr,kc].plot(w,spectrum,linewidth=1)
                        axmfb[kr,kc].plot(w,np.fft.fftshift(np.abs(mfb[k])),'r',linewidth=1)
                        axmfb[kr,kc].set_title("Filter "+str(k))
                    k = k+1
                    if k >= len(mfb):
                        break
                if k >= len(mfb):
                    break

        # Remove the empty plots
        for ax in axmfb.flat[len(mfb):]:
            ax.remove()

        plt.tight_layout()
        plt.show()

    return

def plot_ewt1d(ewt,indep = False, SamplingRate=2*np.pi):

    """ Plot the empirical wavelet components

    Parameter
    ---------
    - ewt: list 
        contains the 1D numpy arrays corresponding to each emprical
        wavelet components
    - indep: True of False (default)
        if False the components are plotted in different subplots. If True, they 
        are plotted in different figures
    - SamplingRate: Sampling frequency
        if known we can indicate the used sampling frequency. If unknown, 
        set it to 2*np.pi (default) to have a normalized frequancy axis,
        i.e. [-pi,pi)

    Notes
    -----
    This function plots the empirical wavelet components extracted by the EWT.
    If the components are complex, it plot the real and imaginary parts separately.

    Author: Jerome Gilles
    Institution: San Diego State University
    Version: 1.0 (01/07/2025)
    """

    Newt = np.shape(ewt)[0]

    t = np.arange(np.shape(ewt)[1])
    if SamplingRate != (2*np.pi):
        t = t / SamplingRate

    if indep:
        for k in np.arange(Newt):
            if np.isrealobj(ewt[k]) == True:
                plt.plot(t,ewt[k])
                plt.title("EWT Component "+str(k))
                plt.tight_layout()
                plt.show()
            else:
                figewt, axewt = plt.subplots(1,2, figsize=(12,8))
                axewt[0].plot(t,np.real((ewt[k])))
                axewt[1].plot(t,np.imag((ewt[k])))
                axewt[0].set_title("EWT Component "+str(k)+" - Real part")
                axewt[1].set_title("EWT Component "+str(k)+" - Imaginary part")
                plt.tight_layout()
                plt.show()
    else:
        if np.isrealobj(ewt[1]) == True:
            figewt, axewt = plt.subplots(Newt,1, figsize=(12,8))
        else:
            figewt, axewt = plt.subplots(Newt,2, figsize=(12,8))

        for k in np.arange(Newt):
            if np.isrealobj(ewt[k]) == True:
                axewt[k].plot(t,ewt[k])
                axewt[k].set_title("EWT Component "+str(k))
            else:
                axewt[k][0].plot(t,np.real((ewt[k])),'r')
                axewt[k][1].plot(t,np.imag((ewt[k])),'c')
                axewt[k][0].set_title("EWT Component "+str(k)+" - Real part")
                axewt[k][1].set_title("EWT Component "+str(k)+" - Imaginary part")

        plt.tight_layout()
        plt.show()

    return


# ===========================================================================
#                          TIME-FREQUENCY TOOLS
# ===========================================================================
def timefrequency_plot(ewt, boundaries,sig,color=True,SamplingRate=2*np.pi,GlobalNorm=True,Logmag=False):

    """ Time-Frequency plot

    Parameters
    ----------
    - ewt: list 
        contains the 1D numpy arrays corresponding to each emprical
        wavelet components
    - boundaries: ndarray 
        contains the boundaries used in the construction of the empirical
        wavelet filters.
    - sig: ndarray
        original signal
    - color: True (default) or False
        indicate if the plot must be done in color (True) or grayscale (False)
    - SamplingRate: Sampling frequency
        if known we can indicate the used sampling frequency. If unknown, 
        set it to 2*np.pi (default) to have a normalized frequancy axis,
        i.e. [-pi,pi)
    - GlobalNorm: True (default) or False
        indicate if the magnitude of each time-frequency curve must be normalized globally 
        (i.e. by the maximum across all component) or individually (i.e by its own maximum).
        Note: if individual normalization tends to provide a more visually appealing plot,
        be aware the maxima for each curve (despite having the same color) are different!
    - Logmag: True or False (default)
        indicate if we plot the logarithm of the magnitude.

    Notes
    -----
    This function extract the time-frequency information from the empirical wavelet components
    and plot them in the time-frequency domain. This function works only for real signals.

    Author: Jerome Gilles
    Institution: San Diego State University
    Version: 1.0 (01/07/2025)
    """

    if np.iscomplexobj(ewt):
        print("The time-frequency information cannot be used on complex signals or filters")
        return

    HilbA, HilbF = instantaneous_components(ewt, boundaries)

    if Logmag:
        for k in np.arange(np.shape(HilbA)[0]):
            HilbA[k] = np.log(1+HilbA[k])

    maxv = np.ones(np.shape(HilbA)[0])
    for k in np.arange(np.shape(HilbA)[0]):
        maxv[k] = np.max(HilbA[k])

    if GlobalNorm:
        maxg = np.max(maxv)
        for k in np.arange(np.shape(HilbA)[0]):
            maxv[k] = maxg

    t = np.arange(np.shape(HilbA)[1])
    if SamplingRate != (2*np.pi):
        t = t / SamplingRate

    figtf, axtf = plt.subplots(2,1,figsize=(12,8), gridspec_kw={'height_ratios': [1, 5]})

    if color: # plot in color
        # plot input signal on top
        axtf[0].plot(t,sig[:np.size(t)])

        # plot each time-frequency curve
        for k in np.arange(np.shape(HilbA)[0]):
            HilbF[k] = HilbF[k] * SamplingRate / (2*np.pi)
            axtf[1].scatter(t,HilbF[k],c=cm.turbo(HilbA[k]/maxv[k]), s=0.1)

    else:   # plot in grayscale
        # plot input signal on top
        axtf[0].plot(t,sig[:np.size(t)],color='black')

        # plot each time-frequency curve
        for k in np.arange(np.shape(HilbA)[0]):
            HilbF[k] = HilbF[k] * SamplingRate / (2*np.pi)
            axtf[1].scatter(t,HilbF[k],c=cm.binary(HilbA[k]/maxv[k]), s=0.1)

    axtf[1].set_ylim([-SamplingRate*0.02, 1.05*SamplingRate/2])
    axtf[1].set_xlabel('time')
    axtf[1].set_ylabel('frequency')
    figtf.tight_layout()
    plt.show()

    return

def instantaneous_components(ewt, boundaries):

    """ Extract instantaneous amplitude and frequencies

    Parameters
    ----------
    - ewt: list containing the 1D numpy arrays corresponding to each emprical
        wavelet components
    - boundaries: ndarray containing the boundaries used in the construction of the empirical
        wavelet filters.

    Returns
    ------
    - HilbA: list that contains ndarray with the instantaneous amplitude
    - HilbF: list that contains ndarray with the instantaneous frequencies

    Notes
    -----
    This function extracts the instantaneous amplitude and frequencies for
    each empirical wavelet component using the Hilbert transform.
    
    Author: Jerome Gilles
    Institution: San Diego State University
    Version: 1.0 (01/07/2025)
    """

    gamma = np.pi

    boundaries = np.hstack((0,boundaries))
    boundaries = np.hstack((boundaries, np.pi))

    for k in np.arange(np.size(boundaries)-1):
        r = (boundaries[k+1] - boundaries[k])/(boundaries[k+1] + boundaries[k])
        if r < gamma:
            gamma = r

    # we compute the instantaneous amplitude and frequencies of each component
    HilbA = []
    HilbF = []

    for i in np.arange(np.shape(ewt)[0]):
        ht = hilbert(ewt[i]) # compute the analytic version of the component

        # extract instantaneous amplitude
        HilbA.append(np.abs(ht[:np.size(ht)-1]))

        # extract instantaneous frequencies via phase unwrapping, derivative and 
        # outlier cleaning
        instf = np.diff(np.unwrap(np.angle(ht)))
        HilbF.append(ifcleaning(instf,(1-gamma)*boundaries[i],(1+gamma)*boundaries[i+1]))

    return HilbA, HilbF

def ifcleaning(fin, lb, ub):

    """ Instantaneous frequencies cleaning

    Parameters
    ----------
    - fin: ndarray 
        contains the set of frequencies to clean
    - lb: lower bound defining the expected interval
    - ub: upper bound defining the expected interval

    Returns
    ------
    - f: ndarray 
        contains the cleaned frequencies
    
    Notes
    -----
    This function removes outliers (i.e. frequencies that are not in [lb,ub])
    using interpolation.
        
    Author: Jerome Gilles
    Institution: San Diego State University
    Version: 1.0 (01/07/2025)
    """

    # list all indices
    indi = np.arange(np.size(fin)) 
    # keep all indices inside [lb,ub]
    ind = np.intersect1d(np.argwhere(fin >= lb),np.argwhere(fin <= ub))

    if ind.size == np.size(fin):   # if all values are in [lb,ub], no cleaning needed
        f = fin
    else:
        # extract values corresponding to the "good" indices
        fe = fin[ind]

        # check if we removed the first point, if yes then we need an initial point
        # so, as we don't have any a priori, we fix this initial point to the first
        # kept value
        if ind[0] != 0:
            fe = np.hstack((fe[0],fe))
            ind = np.hstack((0,ind))

        # check if we removed the last point, if yes then we need a final point so,
        # as we don't have any a priori, we fix this final point to the last kept
        # value
        if ind[-1] != np.size(fin)-1:
            fe = np.hstack((fe,fe[-1]))
            ind = np.hstack((ind, np.size(fin)-1))

        # we interpolate at all points based on the knowledge of the "good" points
        # (for the lowest frequency band we use a linear interpolation in order to 
        # avoid possible osccillation which will return us back to negative frequencies,
        # otherwise we use spline interpolation
        if lb == 0:
            f = np.interp(indi,ind,fe)
        else:
            spl = CubicSpline(ind,fe)
            f = spl(indi)

    return f