import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

from .pseudopolarfft import PPFFT
from .usefullfunc import round_away , beta
from .boundaries1d import boundaries_detect

def ewt2d_lp(im,params):

    """ 2D Littlewood-Paley Empirical Wavelet Transform

    Parameters
    ----------
    im: ndarray
        input image
    params: Dictionary
        must be set properly accordingly to the options described
        in the Notes section

    Returns
    -------
    - ewtlp: list
        2-dimensional list containing each empirical wavelet coefficients
    - mfb: list
        each element of that list is a ndarray containing the different filters 
    - BR: ndarray
        list of boundaries (radius) normalized in [0,pi)

    Notes
    -----
    This function builds the set of 2D empirical Littlewood-Paley wavelet 
    filters based on parameters provided in params. 
    It has three main steps: 1) detect the radius of each ring using the pseudo
    polar Fourier transform, 2) build the wavelet filter bank, 3) perform the 
    transform itself. The options for steps 1) are described hereafter.

    STEP 1): radius detection:
    If params["log"] is set to True, then the detection is performed
    on the logarithm of the spectrum of f.

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

    Author: Jerome Gilles
    Institution: San Diego State University
    Version: 1.0 (01/24/2025)
    """

    # check if the input image is of proper complex type to
    # guarantee machine precision in the transform. If it is
    # not then we cast it to the appropriate type.
    if im.dtype != np.complex128:
        im = im.astype(np.complex128)

    # We compute the pseudo polar Fourier transform
    pseudofft = PPFFT(im)

    meanfft = np.fft.fftshift(np.sum(np.abs(pseudofft),axis=1))

    # Detect the radius
    boundaries, presig = boundaries_detect(np.abs(meanfft[0:int(round_away((np.size(meanfft))/2))]),params)
    if boundaries[0] == 0:
        boundaries = boundaries[1:]
    BR = boundaries * np.pi / round_away(np.size(meanfft)/2)

    # Build the filterbank
    mfb = ewt2d_lp_filterbank(BR,np.shape(im)[1],np.shape(im)[0])

    # we filter to extract each subband
    ff = np.fft.fft2(im)

    ewtlp = []
    for k in np.arange(np.shape(mfb)[0]):
        tmp = np.fft.ifft2(np.multiply(np.conjugate(mfb[k]),ff))
        ewtlp.append(tmp)

    return ewtlp, mfb, BR

def ewt2d_lp_filterbank(boundaries,w,h):

    """ Build 2D Littlewood-Paley filterbank

    Parameters
    ----------
    - boundaries: ndarray
        contains the list of detected boundaries (radius)
    - w, h: width and height of the image, respectively.

    Returns
    -------
    - mfb: list
        each element of that list is a ndarray containing the different filters.

    Notes
    -----
    This function generate the 2D Littlewood-Paley filter bank (scaling 
    function + wavelets) corresponding to the provided set of frequency rings.

    Author: Jerome Gilles
    Institution: San Diego State University
    Version: 1.0 (01/24/2025)
    """

    Npic = np.size(boundaries)

    # we compute gamma accordingly to the theory
    gamma = 1
    for k in np.arange(Npic-1):
        r = (boundaries[k+1]-boundaries[k])/(boundaries[k+1]+boundaries[k])
        if r < gamma:
            gamma = r
    
    r = (np.pi-boundaries[-1])/(np.pi+boundaries[-1])
    if r < gamma:
        gamma = r
    gamma = (1-1/np.max([h,w])) * gamma

    mfb = []
    # start with the scaling function
    tmp = ewt2D_lp_scaling(boundaries[0],gamma,w,h)
    mfb.append(tmp)

    # generate the wavelets
    for k in np.arange(Npic-1):
        tmp = ewt2d_lp_wavelet(boundaries[k],boundaries[k+1],gamma,w,h)
        mfb.append(tmp)

    tmp = ewt2d_lp_up_wavelet(boundaries[-1],gamma,w,h)
    mfb.append(tmp)

    return mfb

def ewt2D_lp_scaling(w1, gamma, w, h):

    """ 2D Littlewood-Paley scaling function

    Parameters
    ----------
    - w1: radius
    - gamma: transition ratio
    - w: image width
    - h: image height

    Returns
    -------
    - yms: ndarray
        filter corresponding to the scaling function

    Notes
    -----
    Generate the 2D Littlewood-Paley scaling function in the 
    Fourier domain associated to the disk [0,w1] with 
    transition ratio gamma.
    
    Author: Jerome Gilles
    Institution: San Diego State University
    Version: 1.0 (01/24/2025)
    """

    an = 1/(2*gamma*w1)
    pbn = (1+gamma)*w1
    mbn = (1-gamma)*w1

    Mj = np.floor(h/2)
    Mi = np.floor(w/2)

    yms = np.zeros((h,w)).astype(np.complex128)
    for i in np.arange(w):
        for j in np.arange(h):
            k1 = np.pi * (i-Mi)/Mi
            k2 = np.pi * (j-Mj)/Mj

            w = np.sqrt(k1**2+k2**2)

            if (w < mbn):
                yms[j,i] = 1
            elif ((w >= mbn) and (w <= pbn)):
                yms[j,i] = np.cos(np.pi * beta(an * (w-mbn))/2)

    return np.fft.ifftshift(yms)

def ewt2d_lp_wavelet(wn,wm,gamma,w,h):

    """ 2D Littlewood-Paley wavelet

    Parameters
    ----------
    - wn: lower radius
    - wm: upper radius
    - gamma: transition ratio
    - w: image width
    - h: image height

    Returns
    -------
    - ymw: ndarray
        filter corresponding to the scaling function

    Notes
    -----
    Generate the 2D Littlewood-Paley wavelet in the Fourier domain 
    associated to the ring [wn,wm] with transition ratio gamma.
    
    Author: Jerome Gilles
    Institution: San Diego State University
    Version: 1.0 (01/24/2025)
    """

    an=1/(2*gamma*wn)
    am=1/(2*gamma*wm)
    pbn=(1+gamma)*wn
    mbn=(1-gamma)*wn
    pbm=(1+gamma)*wm
    mbm=(1-gamma)*wm

    Mj = np.floor(h/2)
    Mi = np.floor(w/2)

    ymw = np.zeros((h,w)).astype(np.complex128)
    for i in np.arange(w):
        for j in np.arange(h):
            k1 = np.pi * (i-Mi)/Mi
            k2 = np.pi * (j-Mj)/Mj

            w = np.sqrt(k1**2+k2**2)

            if (w >= pbn) and (w <= mbm):
                ymw[j,i] = 1
            elif (w > mbm) and (w <= pbm):
                ymw[j,i] = np.cos(np.pi * beta(am * (w-mbm))/2)
            elif (w >= mbn) and (w < pbn):
                ymw[j,i] = np.sin(np.pi * beta(an * (w-mbn))/2)

    return np.fft.ifftshift(ymw)

def ewt2d_lp_up_wavelet(wn,gamma,w,h):

    """ 2D Littlewood-Paley wavelet (highest frequencies)

    Parameters
    ----------
    - wn: lower radius
    - gamma: transition ratio
    - w: image width
    - h: image height

    Returns
    -------
    - ymw: ndarray
        filter corresponding to the scaling function

    Notes
    -----
    Generate the 2D Littlewood-Paley wavelet in the Fourier domain 
    associated to the region above the radius wn with transition 
    ratio gamma.
    
    Author: Jerome Gilles
    Institution: San Diego State University
    Version: 1.0 (01/24/2025)
    """


    an=1/(2*gamma*wn)
    pbn=(1+gamma)*wn
    mbn=(1-gamma)*wn

    Mj = np.floor(h/2)
    Mi = np.floor(w/2)

    ymw = np.ones((h,w)).astype(np.complex128)
    for i in np.arange(w):
        for j in np.arange(h):
            k1 = np.pi * (i-Mi)/Mi
            k2 = np.pi * (j-Mj)/Mj

            w = np.sqrt(k1**2+k2**2)

            if (w < mbn):
                ymw[j,i] = 0
            elif (w >= mbn) and (w < pbn):
                ymw[j,i] = np.sin(np.pi * beta(an * (w-mbn))/2)

    return np.fft.ifftshift(ymw)

def iewt2d_lp(ewt,mfb):

    """ Inverse 2D Littlewood-Paley Empirical Wavelet Transform

    Parameters
    ----------
    - ewt: list
        2-dimensional list containing each empirical wavelet coefficients.
    - mfb: list
        list containing the filter bank used to obtain ewt.

    Returns
    -------
    - Reconstructed image

    Notes
    -----
    This function performs the inverse 2D Littlewood-Paley EWT. Note
    that returned image is of complex type (you need to take its real 
    part if you expect a real image)
    
    Author: Jerome Gilles
    Institution: San Diego State University
    Version: 1.0 (01/24/2025)
    """

    tmp = np.zeros(np.shape(mfb[0])).astype(np.complex128)

    for k in np.arange(np.shape(mfb)[0]):
        tmp = tmp + np.multiply(np.fft.fft2(ewt[k]),mfb[k])

    return np.fft.ifft2(tmp)

def plot_lp_comp(ewt, energy=False):

    """ Plot 2D empirical Littlewood-Paley wavelet coefficients

    Parameters:
    -----------
    - ewt: list
        2-dimensional list containing each empirical wavelet coefficients
    - energy: True or False (default)
        If True, the energy will be indicated above each image

    Author: Jerome Gilles
    Institution: San Diego State University
    Version: 1.0 (01/23/2025)    
    """
    # compute the structure of the grid
    nc = int(np.ceil(np.sqrt(len(ewt))))
    if nc > 4:
        nc = 4
    nr = int(np.ceil(len(ewt)/nc))
    if nr > 4:
        nr = 4

    nfig = np.ceil(len(ewt) / (nr * nc))

    fign = 0
    for figi in range(int(nfig)):
        # print real parts first
        figewt, axewt = plt.subplots(nr,nc, figsize=(12,8))
        k = figi * nr * nc
        fign += 1

        if nr == 1:
            for kc in np.arange(nc):
                axewt[kc].imshow(np.real(ewt[k]),cmap='gray',interpolation='none')
                if energy:
                    axewt[kc].set_title(str(np.round(LA.norm(ewt[k]),2)))
                else:
                    axewt[kc].set_title("Component "+str(k))
                axewt[kc].axis('off')

                k = k+1
                if k >= len(ewt):
                    for ax in axewt.flat[len(ewt):]:
                        ax.remove()
                    break
        else:
            for kr in np.arange(nr):
                for kc in np.arange(nc):
                    if k < len(ewt):
                        axewt[kr,kc].imshow(np.real(ewt[k]),cmap='gray',interpolation='none')
                        if energy:
                            axewt[kr,kc].set_title(str(np.round(LA.norm(ewt[k]),2)))
                        else:
                            axewt[kr,kc].set_title("Component "+str(k))
                        axewt[kr,kc].axis('off')

                        k = k+1
                        if k >= len(ewt):
                            for ax in axewt.flat[len(ewt)-figi*nc*nr:]:
                                ax.remove()
                            break

        plt.tight_layout()
        plt.show()

    return

def plot_lp_filterbank(mfb):

    """ Plot 2D empirical Littlewood-Paley filters

    Parameters:
    -----------
    - mfb: list
        list containing each Littlewood-Paley filter

    Author: Jerome Gilles
    Institution: San Diego State University
    Version: 1.0 (01/24/2025)    
    """

    # compute the structure of the grid
    nc = int(np.ceil(np.sqrt(len(mfb))))
    nr = int(np.ceil(len(mfb)/nc))


    # print real parts first
    figmfb, axmfb = plt.subplots(nr,nc, figsize=(12,8))
    k = 0

    if nr == 1:
        for kc in np.arange(nc):
            axmfb[kc].imshow(np.fft.fftshift(np.real(mfb[k])),cmap='gray',interpolation='none')
            axmfb[kc].set_title("Filter "+str(k))
            axmfb[kc].axis('off')
            k = k+1
            if k >= len(mfb):
                break
    else:
        for kr in np.arange(nr):
            for kc in np.arange(nc):
                axmfb[kr,kc].imshow(np.fft.fftshift(np.real(mfb[k])),cmap='gray',interpolation='none')
                axmfb[kr,kc].set_title("Filter "+str(k))
                axmfb[kc,kr].axis('off')
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

def plot_lp_rings(f,BR,logtag=True,title="Littlewood-Paley Fourier supports"):

    """ Plot the rings delineating the Littlewood-Paley supports

    Parameters
    ----------
    - f: 2D nparray
        Input image
    - BR: ndarray
        Array that contain the detected radius.
    - logtag: True (default) or False
        Indicate if the logarithm of the spectrum should be used for the background.
    - title: string
        Title to be plotted on the figure. The default title is "Littlewood-Paley Fourier supports".

    Notes
    -----
    This function plots the position of the detected rings in the Fourier domain. Each
    ring defines an empirical wavelet support.
    
    Author: Jerome Gilles
    Institution: San Diego State University
    Version: 1.0 (01/24/2025)    
    """

    ft = np.fft.fftshift(np.abs(np.fft.fft2(f)))

    if logtag:
        ft = np.log(1+ft)
    
    figb, axb = plt.subplots(figsize=(10,7))
    axb.imshow(ft, cmap="gray", interpolation='none')

    # find center coordinates
    rc = int(round_away(np.shape(ft)[0]/2))
    cc = int(round_away(np.shape(ft)[1]/2))

    #  scale boundaries to the index space
    BR = BR * round_away(np.shape(ft)[0]/2) / np.pi

    for k in np.arange(np.size(BR)):
        axb.add_patch(plt.Circle((cc,rc),BR[k],color='red',fill=False))

    plt.axis('off')
    plt.title(title)
    plt.show()

    return