from matplotlib import cm
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

from .usefullfunc import round_away
from .boundaries1d import boundaries_detect
from .ewt1d import lp_filterbank

def ewt2d_tensor(im, params):

    """ 2D Tensor Empirical Wavelet Transform

    Parameters
    ----------
    f: ndarray
        input image
    params: Dictionary
        must be set properly accordingly to the options described
        in the Notes section

    Returns
    -------
    - ewtC: list
        2-dimensional list containing each empirical wavelet coefficients
    - mfbR: list
        each element of that list is a nparray containing the different filters 
        to process the rows
    - mfbC: list
        each element of that list is a nparray containing the different filters 
        to process the columns
    - BR: ndarray
        list of boundaries normalized in [0,pi) for the rows
    - BC: ndarray
        list of boundaries normalized in [0,pi) for the columns

    Notes
    -----
    This function builds two sets of 1D empirical Littlewood-Paley wavelet 
    filters to respectively process the rows and columns provided in params. 
    It has three main steps: 1) detect the supports boundaries, 2) build the 
    wavelet filter bank, 3) perform the transform itself. The options for 
    steps 1) are described hereafter.

    STEP 1): boundary detection:
    If params["log"] is set to True, then the detection is performed
    on the logarithm of the spectrum of f.

    Two types of preprocessing are available to "clean" the spectrum:
    
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
    Version: 1.0 (01/21/2025)
    """

    # check if the input image is of proper complex type to
    # guarantee machine precision in the transform. If it is
    # not then we cast it to the appropriate type.
    if im.dtype != np.complex128:
        im = im.astype(np.complex128)

    # Compute the FFT of the image
    fftim = np.fft.fft2(im).astype(np.complex128)

    # ==================================================
    #                  ROW PROCESSING
    # ==================================================

    # Compute average of each 1D magnitude spectrum wrt rows
    meanfft = np.sum(np.abs(fftim),axis=0)/np.shape(fftim)[0]

    # Detect the boundaries
    boundaries, presig = boundaries_detect(np.abs(meanfft[0:int(round_away((np.size(meanfft))/2))]),params)
    if boundaries[0] == 0:
        boundaries = boundaries[1:]
    BR = boundaries * np.pi / round_away(np.size(meanfft)/2)

    # Build the corresponding filterbank
    mfbR = lp_filterbank(BR, np.size(meanfft), False)

    # We filter each row with each wavelet
    ewtR = []
    for k in np.arange(np.shape(mfbR)[0]): # go through the different filters
        tmp = np.zeros(np.shape(fftim)).astype(np.complex128)
        for r in np.arange(np.shape(fftim)[0]): # process each row
            tmp[r,:] = np.multiply(np.conjugate(mfbR[k]),fftim[r,:])

        ewtR.append(tmp)

    # ==================================================
    #                  COLUMN PROCESSING
    # ==================================================

    # Compute average of each 1D magnitude spectrum wrt columns
    meanfft = np.sum(np.abs(fftim),axis=1)/np.shape(fftim)[1]

    # Detect the boundaries
    boundaries, presig = boundaries_detect(np.abs(meanfft[0:int(round_away((np.size(meanfft))/2))]),params)

    if boundaries[0] == 0:
        boundaries = boundaries[1:]
    BC = boundaries * np.pi / round_away(np.size(meanfft)/2)

    # Build the corresponding filterbank
    mfbC = lp_filterbank(BC, np.size(meanfft), False)

    # We filter each row with each wavelet
    ewtC = [[0]*np.shape(mfbC)[0] for i in range(np.shape(mfbR)[0])]
    for k in np.arange(np.shape(mfbC)[0]): # go through the different filters
        for l in np.arange(np.shape(mfbR)[0]):
            tmp = np.zeros(np.shape(ewtR[l])).astype(np.complex128)
            for r in np.arange(np.shape(ewtR[l])[1]): # process each column
                tmp[:,r] = np.multiply(np.conjugate(mfbC[k]),ewtR[l][:,r])
            ewtC[l][k] = np.fft.ifft2(tmp)

    return ewtC, mfbR, mfbC, BR, BC

def iewt2d_tensor(ewt, mfbR, mfbC):

    """ Inverse 2D Tensor Empirical Wavelet Transform

    Parameters
    ----------
    - ewt: list
        2-dimensional list containing each empirical wavelet coefficients
    - mfbR: list
        each element of that list is a nparray containing the different filters 
        to process the rows
    - mfbC: list
        each element of that list is a nparray containing the different filters 
        to process the columns

    Returns
    -------
    - 2D nparray
        reconstructed image

    Notes
    -----
    This function performs the inverse tensor 2D empirical wavelet transform.
    The returned image is of dtype complex, thus if you are expecting a real 
    image, you need to take the real part of the returned image.
    
    Author: Jerome Gilles
    Institution: San Diego State University
    Version: 1.0 (01/21/2025)
    """
    
    # Inverse column filtering first
    ewtR = []
    for l in np.arange(np.shape(mfbR)[0]):  # go through the different filters
        tmp = np.zeros(np.shape(ewt[0][0])).astype(np.complex128)
        for k in np.arange(np.shape(mfbC)[0]):
            fftim = np.fft.fft2(ewt[l][k])
            for r in np.arange(np.shape(fftim)[1]): # process each column
                tmp[:,r] = tmp[:,r] + np.multiply(mfbC[k],fftim[:,r])
        ewtR.append(tmp)

    # Inverse row filtering next
    tmp = np.zeros(np.shape(ewt[0][0])).astype(np.complex128)

    for k in np.arange(np.shape(mfbR)[0]):
        fftim = ewtR[k]
        for r in np.arange(np.shape(tmp)[0]): # process each column
            tmp[r,:] = tmp[r,:] + np.multiply(mfbR[k],fftim[r,:])    

    return np.fft.ifft2(tmp).astype(np.complex128)

def plot_comp_tensor(ewt, energy=False, cpx=False):

    """ Plot empirical tensor wavelet coefficients

    Parameters:
    -----------
    - ewt: list
        2-dimensional list containing each empirical wavelet coefficients
    - energy: True or False (default)
        If True, the energy will be indicated above each image
    - cpx: True or False (default)
        Indicate if the expected EWT component should be considered real
        or complex (in the former, only the real parts will be plotted)

    Author: Jerome Gilles
    Institution: San Diego State University
    Version: 1.0 (01/21/2025)    
    """

    if cpx:
        # print real parts first
        figewtr, axewtr = plt.subplots(np.shape(ewt)[0],np.shape(ewt)[1], figsize=(12,8))

        for i in np.arange(np.shape(ewt)[0]):
            for j in np.arange(np.shape(ewt)[1]):
                axewtr[i][j].imshow(np.real(ewt[i][j]), cmap="gray", interpolation='none')
                axewtr[i][j].axis('off')
                if energy:
                    axewtr[i][j].set_title(str(np.round(LA.norm(ewt[i][j]),2)))
        figewtr.suptitle('Real parts')
        plt.tight_layout()
        plt.show()

        # print imaginary parts next
        figewti, axewti = plt.subplots(np.shape(ewt)[0],np.shape(ewt)[1], figsize=(12,8))

        for i in np.arange(np.shape(ewt)[0]):
            for j in np.arange(np.shape(ewt)[1]):
                axewti[i][j].imshow(np.imag(ewt[i][j]), cmap="gray", interpolation='none')
                axewti[i][j].axis('off')
                if energy:
                    axewti[i][j].set_title(str(np.round(LA.norm(ewt[i][j]),2)))
        figewti.suptitle('Imaginary parts')
        plt.tight_layout()
        plt.show()
    else:
        figewt, axewt = plt.subplots(np.shape(ewt)[0],np.shape(ewt)[1], figsize=(12,8))

        for i in np.arange(np.shape(ewt)[0]):
            for j in np.arange(np.shape(ewt)[1]):
                axewt[i][j].imshow(np.real(ewt[i][j]), cmap="gray", interpolation='none')
                axewt[i][j].axis('off')
                if energy:
                    axewt[i][j].set_title(str(np.round(LA.norm(ewt[i][j]),2)))

        plt.tight_layout()
        plt.show()

    return

def plot_tensor_boundaries(f,BR,BC,logtag=True,title="Tensor Fourier supports"):

    """ Plot the tensor empirical wavelet supports

    Parameters
    ----------
    - f: 2D nparray
        Input image
    - BR, BC: ndarray
        Arrays that contain the detected boundaries for rows and columns, respectively.
    - logtag: True (default) or False
        Indicate if the logarithm of the spectrum should be used for the background.
    - title: string
        Title to be plotted on the figure. The default title is "Tensor Fourier supports".

    Notes
    -----
    This function plots the position of the detected boundaries in the Fourier domain. Each
    rectangle (with its symmetric) defines an empirical wavelet support.
    
    Author: Jerome Gilles
    Institution: San Diego State University
    Version: 1.0 (01/21/2025)    
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
    BC = BC * round_away(np.shape(ft)[1]/2) / np.pi

    for k in np.arange(np.size(BC)):
        axb.axvline(x=(cc+BC[k]), color='r', linestyle='--')
        axb.axvline(x=(cc-BC[k]), color='r', linestyle='--')

    for k in np.arange(np.size(BR)):
        axb.axhline(y=(rc+BR[k]), color='r', linestyle='--')
        axb.axhline(y=(rc-BR[k]), color='r', linestyle='--')


    plt.axis('off')
    plt.title(title)
    plt.show()

    return