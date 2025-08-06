import numpy as np
import scipy as scipy
import matplotlib.pyplot as plt
from skimage.segmentation import watershed

# from pyewt.src.boundaries2d import ewt2d_spectrum_regularize, symmetrize_fourier, UnSymmetrize_Fourier, ewt2d_get_maxima
# from pyewt.src.ewt2dvoronoi import ewt2d_Voronoi_Filterbank, ewt2d_merge_symmetric 
from .boundaries2d import ewt2d_spectrum_regularize, symmetrize_fourier, UnSymmetrize_Fourier, ewt2d_get_maxima
from .ewt2dvoronoi import ewt2d_Voronoi_Filterbank, ewt2d_merge_symmetric 


def ewt2d_watershed(f,params):

    """ Compute the 2D EWT based on a Watershed partitioning of the Fourier domain.

    Parameters
    ----------
    f : 2D ndarray
        Input image.
    params : dict
        Transform parameters.

    Returns
    -------
    ewtw : list of 2D ndarray
        Collection of outputs of each EW filter.
    mfb : list of 2D ndarray
        The built filter bank.
    maxima : ndarray
        Coordinates of each meaningful detected maxima.
    waterpartition : 2D ndarray
        Detected Voronoi partition.
    plane : ndarray
        Scale-space domain.

    Notes
    -----
    If params["log"] is set to True, then the detection is performed on the logarithm 
    of the spectrum of f.
    params["typeDetect"] must be set to the wanted classification
        method. The available options are:
        - "otsu" : uses Otsu's technique
        - "halfnormal" : uses a half-normal law to model the problem
        - "empiricallaw" : uses the data itself to build a model of
        the problem
        - "kmeans" : uses kmeans to classify
    params["t"]: is the initial scale for the Gaussian kernel (0.8 is a good default value).
    params["kn"]: is the kernel size for the Gaussian kernel (6 is a good default value).
    params["niter"]: is the number of iterations through the scales (4 is a good default value).
    params["edge"]: is the size (in pixels) of the strip to ignore at the edge of the image (0 is no strip).
    params["includeCenter"]: if 1, the center of the image is included in the scale space maxima (0 is not included).
    params["tau"]: is the half width of the transition area for the Voronoi partition (0.1 is a good default value).
    params["complex"]: if 1, the watershed partition is complex, otherwise it is real (0 is real).
    
    Though usually not providing better results, this function allows regularization of the spectrum of f using the 
    selected method set in params["reg"]. The available methods are:
    - "none" : does nothing, returns f
    - "gaussian" : convolve f with a Gaussian of length and standard
        deviation given by params["lengthfilter"] and 
        params["sigmafilter"], respectively
    - "average": convolve f with a constant filter of length given 
        by params["lengthfilter"]
    
    Author: Jerome Gilles
    Institution: San Diego State University
    Version: 1.0 (07/28/2025)
    """

    if f.dtype != np.complex128:
        f = f.astype(np.complex128)

    # Get Fourier transform and its magnitude
    ff_orig = np.fft.fftshift(np.fft.fft2(f))

    # Symmetrize the spectrum if needed
    absff, extH, extW = symmetrize_fourier(np.abs(ff_orig))

    if params.get("log", 0) == 1:
        absff = np.log(absff)

    # Regularization (if requested)
    absff = ewt2d_spectrum_regularize(absff, params)

    # Get meaningful maxima
    maxima, plane = ewt2d_get_maxima(absff, params, extH, extW)

    # create the image of markers for watershed
    markers = np.zeros_like(absff, dtype=np.int32)
    for i, (y, x) in enumerate(maxima):
        markers[y, x] = i + 1

    # Build the Watershed cells
    labels = watershed(-absff, markers)

    waterpartition = UnSymmetrize_Fourier(labels, extH, extW)
    # Extract each Watershed cell mask
    # Make sure the labels are continuous
    lab = np.unique(labels).tolist()
    watercel = []
    for k in lab:
        mask = np.zeros(absff.shape, dtype=int)
        mask[labels == k] = 1
        watercel.append(mask)

    # Group symmetric cells if real transform is required
    if params.get("complex", 0) != 1:
        waterocells = ewt2d_merge_symmetric(watercel,maxima)
        mfb = ewt2d_Voronoi_Filterbank(absff.shape, waterocells, params["tau"], extH, extW)
    else:
        mfb = ewt2d_Voronoi_Filterbank(absff.shape, watercel, params["tau"], extH, extW)

    # Perform the filtering
    ewtw = []
    for filt in mfb:
        filtered = np.fft.ifft2(np.fft.ifftshift(ff_orig * filt))
        ewtw.append(filtered)

    return ewtw, mfb, maxima, waterpartition, plane

def iewt2d_watershed(ewtw, mfb):
    """
    Performs the inverse Empirical Watershed Wavelet Transform,
    returning a reconstruction of the image.

    Parameters
    ----------
    ewtw : list of 2D ndarray
        Empirical wavelet coefficients.
    mfb : list of 2D ndarray
        Corresponding empirical wavelet filters.

    Returns
    -------
    rec : 2D ndarray
        Reconstructed image.

    Author: Jerome Gilles
    Institution: San Diego State University
    Version: 1.0 (07/28/2025)
    """
    dual_sum = mfb[0] ** 2
    rec = np.fft.fftshift(np.fft.fft2(ewtw[0])) * mfb[0]
    for i in range(1, len(mfb)):
        rec += np.fft.fftshift(np.fft.fft2(ewtw[i])) * mfb[i]
        dual_sum += mfb[i] ** 2

    rec = np.fft.ifft2(np.fft.ifftshift(rec / dual_sum))

    return rec

def show_ewt2d_watershed_boundaries(f, wat, color=None, logspec=0):

    """ Plots the edges of the watershed partition onto the magnitude of the Fourier spectrum of the input image.

    Parameters:
        f: ndarray
            Input image.
        wat: ndarray 
            Watershed partition image (same size as f).
        color: list or tuple 
            RGB color for the partition edges, values in [0,1]. Default is red.
        logspec: int
            If 1, plot the logarithm of the spectrum. Default is 0.

    Author: Jerome Gilles
    Institution: San Diego State University
    Version: 1.0 (07/28/2025)
    """
    if color is None:
        color = [1, 0, 0]
    
    # Find the edges of the Watershed partition
    gr = np.zeros_like(f)
    gr[:-1, :] += np.abs(np.diff(wat, axis=0))
    gr[:, :-1] += np.abs(np.diff(wat, axis=1))
    
    # Get the magnitude of the Fourier spectrum of f
    absff = np.abs(np.fft.fftshift(np.fft.fft2(f)))
    if logspec == 1:
        absff = np.log(1 + absff)
    
    # Normalize between [0,1]
    absff = (absff - absff.min()) / (absff.max() - absff.min())
    
    # Make it an RGB image
    rgb_img = np.stack([absff]*3, axis=-1)
    
    # Tag each edge pixel to the wanted color
    edge_pixels = gr != 0
    for k in range(3):
        rgb_img[..., k][edge_pixels] = color[k]
    
    plt.figure()
    plt.imshow(rgb_img)
    plt.axis('off')
    plt.show()