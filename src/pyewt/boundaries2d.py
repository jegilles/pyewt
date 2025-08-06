import numpy as np
import scipy as scipy
from scipy import signal
from scipy.special import ive, erf, erfinv
import matplotlib.pyplot as plt
from skimage import measure
import cc3d
from sklearn.cluster import KMeans
# from pyewt.src.usefullfunc import round_away
from .usefullfunc import round_away

def ewt2d_get_maxima(absff,params,extH,extW):

    """ Build a scale-space representation and post-process detected maxima.
    Discards maxima near the edge and, if needed, the center.

    Parameters
    ----------
    absff : 2D ndarray
        Input image (Fourier spectrum).
    params : dict
        Empirical wavelet parameters (see Notes below).
    extH : int
        1 if height was extended, 0 otherwise.
    extW : int
        1 if width was extended, 0 otherwise.

    Returns
    -------
    centers : ndarray
        Coordinates of significant maxima.
    plane : ndarray
        Scale-space plane.

    Notes
    -----
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

    Author: Jerome Gilles
    Institution: San Diego State University
    Version: 1.0 (07/15/2025)
    """
    hh, ww = absff.shape
    centers, _, plane = ewt2d_gss(absff, params)

    # Discard maxima near the edge or at the center if requested
    if params["edge"] > 0 or params["includeCenter"] == 1:
        center_flag = 1  # Assume center not present
        i = 0
        while i < centers.shape[0]:
            if (centers[i, 0] == round(hh / 2)) and (centers[i, 1] == round(ww / 2)):
                center_flag = 0  # Center found
            if (centers[i, 0] <= params["edge"]) or (centers[i, 1] <= params["edge"]):
                centers = np.delete(centers, i, axis=0)
            elif (centers[i, 0] >= hh + 1 - params["edge"]) or (centers[i, 1] >= ww + 1 - params["edge"]):
                centers = np.delete(centers, i, axis=0)
            else:
                i += 1
        if center_flag == 1:
            centers = np.vstack([centers, [round(hh / 2), round(ww / 2)]])

    # Remove maxima in the extended rows (if height was extended)
    if extH == 1:
        i = 0
        while i < centers.shape[0]:
            if centers[i, 0] == 0 or centers[i, 0] == hh-1:
                centers = np.delete(centers, i, axis=0)
            else:
                i += 1

    # Remove maxima in the extended columns (if width was extended)
    if extW == 1:
        i = 0
        while i < centers.shape[0]:
            if centers[i, 1] == 0 or centers[i, 1] == ww-1:
                centers = np.delete(centers, i, axis=0)
            else:
                i += 1

    centers = np.unique(centers, axis=0)

    return centers, plane

def ewt2d_gss(f,params):
    """ Build a 2D Gaussian scale space and detect local maxima.

    Parameters
    ----------
    f : 2D ndarray
        Input image (Fourier spectrum).
    params : dict
        Parameters for the scale space and maxima detection (see Notes below).

    Returns
    -------
    worm_bound : ndarray
        Coordinates of the detected maxima.
    Th : float
        Threshold for maxima detection.
    plane : ndarray
        Scale-space plane.

    Notes
    -----
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

    Author: Jerome Gilles
    Institution: San Diego State University
    Version: 1.0 (07/21/2025)
    """
    
    t = params["t"]
    kn = params["kn"]

    ker=np.exp(-t) * ive(np.linspace(-kn,kn,2*kn+1),t)   # 1D discrete Gaussian kernel via Bessel function
    kernel = np.outer(ker,ker) # form the 2D version
    kernel = kernel / np.sum(kernel)
    Niter = params["niter"] * np.ceil(np.min((np.shape(f)[0],np.shape(f)[1]))/kn).astype(int)  # number of iteration through the scales
    
    plane = np.zeros((np.shape(f)[0],np.shape(f)[1],Niter+1))
    plane[:,:,0] = ewt2d_localmax(f) # local maxima on the original spectrum

    Nlmax = np.zeros((Niter+1,1))
    Nlmax[0] = np.sum(plane[:,:,0]) # number of local maxima at the original scale

    # now we go through the scales and build the scale-space domain
    for i in range(1,Niter+1):
        f = ewt2d_fourier_extend(f,kn)
        f = signal.convolve2d(f,kernel,'same')
        f = ewt2d_fourier_shrink(f,kn)
        plane[:,:,i] = ewt2d_localmax(f)
        Nlmax[i] = np.sum(plane[:,:,i])

    # detect the worms as connected components
    plane_index = cc3d.connected_components(plane)

    if params["typeDetect"] == "kmeans":
        # number of worms
        n_worm = np.max(plane_index)
        worm_life = np.zeros(n_worm) # to store lifetime of each worm

        [X,Y,Z] = np.where(plane == 1) # extract all maxima
        for k in range(len(X)):
            worm_life[plane_index[X[k], Y[k], Z[k]] - 1] += 1  # count maxima for each worm

        Th = np.round(ewt2d_kmeansdetect(worm_life)).astype(int)
    elif params["typeDetect"] == "halfnormal":
        # number of worms
        n_worm = np.max(plane_index)
        worm_life = np.zeros(n_worm) # to store lifetime of each worm
    
        [X,Y,Z] = np.where(plane == 1) # extract all maxima
        for k in range(len(X)):
            worm_life[plane_index[X[k], Y[k], Z[k]] - 1] += 1  # count maxima for each worm

        Th = np.round(ewt2d_HalfNormalLaw(worm_life, np.shape(plane)[2])).astype(int)
    elif params["typeDetect"] == "empiricallaw":
        sum_worm = np.sum(plane, axis=(0, 1))
        worm_life = sum_worm[0:-1] - sum_worm[1:]
        Th = ewt2d_EmpiricalLaw(worm_life/np.sum(worm_life))
    else: # Otsu's method
        sum_worm = np.sum(plane, axis=(0, 1))
        worm_life = sum_worm[0:-1] - sum_worm[1:]
        Th = ewt2d_OtsuMethod(worm_life/np.sum(worm_life))

    # extract the label of the worms to keep
    arr = plane_index[:,:,Th]
    worm_labels = arr[arr != 0]

    # extract the root coordinates of the selected worms
    worm_bound = []
    ccx = np.floor(np.shape(f)[0]/2).astype(int) + 1
    ccy = np.floor(np.shape(f)[1]/2).astype(int) + 1
    for labels in worm_labels:
        root_coords = np.argwhere(plane_index[:,:,0] == labels)

        if root_coords.size == 0: # does not have a root -> don't keep it
            continue
        elif root_coords.shape[0] > 1: # has several roots -> keep the closest to the center
            cc = np.tile(np.array([ccx, ccy]),(root_coords.shape[0],1))
            dist = np.sum(np.square(root_coords - cc), axis=1)
            idxmin = np.argmin(dist)
            worm_bound.append(root_coords[idxmin])
        else:
            worm_bound.append(root_coords[0])

    worm_bound = np.array(worm_bound)

    return worm_bound, Th, plane

def ewt2d_OtsuMethod(histo):
    """ Classifies the set of minima curve lengths stored in histo into two classes using Otsu's method.
    Returns the boundaries which are supposed to be the meaningful ones.

    Parameters
    ----------
    histo : ndarray
        Set of minima curve lengths (1D array).

    Returns
    -------
    bounds : ndarray
        Detected bounds (subset of ind).
    th : int
        Detected scale threshold.
    
    Author: Jerome Gilles
    Institution: San Diego State University
    Version: 1.0 (07/05/2025)
    """

    Nt = np.sum(histo)
    histo = histo / Nt

    muT = np.sum((np.arange(1, len(histo) + 1)) * histo)

    sigbcv = np.zeros(len(histo) - 1)
    for k in range(len(histo) - 1):
        wb = np.sum(histo[:k+1])
        mu = np.sum((np.arange(1, k+2)) * histo[:k+1])
        wf = 1 - wb
        if wb == 0 or wf == 0:
            sigbcv[k] = 0
            continue
        mub = mu / wb
        muf = (muT - mu) / wf
        sigbcv[k] = wb * wf * (mub - muf) ** 2

    th = np.argmax(sigbcv)

    return th

def ewt2d_kmeansdetect(L):
    """ Detect meaningful minima using kmeans (for 1D L).

    Parameters
    ----------
    L : 1D ndarray
        Vector of the length of the minima curves.

    Returns
    -------
    th : float
        Detected scale threshold.

    Author: Jerome Gilles
    Institution: San Diego State University
    Version: 1.0 (07/15/2025)
    """
    # Reshape L for sklearn KMeans
    L_reshaped = L.reshape(-1, 1)
    km_model = KMeans(n_clusters=2, n_init=10, init='random', random_state=0)
    clusters = km_model.fit_predict(L_reshaped)

    i_max = np.argmax(L)
    nc = clusters[i_max]

    th = np.min(L[clusters == nc])

    return th

def ewt2d_HalfNormalLaw(L, Lmax):
    """ Classifies minima curve lengths into two classes using a half-normal law.

    Parameters
    ----------
    L : 1D ndarray
        Set of minima curve lengths.
    ind : 2D ndarray
        Original index of each minima (shape: [n_minima, 2]).
    Lmax : float
        Maximum possible length of a minima curve.

    Returns
    -------
    bounds : ndarray
        Detected bounds (subset of ind).
    th : float
        Detected scale threshold.

    Author: Jerome Gilles
    Institution: San Diego State University
    Version: 1.0 (07/15/2025)
    """
    # Estimate sigma
    sigma = np.sqrt(np.pi / 2) * np.mean(L)

    # Compute threshold
    th = np.sqrt(2) * sigma * erfinv(erf(Lmax / (np.sqrt(2) * sigma)) - 1 / len(L))

    # Keep only meaningful minima
    Lth = L.copy()
    Lth[Lth <= th] = 0

    return th

def ewt2d_EmpiricalLaw(histo):
    """ Classifies minima curve lengths into two classes using an empirical law.

    Parameters
    ----------
    L : 1D ndarray
        Set of minima curve lengths.
    ind : 2D ndarray
        Original index of each minima (shape: [n_minima, 2]).

    Returns
    -------
    bounds : ndarray
        Detected bounds (subset of ind).
    th : float
        Detected scale threshold.

    Author: Jerome Gilles
    Institution: San Diego State University
    Version: 1.0 (07/15/2025)
    """
    # Build normalized histogram
    chisto = np.cumsum(histo / np.sum(histo))

    # Find threshold
    idx = np.where(chisto > (1 - 1/len(histo)))[0]
    th = idx[0]

    return th

def ewt2d_plot_ssplane(plane):

    """ Plot 3D scale-space

    Parameters
    ----------
    - plane DOK sparse matrix
        contains the 3D scale-space domain

    Author: Jerome Gilles
    Institution: San Diego State University
    Version: 1.0 (02/11/2025)
    """

    [cx,cy,cz] = np.where(plane==1)

    fig, ax = plt.subplots(figsize=(12,12),subplot_kw={"projection": "3d"})
    ax.scatter(cx,cy,cz,marker='.')
    plt.show()

def ewt2d_fourier_extend(f,n):

    """ Extend image by periodization

    Parameters
    ----------
    - f : ndarray
        image to extend
    - n : extension size (in pixels)
    
    Returns
    -------
    - fe : ndarray
        extended image

    Notes
    -----
    This function extend the provided image f by periodization on a strip of width n around the image

    Author: Jerome Gilles
    Institution: San Diego State University
    Version: 1.0 (02/11/2025)    
    """

    fe = np.zeros((np.shape(f)[0]+2*n,np.shape(f)[1]+2*n))

    # copy f in the center of the extension
    fe[n:n+np.shape(f)[0],n:n+np.shape(f)[1]] = f

    # extend horizontally
    fe[0:n,:] = fe[np.shape(fe)[0]-2*n:np.shape(fe)[0]-n,:]
    fe[np.shape(fe)[0]-n:,:] = fe[n:2*n,:]

    # extend vertically
    fe[:,0:n] = fe[:,np.shape(fe)[1]-2*n:np.shape(fe)[1]-n]
    fe[:,np.shape(fe)[1]-n:] = fe[:,n:2*n]

    return fe

def ewt2d_fourier_shrink(f,n):

    """ shrink image

    Parameters
    ----------
    - f : ndarray
        image to extend
    - n : shrinking size (in pixels)
    
    Returns
    -------
    - fe : ndarray
        shrinked image

    Notes
    -----
    This function shrinks the provided image f by removing a strip of width n around the image

    Author: Jerome Gilles
    Institution: San Diego State University
    Version: 1.0 (02/11/2025)    
    """

    fe = f[n:np.shape(f)[0]-n,n:np.shape(f)[1]-n]

    return fe

def ewt2d_localmax(f):

    """ Find local maxmima on a 2D image

    Parameters
    ----------
    - f: ndarray
        image on which to perform the detection.

    Returns
    -------
    - tag: ndarray
        image with pixels either at 0 or 1 where 1 means a local maximum has been detected.

    Author: Jerome Gilles
    Institution: San Diego State University
    Version: 1.0 (02/11/2025)
    """

    tag = np.zeros(np.shape(f))

    # check center
    for i in range(1,np.shape(f)[0]-1):
        for j in range(1,np.shape(f)[1]-1):
            if (f[i,j] > f[i-1,j-1]) and (f[i,j] > f[i-1,j]) and (f[i,j] > f[i-1,j+1]) and (f[i,j] > f[i,j-1]) and (f[i,j] > f[i,j+1]) and (f[i,j] > f[i+1,j-1]) and (f[i,j] > f[i+1,j]) and (f[i,j] > f[i+1,j+1]):
                tag[i,j] = 1

    # check vertical edges
    for i in range(1,np.shape(f)[0]-1):
        if (f[i,0] > f[i-1,0]) and (f[i,0] > f[i-1,1]) and (f[i,0] > f[i,1]) and (f[i,0] > f[i+1,0]) and (f[i,0] > f[i+1,1]):
            tag[i,0] = 1

        if (f[i,-1] > f[i-1,-1]) and (f[i,-1] > f[i-1,-2]) and (f[i,-1] > f[i,-2]) and (f[i,-1] > f[i+1,-1]) and (f[i,-1] > f[i+1,-2]):
            tag[i,-1] = 1

    # check horizontal edges
    for j in range(1,np.shape(f)[1]-1):
        if (f[0,j] > f[0,j-1]) and (f[0,j] > f[1,j-1]) and (f[0,j] > f[1,j]) and (f[0,j] > f[0,j+1]) and (f[0,j] > f[1,j+1]):
            tag[0,j] = 1

        if (f[-1,j] > f[-1,j-1]) and (f[-1,j] > f[-2,j-1]) and (f[-1,j] > f[-2,j]) and (f[-1,j] > f[-1,j+1]) and (f[-1,j] > f[-2,j+1]):
            tag[-1,j] = 1

    # check four corners
    if (f[0,0] > f[0,1]) and (f[0,0] > f[1,0]) and (f[0,0] > f[1,1]):
        tag[0,0] = 1

    if (f[0,-1] > f[0,-2]) and (f[0,-1] > f[1,-1]) and (f[0,-1] > f[1,-2]):
        tag[0,-1] = 1

    if (f[-1,0] > f[-2,0]) and (f[-1,0] > f[-2,1]) and (f[-1,0] > f[-1,1]):
        tag[-1,0] = 1

    if (f[-1,-1] > f[-2,-1]) and (f[-1,-1] > f[-1,-2]) and (f[-1,-1] > f[-2,-2]):
        tag[-1,-1] = 1

    # detect the plateaux
    tagg = measure.label(tag)
    tag = np.zeros(np.shape(f))

    for i in range(1,np.max(tagg)+1):
        [pixx,pixy] = np.where(tagg == i)
        if np.size(pixx) > 1:
            centx = round_away(np.mean(pixx))
            centy = round_away(np.mean(pixy))
            tag[centx,centy] = 1
        else:
            tag[pixx,pixy] = 1

    return tag

def ewt2d_spectrum_regularize(f,params):

    """ 2D Spectrum regularization

    Parameters
    ----------
        f:  ndarray
            matrix containing the spectrum
        params: Dictionary
            must have params["reg"] to be set to one of the methods 
            listed below
            must have params["lengthfilter"] to be set for all methods
            must have params["sigmafilter"] to be set to the standard
            deviation for the Gaussian method

    Returns
    -------
        ff: ndarray
            matrix containing the regularized spectrum

    Notes
    -----
    This function regularizes f using the selected method set in params["reg"]. 
    The available methods are:
    - "none" : does nothing, returns f
    - "gaussian" : convolve f with a Gaussian of length and standard
        deviation given by params["lengthfilter"] and 
        params["sigmafilter"], respectively
    - "average": convolve f with a constant filter of length given 
        by params["lengthfilter"]

    Author: Jerome Gilles
    Institution: San Diego State University
    Version: 1.0 (02/09/2025)
    """

    if params["reg"] == "gaussian":
        gaussian1D = signal.windows.gaussian(params["lengthfilter"], params["sigmafilter"])
        kernel = np.outer(gaussian1D, gaussian1D)
        kernel /= (2*np.pi*(params["sigmafilter"]**2))

        ff = signal.convolve2d(f,kernel,mode='same')
    elif params["reg"] == "average":
        kernel = np.ones((params["lengthfilter"],params["lengthfilter"]))
        kernel = kernel / np.sum(kernel)
        ff = signal.convolve2d(f,kernel,mode='same')
    else:
        ff = f

    return ff

def symmetrize_fourier(f):

    """ Symmetric extension of the Fourier spectrum

    Parameters
    ----------
        f: ndarray 
            input 2D Fourier spectrum

    Returns
    -------
        sym: ndarray
            symmetrized Fourier spectrum
        extH: True/False
            indicate a vertical extension
        extW: True/False
            indicate an horizontal extension

    Notes
    -----
    This function returns a symetrized Fourier spectrum by properly extending
    the expected directions. It returns a new spectrum where each dimension is
    odd.

    Author: Jerome Gilles
    Institution: San Diego State University
    Version: 1.0 (02/09/2025)
    """

    if (np.remainder(np.shape(f)[0],2) == 0) and (np.remainder(np.shape(f)[1],2) != 0): # vertical extension
        sym = np.zeros((np.shape(f)[0]+1,np.shape(f)[1]))
        sym[0:-1,:] = f
        sym[-1,:] = np.flip(f[0,:])
        extH = True
        extW = False
    elif (np.remainder(np.shape(f)[0],2) != 0) and (np.remainder(np.shape(f)[1],2) == 0): # horizontal extension
        sym = np.zeros((np.shape(f)[0],np.shape(f)[1]+1))
        sym[:,0:-1] = f
        sym[:,-1] = np.flip(f[:,0])
        extH = False
        extW = True
    elif (np.remainder(np.shape(f)[0],2) == 0) and (np.remainder(np.shape(f)[1],2) == 0): # extension in both directions
        sym = np.zeros((np.shape(f)[0]+1,np.shape(f)[1]+1))
        sym[0:-1,0:-1] = f
        sym[-1,1:-1] = np.flip(f[0,1:])
        sym[1:-1,-1] = np.flip(f[1:,0])
        sym[0,0] = f[0,0]
        sym[0,-1] = f[0,0]
        sym[-1,0] = f[0,0]
        sym[-1,-1] = f[0,0]
        extH = True
        extW = True
    else: # no extension needed
        sym = f
        extH = False
        extW = False

    return sym, extH, extW

def UnSymmetrize_Fourier(in_array, extH, extW):
    """
    Removes the extension of the spectrum that was introduced to symmetrize the spectrum.

    Parameters
    ----------
    in_array : ndarray
        Magnitude of the spectrum to unsymmetrize.
    extH : bool or int
        1 (or True) if there is vertical extension, 0 otherwise.
    extW : bool or int
        1 (or True) if there is horizontal extension, 0 otherwise.

    Returns
    -------
    usym : ndarray
        Spectrum with original size.

    Author: Jerome Gilles
    Institution: San Diego State University
    Version: 1.0 (07/15/2025)
    """
    if extH and not extW:
        usym = in_array[:-1, :]
    elif not extH and extW:
        usym = in_array[:, :-1]
    elif extH and extW:
        usym = in_array[:-1, :-1]
    else:
        usym = in_array
    return usym

def plot_scalespace_maxima(f,maxima,logtag=True,title="Scale-space Maxima", radius=1):

    """ Plot the scale-space maxima on the Fourier spectrum

    Parameters
    ----------
    - f: 2D nparray
        Input image
    - maxima: ndarray
        Arrays that contain the coordinates of the detected maxima in the Fourier domain.
    - logtag: True (default) or False
        Indicate if the logarithm of the spectrum should be used for the background.
    - title: string
        Title to be plotted on the figure. The default title is "Scale-space Maxima".
    - radius: int
        Radius of the maxima to be plotted in pixels. The default value is 1 pixel.
        
    Notes
    -----
    This function plots the position of the detected maxima in the Fourier domain via the
    scale-space method. 
    
    Author: Jerome Gilles
    Institution: San Diego State University
    Version: 1.0 (07/15/2025)    
    """

    ft = np.fft.fftshift(np.abs(np.fft.fft2(f)))

    if logtag:
        ft = np.log(1+ft)
    
    
    plt.imshow(ft, cmap="gray", interpolation='none')
    plt.scatter(maxima[:,1], maxima[:,0], color='red', s=radius)

    plt.axis('off')
    plt.title(title)
    plt.show()

    return
