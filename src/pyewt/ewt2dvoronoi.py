import numpy as np
import scipy as scipy
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt

# from pyewt.src.usefullfunc import beta
# from pyewt.src.boundaries2d import ewt2d_spectrum_regularize, symmetrize_fourier, UnSymmetrize_Fourier, ewt2d_get_maxima

from .usefullfunc import beta
from .boundaries2d import ewt2d_spectrum_regularize, symmetrize_fourier, UnSymmetrize_Fourier, ewt2d_get_maxima


def ewt2d_voronoi(f,params):

    """ Compute the 2D EWT based on a Voronoi partitioning of the Fourier domain.

    Parameters
    ----------
    f : 2D ndarray
        Input image.
    params : dict
        Transform parameters.

    Returns
    -------
    ewtc : list of 2D ndarray
        Collection of outputs of each EW filter.
    mfb : list of 2D ndarray
        The built filter bank.
    maxima : ndarray
        Coordinates of each meaningful detected maxima.
    vorpartition : 2D ndarray
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
    params["complex"]: if 1, the Voronoi partition is complex, otherwise it is real (0 is real).
    
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
    Version: 1.0 (07/21/2025)
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
    # maxima = np.unique(maxima, axis=0)

    # Build the Voronoi cells
    vorpartition, vorcel = ewt2d_Voronoi_Partition(maxima, absff.shape)
    vorpartition = UnSymmetrize_Fourier(vorpartition, extH, extW)

    # Group symmetric cells if real transform is required
    if params.get("complex", 0) != 1:
        vorocells = ewt2d_merge_symmetric(vorcel,maxima)
        mfb = ewt2d_Voronoi_Filterbank(absff.shape, vorocells, params["tau"], extH, extW)
    else:
        mfb = ewt2d_Voronoi_Filterbank(absff.shape, vorcel, params["tau"], extH, extW)

    # Perform the filtering
    ewtc = []
    for filt in mfb:
        filtered = np.fft.ifft2(np.fft.ifftshift(ff_orig * filt))
        ewtc.append(filtered)

    return ewtc, mfb, maxima, vorpartition, plane

def iewt2d_voronoi(ewtc, mfb):
    """
    Performs the inverse Empirical Voronoi Wavelet Transform,
    returning a reconstruction of the image.

    Parameters
    ----------
    ewtc : list of 2D ndarray
        Empirical wavelet coefficients.
    mfb : list of 2D ndarray
        Corresponding empirical wavelet filters.

    Returns
    -------
    rec : 2D ndarray
        Reconstructed image.

    Author: Jerome Gilles
    Institution: San Diego State University
    Version: 1.0 (07/16/2025)
    """
    dual_sum = mfb[0] ** 2
    rec = np.fft.fftshift(np.fft.fft2(ewtc[0])) * mfb[0]
    for i in range(1, len(mfb)):
        rec += np.fft.fftshift(np.fft.fft2(ewtc[i])) * mfb[i]
        dual_sum += mfb[i] ** 2

    plt.imshow(dual_sum, cmap='gray')
    plt.title('Dual sum of filters')
    plt.axis('off')
    plt.show()

    rec = np.fft.ifft2(np.fft.ifftshift(rec / dual_sum))

    return rec

def ewt2d_Voronoi_Filterbank(sizeImg, vorocells, tau, extH, extW):
    """
    Builds the filter bank based on the detected Voronoi cells.

    Parameters
    ----------
    sizeImg : tuple
        Size of the image (height, width).
    vorocells : list of ndarray
        List of Voronoi cell masks (binary arrays).
    tau : float
        Transition width.
    extH : bool or int
        1 (or True) if there is vertical extension, 0 otherwise.
    extW : bool or int
        1 (or True) if there is horizontal extension, 0 otherwise.

    Returns
    -------
    mfb : list of ndarray
        List of filters (same size as sizeImg).

    Author: Jerome Gilles
    Institution: San Diego State University
    Version: 1.0 (07/16/2025)
    """
    dist = []
    for cell in vorocells:
        # bwdist for foreground and background
        d1 = distance_transform_edt(1 - cell)
        d2 = distance_transform_edt(cell)
        normval = np.linalg.norm(cell.shape, 2)
        dist.append(2 * np.pi * (-d1 + d2) / normval)

    mfb = []
    for l in range(len(dist)):
        filt = np.zeros(sizeImg)
        for i in range(sizeImg[0]):
            for j in range(sizeImg[1]):
                filt[i, j] = ewt2d_Voronoi_LP_function(dist[l][i, j], tau)
        # manage the expected symmetry if needed
        if extH == 1 or extH is True:
            filt[0, :] = 0.5 * (filt[0, :] + filt[0, ::-1])
        if extW == 1 or extW is True:
            filt[:, 0] = 0.5 * (filt[:, 0] + filt[::-1, 0])
        # remove the extensions
        filt = UnSymmetrize_Fourier(filt, extH, extW)
        mfb.append(filt)

    return mfb

def ewt2d_Voronoi_LP_function(x, tau):
    """
    Calculate the Littlewood-Paley value based on the provided distance.

    Parameters
    ----------
    x : float
        Distance from the edge.
    tau : float
        Transition width.

    Returns
    -------
    y : float
        Value of the filter.

    Author: Jerome Gilles
    Institution: San Diego State University
    Version: 1.0 (07/16/2025)
    """
    if x > tau:
        y = 1.0
    elif x < -tau:
        y = 0.0
    else:
        y = np.cos(np.pi * beta((tau - x) / (2 * tau)) / 2)
    return y

def ewt2d_Voronoi_Partition(centroids, sizeImg):
    """
    Return the Voronoi partition corresponding to the provided centroids.
    The cells are properly assigned in order to guarantee a symmetric partition.

    Parameters
    ----------
    centroids : ndarray
        Array of shape (n_centroids, 2) containing the coordinates of each centroid.
    sizeImg : tuple
        Size of the image (height, width).

    Returns
    -------
    labelImage : ndarray
        Image containing the Voronoi partition tagged from 1 to the number of cells.
    voronoi_cells : list of ndarray
        List of binary images containing the mask of each Voronoi cell.

    Author: Jerome Gilles
    Institution: San Diego State University
    Version: 1.0 (07/15/2025)
    """
    labelImage = np.zeros(sizeImg, dtype=int)

    # Find closest centroid for each voxel
    for i in range(sizeImg[0]):
        for j in range(sizeImg[1]):
            eucDistance = np.sum((centroids - np.array([i, j]))**2, axis=1)
            labelImage[i, j] = np.argmin(eucDistance) + 1

    # Make sure the labels are continuous
    lab = np.unique(labelImage).tolist()
    
    # Extract each Voronoi cell mask
    voronoi_cells = []
    for k in lab:
        mask = np.zeros(sizeImg, dtype=int)
        mask[labelImage == k] = 1
        voronoi_cells.append(mask)
    return labelImage, voronoi_cells

def ewt2d_merge_symmetric(vor_cells, centers):

    """ Merge symmetric Voronoi cells based on their centers.
    Parameters
    ----------
    vor_cells : list of ndarray
        List of Voronoi cell masks (binary arrays).
    centers : ndarray
        Array of shape (n_centers, 2) containing the coordinates of each center.

    Returns
    -------
    symvorcells : list of ndarray
        List of merged Voronoi cell masks.

    """
        
    # find center of the image
    i0 = int(np.floor(vor_cells[0].shape[0] / 2))
    j0 = int(np.floor(vor_cells[0].shape[1] / 2))

    n_cells = len(vor_cells)
    if n_cells % 2 == 0: # there's no cell centered at the origin
        symvorcells = [None] * (int(np.floor(n_cells / 2)))
        start = 0
    else: # there's a cell centered at the origin
        symvorcells = [None] * (int(np.floor(n_cells / 2)) + 1)
        start = 1
        # Find the cell centered at origin
        ind = None
        for n, cell in enumerate(vor_cells):
            if cell[i0, j0] == 1:  
                symvorcells[0] = cell.copy()
                ind = n
                del vor_cells[ind]
                centers = np.delete(centers, ind, axis=0)

                break

    # Find the cells corresponding to the centers
    indexcells = np.zeros(len(centers))
    for k in range(len(centers)):
        for n, cell in enumerate(vor_cells):
            if cell[centers[k, 0], centers[k, 1]] == 1:
                indexcells[k] = n
                break

    # find the symmetric centers
    symcenters = np.shape(vor_cells[0]) - np.array([1 , 1]) - centers

    # find the pairs
    tag = np.zeros(len(centers), dtype=int)
    for cur in range(len(centers)):
        tag[cur] = 1
        for k in range(len(symcenters)):
            if tag[k] == 1:
                continue
            if np.array_equal(symcenters[k], centers[cur]):
                symvorcells[start + cur] = vor_cells[int(indexcells[cur])].copy() + vor_cells[int(indexcells[k])].copy()
                tag[k] = 1
                break

    return symvorcells

def show_ewt2d_voronoi_boundaries(f, vor, color=None, logspec=0):

    """ Plots the edges of the Voronoi partition onto the magnitude of the Fourier spectrum of the input image.

    Parameters:
        f: ndarray
            Input image.
        vor: ndarray 
            Voronoi partition image (same size as f).
        color: list or tuple 
            RGB color for the partition edges, values in [0,1]. Default is red.
        logspec: int
            If 1, plot the logarithm of the spectrum. Default is 0.

    Author: Jerome Gilles
    Institution: San Diego State University
    Version: 1.0 (07/16/2025)
    """
    if color is None:
        color = [1, 0, 0]
    
    # Find the edges of the Voronoi partition
    gr = np.zeros_like(f)
    gr[:-1, :] += np.diff(vor, axis=0)
    gr[:, :-1] += np.abs(np.diff(vor, axis=1))
    
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

def plot_voronoi_filterbank(mfb):

    """ Plot 2D empirical Voronoi filters

    Parameters:
    -----------
    - mfb: list
        list containing each Voronoi filter

    Author: Jerome Gilles
    Institution: San Diego State University
    Version: 1.0 (07/16/2025)    
    """

    # compute the structure of the grid
    nc = int(np.ceil(np.sqrt(len(mfb))))
    if nc > 4:
        nc = 4
    nr = int(np.ceil(len(mfb)/nc))
    if nr > 4:
        nr = 4

    nfig = int(np.ceil(len(mfb) / (nr * nc)))

    fign = 0
    for figi in range(int(nfig)):
        figmfb, axmfb = plt.subplots(nr, nc, figsize=(12, 8))
        k = figi * nr * nc

        if nr == 1:
            for kc in np.arange(nc):
                if k < len(mfb):
                    axmfb[kc].imshow(np.real(mfb[k]), cmap='gray', interpolation='none')
                    axmfb[kc].set_title("Filter " + str(k))
                    axmfb[kc].axis('off')
                    k += 1
                    if k >= len(mfb):
                        for ax in axmfb.flat[len(mfb):]:
                            ax.remove()
                        break
        else:
            for kr in np.arange(nr):
                for kc in np.arange(nc):
                    if k < len(mfb):
                        axmfb[kr, kc].imshow(np.real(mfb[k]), cmap='gray', interpolation='none')
                        axmfb[kr, kc].set_title("Filter " + str(k))
                        axmfb[kr, kc].axis('off')
                        k += 1

                        if k >= len(mfb):
                            # Remove the empty plots
                            for ax in axmfb.flat[len(mfb)-figi*nc*nr:]:
                                ax.remove()
                            break
        fign += 1
        plt.tight_layout()
        plt.show()

    return