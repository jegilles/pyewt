import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import pyewt


# set parameters
params = {
    # General parameters
    "log": False, # if True the boundaries detection will be perform on the log of the spectrum
    "N": 4, # Number of expected modes
    "SamplingRate": 2*np.pi, # sampling frequency, if unknown set to 2pi for a normalized frequency line. i.e. [-pi,pi]
    
    # Global trend parameters
    "globtrend": "none",  # options: none, powerlaw, polylaw, morpho, tophat, opening
    "degree": 7, # degree of polynomial interpolation for polylaw

    # Regularization parameters
    "reg": "none", # options: none, gaussian, average, closing
    "lengthfilter": 11, # length of filter
    "sigmafilter": 1.5, # standard deviation for Gaussian regularization
    
    # Boundary detection parameters
    "detect": "scalespace", #detection methods: locmax, locmaxmin, locmaxminf, adaptivereg, adaptive, scalespace
    "typeDetect": "otsu", #scale-space method: "otsu", "halfnormal", "empiricallaw", "mean", "kmeans"
    "InitBounds": np.array([5, 28, 54, 81]), #array of initial bounds to be adapted (adaptive and adaptivereg methods)
    
    "Completion": False, # Request completion of the number of mode if less than N
}

# Choose which image to test, the options are 'texture', 'lena' and 'barbara'
imtotest = 'barbara'

# Choose what to plot
plot_filters = True # plot the filterbank
plot_comp = True # plot empirical coefficients
plot_bounds = True # plot the Fourier supports
plot_inverse = True # plot the inverse transform

# Load the image
if imtotest == 'lena':
    f = mpimg.imread('lena.png')
elif imtotest == 'barbara':
    f = mpimg.imread('barb.png')
else:
    f = mpimg.imread('texture.png')

# Perform the 2D Littlewood-Paley EWT
ewtlp, mfb, BR = pyewt.ewt2d_lp(f,params)

# Plot the filterbank
if plot_filters:
    pyewt.plot_lp_filterbank(mfb)

# Plot the empirical wavelet coefficients
if plot_comp:
    pyewt.plot_lp_comp(ewtlp,energy=True)

# # plot the fourier supports delineated by the rows and columns boundaries
if plot_bounds:
    pyewt.plot_lp_rings(f,BR)
    
# # compute the inverse transform and plot it
if plot_inverse:
    inv = pyewt.iewt2d_lp(ewtlp, mfb)

    figinv, axinv = plt.subplots(1,2, figsize=(18,6))
    axinv[0].imshow(np.real(f), cmap="gray", interpolation='none')
    axinv[0].set_title('Original')
    axinv[0].axis('off')
    axinv[1].imshow(np.real(inv), cmap="gray", interpolation='none')
    axinv[1].set_title('Inverse')
    axinv[1].axis('off')

    plt.tight_layout()
    plt.show()

    # compute the reconstruction error
    print("Reconstruction error = " + str(LA.norm(f-inv)))