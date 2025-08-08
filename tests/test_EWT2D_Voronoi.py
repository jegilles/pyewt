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
    "reg": "gaussian", # options: none, gaussian, average
    "lengthfilter": 3, # length of filter for regularization
    "sigmafilter": 1.5, # standard deviation for Gaussian regularization
    
    # Boundary detection parameters
    "detect": "scalespace", #detection methods: locmax, locmaxmin, locmaxminf, adaptivereg, adaptive, scalespace
    "typeDetect": "otsu", #scale-space method: "otsu", "halfnormal", "empiricallaw", "mean", "kmeans"
    "kn": 6, #scale kernel size for scalespace kernel
    "t": 0.8, #initial scale
    "InitBounds": np.array([5, 28, 54, 81]), #array of initial bounds to be adapted (adaptive and adaptivereg methods)
    "Completion": False, # Request completion of the number of mode if less than N

    # Curvelet parameters
    "option": 3, # specify which transform option: 1 = independent scales and angles, 2 = scales first then angles, 3 = angles first then scales
    "curvdegree": 4, # degree of polynomial interpolation for polylaw for angles detection
    "curvpreproc": "none",  # options: none, powerlaw, polylaw, morpho, tophat, opening for angles detection
    "curvreg": "none", # options: none, gaussian, average, closing for angles detection
    "curvlengthfilter": 10, # length of filter for angles detection
    "curvsigmafilter": 1.5, # standard deviation for Gaussian regularization for angles detection
    "curvmethod": "scalespace", # detection methods: locmax, locmaxmin, scalespace for angles detection
    "curvN": 6, # number of expected angular sector

    # 2D Gaussian scale space extra parameters
    "niter": 4, # number of iterations
    "edge": 0, # size (in pixels) of the strip to ignore at the edge of the image
    "includeCenter": 0, # if 1, the center of the image is included in the scale space maxima

    # Voronoi and watershed partition parameters
    "complex": 0, # if 1, the Voronoi partition is complex, otherwise it is real
    "tau": 0.1 # half width of the transition area
}

# Choose what to plot
plot_filters = True # plot the filterbank
plot_comp = True # plot empirical coefficients
plot_bounds = True # plot the Fourier supports
plot_inverse = True # plot the inverse transform
plot_maxima = False # plot the scale-space maxima

# Choose which image to test, the options are 'texture', 'lena' and 'barbara'
imtotest = 'texture'
# Load the image
if imtotest == 'lena':
    f = mpimg.imread('lena.png')
elif imtotest == 'barbara':
    f = mpimg.imread('barb.png')
else:
    f = mpimg.imread('texture.png')

# Compute the empirical voronoi wavelet transform
ewtv, mfb, maxima, vorpartition, plane = pyewt.ewt2d_voronoi(f,params)

# Plot the detected scale-space maxima on the Fourier spectrum
if plot_maxima:
    pyewt.plot_scalespace_maxima(f,maxima)

# plot the Fourier supports delineated by the Voronoi cells boundaries
if plot_bounds:
    pyewt.show_ewt2d_voronoi_boundaries(f, vorpartition, logspec=1)

# Plot the filterbank
if plot_filters:
    pyewt.plot_voronoi_filterbank(mfb)

# Plot the empirical wavelet coefficients
if plot_comp:
    pyewt.plot_lp_comp(ewtv,energy=True)

# compute the inverse transform and plot it
if plot_inverse:
    inv = pyewt.iewt2d_voronoi(ewtv, mfb)

    figinv, axinv = plt.subplots(1,2, figsize=(18,6))
    axinv[0].imshow(np.real(f), cmap="gray", interpolation='none')
    axinv[0].set_title('Original')
    axinv[0].axis('off')
    axinv[1].imshow(np.real(inv), cmap="gray", interpolation='none')
    axinv[1].set_title('Inverse')
    axinv[1].axis('off')
    plt.show()  # Show all plots at once

    print("Reconstruction error = " + str(LA.norm(f-inv)))