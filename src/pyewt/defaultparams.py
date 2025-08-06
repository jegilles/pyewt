import numpy as np

def Default_Params():

    """
    This function creates a set of parameters, stored in params, with default values

    Returns:
    - params: dictionary containing the different parameters

    Author: Jerome Gilles
    Institution: San Diego State University
    Version: 1.0 (12/18/2024)
    """
    
    params = {
        # General parameters
        "log": False, # if 1 the boundaries detection will be perform on the log of the spectrum
        "N": 4, # Number of expected modes
        "SamplingRate": 2*np.pi, # sampling frequency, if unknown set to 2pi for a normalized frequency line. i.e. [-pi,pi]
    
        # Global trend parameters
        "globtrend": "none",  # options: none, powerlaw, polylaw, morpho, tophat, opening
        "degree": 7, # degree of polynomial interpolation for polylaw

        # Regularization parameters
        "reg": "none", # options: none, gaussian, average, closing
        "lengthfilter": 3, # length of filter
        "sigmafilter": 1.5, # standard deviation for Gaussian regularization
    
        # Boundary detection parameters
        "detect": "scalespace", #detection methods: locmax, locmaxmin, locmaxminf, adaptivereg, adaptive, scalespace
        "typeDetect": "otsu", #scale-space method: "otsu", "halfnormal", "empiricallaw", "mean", "kmeans"
        "kn": 3, #scale kernel size for scalespace kernel
        "t": 0.16, #initial scale
        "InitBounds": np.array([5, 28, 54, 81]), #array of initial bounds to be adapted (adaptive and adaptivereg methods)
        "Completion": False, # Request completion of the number of mode if less than N

        # 1D transform
        "wavname": "littlewood-paley", # mother wavelet: littlewood-paley, shannon, meyer, gabor1, gabor2

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

    return params
