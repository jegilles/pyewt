import numpy as np
import matplotlib.pyplot as plt
import scipy as scipy

import pyewt

# ===============================================================
#                   BOUNDARY DETECTION DEMO
# ===============================================================

# set parameters
params = {
    # General parameters
    "log": 0, # if 1 the boundaries detection will be perform on the log of the spectrum
    "N": 4, # Number of expected modes
    "SamplingRate": -1, 
    
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
    
    "Completion": 0 # Request completion of the number of mode if less than N
}

# available signals: sig1, sig2, sig3, ecg, seismic, eeg, heeg, 
# texture (default), csig1 (complex signal)

sigtotest = "csig1"

# Load the selected histogram/signal
if sigtotest == "ecg":
    s = np.genfromtxt('sig4.csv', delimiter=',')
    f = np.abs(np.fft.rfft(s))
    sizef = np.size(f) * 2
    cpx = 0
elif sigtotest == "sig1":
    s = np.genfromtxt('sig1.csv', delimiter=',')
    f = np.abs(np.fft.rfft(s))
    sizef = np.size(f) * 2
    cpx = 0
elif sigtotest == "sig2":
    s = np.genfromtxt('sig2.csv', delimiter=',')
    f = np.abs(np.fft.rfft(s))
    sizef = np.size(f) * 2
    cpx = 0
elif sigtotest == "sig3":
    s = np.genfromtxt('sig3.csv', delimiter=',')
    f = np.abs(np.fft.rfft(s))
    sizef = np.size(f) * 2
    cpx = 0
elif sigtotest == "seismic":
    s = np.genfromtxt('seismic.csv', delimiter=',')
    s = s[::10] # we downsample to save time
    f = np.abs(np.fft.rfft(s))
    sizef = np.size(f) * 2
    cpx = 0
elif sigtotest == "eeg":
    s = np.genfromtxt('eeg.csv', delimiter=',')
    #f = np.log(1+np.abs(np.fft.rfft(s)))
    f = np.abs(np.fft.rfft(s))
    sizef = np.size(f) * 2
    cpx = 0
elif sigtotest == "heeg":
    s = np.genfromtxt('Heeg.csv', delimiter=',')
    #f = np.log(1+np.abs(np.fft.rfft(s)))
    f = np.abs(np.fft.rfft(s))
    sizef = np.size(f) * 2
    cpx = 0
elif sigtotest == "csig1":
    s = np.genfromtxt('csig1.csv', converters={0: lambda x: x.replace('i','j')},
                dtype=complex).astype(complex)
    f = np.abs(np.fft.fft(s))
    f = np.fft.fftshift(f)
    sizef = np.size(f)
    cpx = 1
else:
    # this file directly provides the histogram
    s = 0
    f = np.genfromtxt('signals/texture.csv', delimiter=',')
    sizef = np.size(f) * 2
    cpx = 0

# bounds, presig = Boundaries_Detect(f,params)
bounds, presig = pyewt.boundaries_detect(f,params)
bounds = bounds * 2 * np.pi / sizef
if cpx == 1:
    bounds = bounds - np.pi

# if signal is real, we remove a boundary at 0
if np.isrealobj(s):
    if bounds[0] == 0:
        bounds = bounds[1:]

# freq = PlotBoundaries(f,bounds,("Bounds",params["typeDetect"]), Cpx = cpx, logtag = params["log"], SamplingRate = params["SamplingRate"])
freq = pyewt.plotboundaries(f,bounds,("Bounds",params["typeDetect"]), Cpx = cpx, logtag = params["log"], SamplingRate = params["SamplingRate"])