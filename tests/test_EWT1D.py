import numpy as np
import matplotlib.pyplot as plt
import scipy as scipy
from numpy import linalg as LA

import pyewt


# ===============================================================
#                          EWT 1D DEMO
# ===============================================================

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

    # 1D transform
    "wavname": "littlewood-paley", # mother wavelet: littlewood-paley, shannon, meyer, gabor1, gabor2
}

# set input signal
# available signals: sig1, sig2, sig3, ecg (default), seismic, eeg, heeg, 
# csig1 (complex signal)s
sigtotest = "csig1"

# set wanted ouput (either True or False)
plot_bounds = True
plot_filters = True
plot_ewt = False
plot_reconstruction = True
plot_timefrequency = False

# Load the selected signal (s is the signal, f either the full or half magnitude spectrum depending if s is real or complex)
cpx = False
if sigtotest == "csig1": # complex signal
    s = np.genfromtxt('csig1.csv', converters={0: lambda x: x.replace('i','j')},
                dtype=complex).astype(complex)
    f = np.abs(np.fft.fft(s))
    cpx = True
elif sigtotest == "sig1":
    s = np.genfromtxt('sig1.csv', delimiter=',')
    f = np.abs(np.fft.rfft(s))
elif sigtotest == "sig2":
    s = np.genfromtxt('sig2.csv', delimiter=',')
    f = np.abs(np.fft.rfft(s))
elif sigtotest == "sig3":
    s = np.genfromtxt('sig3.csv', delimiter=',')
    f = np.abs(np.fft.rfft(s))
elif sigtotest == "seismic":
    s = np.genfromtxt('seismic.csv', delimiter=',')
    s = s[::10] # we downsample to save time
    f = np.abs(np.fft.rfft(s))
elif sigtotest == "eeg":
    s = np.genfromtxt('eeg.csv', delimiter=',')
    f = np.abs(np.fft.rfft(s))
elif sigtotest == "heeg":
    s = np.genfromtxt('Heeg.csv', delimiter=',')
    f = np.abs(np.fft.rfft(s))
else:   # ECG by default
    s = np.genfromtxt('sig4.csv', delimiter=',')
    f = np.abs(np.fft.rfft(s))


# we compute the transform
ewt, mfb, bounds = pyewt.ewt1d(s, params)

# to plot the boundaries
if plot_bounds:
    if cpx:
        f = np.fft.fftshift(f)
        pyewt.plotboundaries(f,bounds,("Bounds",params["typeDetect"]),Cpx=1)
    else:
        pyewt.plotboundaries(f,bounds,("Bounds",params["typeDetect"]))

# to plot the filters
if plot_filters:
    pyewt.plot_filterBank1d(mfb,spectrum=np.abs(np.fft.fft(s)),Grouped=False,SamplingRate=params["SamplingRate"])

# to plot the EW components
if plot_ewt:
    pyewt.plot_ewt1d(ewt,SamplingRate=params["SamplingRate"])

# to plot the reconstructed signal
if plot_reconstruction:
    # compute the inverse transform
    srec = pyewt.iewt1d(ewt, mfb, cpx)

    if cpx:
        figrec, axrec = plt.subplots(2,2, figsize=(8,5))
        axrec[0][0].plot(np.real(s))
        axrec[0][1].plot(np.imag(s))
        axrec[0][0].set_title("Original - Real part")
        axrec[0][1].set_title("Original - Imaginary part")
        axrec[1][0].plot(np.real(srec))
        axrec[1][1].plot(np.imag(srec))
        axrec[1][0].set_title("Reconstructed - Real part")
        axrec[1][1].set_title("Reconstructed - Imaginary part")
        plt.tight_layout()
        plt.show()
    else:
        figrec, axrec = plt.subplots(2,1, figsize=(8,5))
        axrec[0].plot(s)
        axrec[0].set_title("Original")
        axrec[1].plot(srec)
        axrec[1].set_title("Reconstructed")
        plt.tight_layout()
        plt.show()

    # compute the reconstruction error
    print("Reconstruction error = " + str(LA.norm(s-srec)))

# to plot the time-frequency domain
if plot_timefrequency:
    pyewt.timefrequency_plot(ewt, bounds,s,SamplingRate=params["SamplingRate"],GlobalNorm=False)