import numpy as np

def my_FRFT(x, alpha):

    """ Fractional Fourier Transform

    Parameters
    ----------
    - x: nd-array
        image to transform
    - alpha: scaling factor
        for pseudo-polar it should be in the range [-1/N,1/N] where is the number
        of elements in x

    Returns
    -------
    - y: 1D nd-array
        transformed vector of an N elements input

    Notes
    -----
    This function computes the fractional Fourier Transform

    y[n]=sum_{k=0}^{N-1} x(k) e^{-i 2pi k n alpha}   ;    n=0,1,..., N-1

    So that for alpha=1/N we have the regular FFT, anf for alpha=-1/n we have the regular IFFT

    This code is a translation of the Matlab code provided by Michael Elad in his Polarlab 
    toolbox: https://elad.cs.technion.ac.il/software/

    Author: Jerome Gilles
    Institution: San Diego State University
    Version: 1.0 (01/23/2025)
    """

    x = np.ravel(x, order='F')
    N = np.size(x)

    n = np.hstack((np.linspace(0,N-1,N), np.linspace(-N,-1,N)))
    Factor = np.exp(-1j * np.pi * alpha * np.square(n))

    xt = np.hstack((x , np.zeros(N).astype(np.complex128)))
    xt = np.multiply(xt,Factor)

    XX = np.fft.fft(xt)
    YY = np.fft.fft(np.conjugate(Factor))

    y = np.fft.ifft(np.multiply(XX,YY))
    y = np.multiply(y,Factor)
    y = y[0:N]

    return y

def my_FRFT_Centered(x,alpha):

    """ Centered Fractional Fourier Transform

    Parameters
    ----------
    - x: nd-array
        image to transform
    - alpha: scaling factor
        for pseudo-polar it should be in the range [-1/N,1/N] where is the number
        of elements in x

    Returns
    -------
    - y: 1D nd-array
        transformed vector of an N elements input

    Notes
    -----
    This function computes the fractional Fourier Transform

    y[n]=sum_{k=0}^{N-1} x(k) e^{-i 2pi k n alpha}   ;    n=0,1,..., N-1

    So that for alpha=1/N we have the regular FFT, anf for alpha=-1/n we have the regular IFFT

    This code is a translation of the Matlab code provided by Michael Elad in his Polarlab 
    toolbox: https://elad.cs.technion.ac.il/software/

    Author: Jerome Gilles
    Institution: San Diego State University
    Version: 1.0 (01/23/2025)
    """

    x = np.ravel(x, order='F')
    N = np.size(x)

    Factor2 = np.exp(1j * np.pi * np.arange(N) * N * alpha)
    xt = np.multiply(x,Factor2)

    n = np.hstack((np.linspace(0,N-1,N), np.linspace(-N,-1,N)))
    Factor = np.exp(-1j * np.pi * alpha * np.square(n))

    xt = np.hstack((xt, np.zeros(N).astype(np.complex128)))
    xt = np.multiply(xt,Factor)

    XX = np.fft.fft(xt)
    YY = np.fft.fft(np.conjugate(Factor))

    y = np.fft.ifft(np.multiply(XX,YY))
    y = np.multiply(y,Factor)
    y = y[0:N]

    return y

def PPFFT(X):

    """ Pseudo Polar Fourier Transform

    Parameters
    ----------
    - X: 2D nd-array
        Input image

    Returns
    -------
    - Y: 2D nd-array
        Pseudo polar Fourier transform (vertical=theta, horizontal=radius)
    
    Notes
    -----
    This function performs the pseudo (recto) polar Fourier transform of the 
    image X. If X is of size N x N, the output will have size 2N x 2N. If X 
    is not square, it is squared before the transform (padded with 0s). Also
    the size is increased to get even numbers of rows and columns.
        
    This code is inspired by the Matlab code provided by Michael Elad in his 
    Polarlab toolbox: https://elad.cs.technion.ac.il/software/

    Author: Jerome Gilles
    Institution: San Diego State University
    Version: 1.0 (01/23/2025)
    """

    if X.dtype != np.complex128:
        X = X.astype(np.complex128)

    # STEP 1: check sizes
    [N1, N2] = np.shape(X)
    N = np.ceil(np.max([N1, N2])/2).astype(int) * 2

    Xnew = np.zeros((N,N)).astype(np.complex128)
    Xnew[int(N/2-np.floor(N1/2)):int(N/2-np.floor(N1/2)+N1), int(N/2-np.floor(N2/2)):int(N/2-np.floor(N2/2)+N2)] = X
    X = Xnew
    Y = np.zeros((2*N,2*N)).astype(np.complex128)

    # STEP 2: construct quadrant 1 and 3
    ft = np.fft.fft(np.vstack((X,np.zeros((N,N)))),axis=0)
    ft = np.fft.fftshift(ft,axes=(0,))

    for ll in np.linspace(-N,N-1,2*N):
        Y[int(ll+N),0:N] = np.flip(my_FRFT_Centered(ft[int(ll)+N,:],ll/(N**2)))

    # STEP 3: construct quadrant 2 and 4
    ft = np.fft.fft(np.hstack((X,np.zeros((N,N)))),axis=1)
    ft = np.fft.fftshift(ft,axes=(1,))
    ft = np.transpose(ft)

    for ll in np.linspace(-N,N-1,2*N):
        Factor = np.exp(1j * 2 * np.pi * np.linspace(0,N-1,N) * (N/2-1) * ll /(N**2))
        Y[int(ll+N),N:] = my_FRFT(np.multiply(ft[int(ll)+N,:],Factor),ll/(N**2))
        
    return Y
