import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import pyewt

# This script is to test the pseudo polar Fourier transform

# Choose which image to test, the options are 'texture', 'lena' and 'barbara'
imtotest = 'barbara'

# Load the image
if imtotest == 'lena':
    f = mpimg.imread('lena.png')
elif imtotest == 'barbara':
    f = mpimg.imread('barb.png')
else:
    f = mpimg.imread('texture.png')

pft = pyewt.PPFFT(f)

fig = plt.figure(figsize=(10,10))
plt.imshow(np.log(1+np.abs(pft)),cmap='gray',interpolation='none')