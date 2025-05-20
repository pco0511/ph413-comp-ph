from scipy import ndimage, datasets, interpolate
import scipy as sp
import numpy as np

# generate projection (forward radon transf.)
def radon_transform(Image, AngleArr):
    Sinogram = np.zeros((Image.shape[0],(len(AngleArr))), dtype='float64')
    for s in range(len(AngleArr)):
        Sinogram[:,s] = np.sum(ndimage.rotate(Image, AngleArr[s], reshape=False),axis=0)
    return Sinogram


# reconstruction (inverse radon transf., filtered backprojection)
def iradon_transform(Sinogram, AngleArr, interpolation='linear'):
    Dim = Sinogram.shape[0]
    #radon_image = sinogram_circle_to_square(radon_image)
    AngleArr_rad = (np.pi / 180.0) * AngleArr
    
    # resize image to next power of two (but no less than 64) for
    # Fourier analysis; speeds up Fourier and lessens artifacts
    projection_size_padded = max(64, int(2 ** np.ceil(np.log2(2 * Sinogram.shape[0]))))
    pad_width = ((0, projection_size_padded - Sinogram.shape[0]), (0, 0))
    PaddedSinogram = np.pad(Sinogram, pad_width, mode='constant', constant_values=0)
    
    f = np.fft.fftfreq(projection_size_padded).reshape(-1, 1)   # digital frequency
    omega = 2 * np.pi * f                                # angular frequency
    fourier_filter = 2 * np.abs(f)                       # ramp filter
    Projection = np.fft.fft(PaddedSinogram, axis=0) * fourier_filter
    
    FilteredSinogram = np.real(np.fft.ifft(Projection, axis=0))
    FilteredSinogram = FilteredSinogram[:Sinogram.shape[0], :] # undo padding
    
    Reconstruction = np.zeros((Dim, Dim))
    Mid_index = Sinogram.shape[0] // 2
    [X, Y] = np.mgrid[0:Dim, 0:Dim]
    xpr = X - int(Dim) // 2
    ypr = Y - int(Dim) // 2
    
    # Reconstruct image by adding interpolated backprojections
    interpolation_types = ('linear', 'nearest', 'cubic')
    
    if interpolation not in interpolation_types:
        raise ValueError("Unknown interpolation: %s" % interpolation)
        
    for i in range(len(AngleArr_rad)):
        t = + ypr * np.cos(AngleArr_rad[i]) + xpr * np.sin(AngleArr_rad[i])
        x = np.arange(FilteredSinogram.shape[0]) - Mid_index
        if interpolation == 'linear':
            Backprojected = np.interp(t, x, FilteredSinogram[:, i],
                                      left=0, right=0)
        else:
            interpolant = sp.interpolate.interp1d(x, FilteredSinogram[:, i], kind=interpolation,
                                   bounds_error=False, fill_value=0)
            Backprojected = interpolant(t)
            
        Reconstruction += Backprojected
    Rad = Dim // 2
    Rec_circle = (xpr ** 2 + ypr ** 2) <= Rad ** 2
    Reconstruction[~Rec_circle] = 0.

    return Reconstruction * np.pi / (2 * len(AngleArr_rad))