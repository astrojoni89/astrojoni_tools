import numpy as np
from tqdm import trange
from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
from astropy.coordinates import Galactic
from spectral_cube import SpectralCube




def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx


#convert axes indices to world coordinates
def velocity_axes(name):
    header = fits.getheader(name)
    n = header['NAXIS3'] #number of channels on spectral axis
    velocity = (header['CRVAL3'] - header['CRPIX3'] * header['CDELT3']) + (np.arange(n)+1) * header['CDELT3']
    velocity = velocity / 1000
    return velocity
  
def latitude_axes(name):
    header = fits.getheader(name)
    n = header['NAXIS2'] #number of pixels along latitude axis
    latitude = (header['CRVAL2'] - header['CRPIX2'] * header['CDELT2']) + (np.arange(n)+1) * header['CDELT2']
    return latitude

def longitude_axes(name):
    header = fits.getheader(name)
    n = header['NAXIS1'] #number of pixels along longitude axis
    longitude = (header['CRVAL1'] - header['CRPIX1'] * header['CDELT1']) + (np.arange(n)+1) * header['CDELT1']
    return longitude


#convert world coordinates to pixel values from .FITS
def world_to_pixel(fitsfile,longtitude,latitude,velocity=0):
    w = WCS(fitsfile)
    if w.wcs.naxis == 3:
        return w.all_world2pix(longitude, latitude, velocity, 1)
    elif w.wcs.naxis == 2:
        return w.all_world2pix(longitude, latitude, 1)
    else:
        raise ValueError('Something wrong with the header.')

def pixel_to_world(fitsfile,x,y,ch=0):
    w = WCS(fitsfile)
    if w.wcs.naxis == 3:
        return w.all_pix2world(x, y, ch, 1)
    elif w.wcs.naxis == 2:
        return w.all_pix2world(x, y, 1)
    else:
        raise ValueError('Something wrong with the header.')


def calculate_spectrum(fitsfile,pixel_array):
    header = fits.getheader(fitsfile)
    image = fits.getdata(fitsfile)
    number_of_channels = header['NAXIS3']
    spectrum_add = np.zeros(number_of_channels)
    n=0
    for i in trange(0,len(pixel_array)):
        x_1,y_1 = pixel_array[i]
        spectrum_i = image[:,y_1,x_1]
        if any([np.isnan(spectrum_i[i]) for i in range(len(spectrum_i))]):
            print('Warning: region contains NaNs!')
            spectrum_add = spectrum_add + 0
            n+=1
        else:
            spectrum_add = spectrum_add + spectrum_i
    spectrum_average = spectrum_add / (len(pixel_array)-n)
    return spectrum_average

def calculate_average_value_pixelArray(fitsfile,pixel_array): #nan treatment?
    image = fits.getdata(fitsfile)
    value_add = 0
    n=0
    for i in range(0,len(pixel_array)):
        x_1,y_1 = pixel_array[i]
        value_i = image[y_1,x_1]
        if np.isnan(value_i):
            print('Warning: region contains NaNs!')
            value_add = value_add + 0
            n+=1
        else:
            value_add = value_add + value_i
    if n<(len(pixel_array)/3.):
        value_average = value_add / (len(pixel_array)-n)
    else:
        value_average = np.nan
    return value_average
 
