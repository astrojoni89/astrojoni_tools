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


#convert world/pixel coords to pixel/world coords from .FITS
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
 

def moment_0(fitsfile,velocity_start,velocity_end):
    image = fits.getdata(fitsfile)
    header = fits.getheader(fitsfile)
    velocity = velocity_axes(fitsfile)
    velocity = velocity.round(decimals=4)
    lower_channel = find_nearest(velocity,velocity_start)
    upper_channel = find_nearest(velocity,velocity_end)
    print('channel-range: '+str(lower_channel)+' - '+str(upper_channel))
    print('velocity-range: '+str(velocity[lower_channel])+' - '+str(velocity[upper_channel]))
    if header['NAXIS']==4:
        moment_0_map = np.zeros((1,1,header['NAXIS2'],header['NAXIS1']))
        for i in range(lower_channel,upper_channel+1,1):
            moment_0_map = moment_0_map + image[0,i,:,:]
    elif header['NAXIS']==3:
        moment_0_map = np.zeros((1,header['NAXIS2'],header['NAXIS1']))
        for i in range(lower_channel,upper_channel+1,1):
            moment_0_map = moment_0_map + image[i,:,:]
    else:
        print('Something wrong with the header.')
    moment_0_map = moment_0_map *header['CDELT3']/1000
    header['BUNIT'] = header['BUNIT']+'.KM/S'
    fits.writeto(fitsfile.split('.fits')[0]+'_mom-0_'+str(velocity_start)+'_to_'+str(velocity_end)+'km-s.fits', moment_0_map, header=header, overwrite=True)
    print(fitsfile.split('.fits')[0]+'_mom-0_'+str(velocity_start)+'_to_'+str(velocity_end)+'km-s.fits')
    return [fitsfile.split('.fits')[0]+'_mom-0_'+str(velocity_start)+'_to_'+str(velocity_end)+'km-s.fits',lower_channel,upper_channel]


def add_up_channels(fitsfile,velocity_start,velocity_end):
    image = fits.getdata(fitsfile)
    header = fits.getheader(fitsfile)
    velocity = velocity_axes(fitsfile)
    velocity = velocity.round(decimals=4)
    lower_channel = find_nearest(velocity,velocity_start)
    upper_channel = find_nearest(velocity,velocity_end)
    print('channel-range: '+str(lower_channel)+' - '+str(upper_channel))
    print('velocity-range: '+str(velocity[lower_channel])+' - '+str(velocity[upper_channel]))
    if header['NAXIS']==4:
        moment_0_map = np.zeros((1,header['NAXIS2'],header['NAXIS1']))
        for i in range(lower_channel,upper_channel+1,1):
            moment_0_map = moment_0_map + image[0,i,:,:]
    elif header['NAXIS']==3:
        moment_0_map = np.zeros((1,header['NAXIS2'],header['NAXIS1']))
        for i in range(lower_channel,upper_channel+1,1):
            moment_0_map = moment_0_map + image[i,:,:]
    else:
        print('Something wrong with the header.')
    fits.writeto(fitsfile.split('.fits')[0]+'_sum_'+str(velocity[lower_channel])+'_to_'+str(velocity[upper_channel])+'km-s.fits', moment_0_map, header=header, overwrite=True)
    print(fitsfile.split('.fits')[0]+'_sum_'+str(velocity[lower_channel])+'_to_'+str(velocity[upper_channel])+'km-s.fits')
    return [fitsfile.split('.fits')[0]+'_sum_'+str(velocity[lower_channel])+'_to_'+str(velocity[upper_channel])+'km-s.fits',lower_channel,upper_channel]


def channel_averaged(fitsfile,velocity_start,velocity_end):
    image = fits.getdata(fitsfile)
    header = fits.getheader(fitsfile)
    velocity = velocity_axes(fitsfile)
    velocity = velocity.round(decimals=4)
    lower_channel = find_nearest(velocity,velocity_start)
    upper_channel = find_nearest(velocity,velocity_end)
    print('channel-range: '+str(lower_channel)+' - '+str(upper_channel))
    print('velocity-range: '+str(velocity[lower_channel])+' - '+str(velocity[upper_channel]))
    j=0
    if header['NAXIS']==4:
        moment_0_map = np.zeros((1,1,header['NAXIS2'],header['NAXIS1']))
        for i in range(lower_channel,upper_channel+1,1):
            moment_0_map = moment_0_map + image[0,i,:,:]
            j=j+1
    elif header['NAXIS']==3:
        moment_0_map = np.zeros((1,header['NAXIS2'],header['NAXIS1']))
        for i in range(lower_channel,upper_channel+1,1):
            moment_0_map = moment_0_map + image[i,:,:]
            j=j+1
    else:
        print('Something wrong with the header...')
    moment_0_map = moment_0_map /float(j)
    fits.writeto(fitsfile.split('.fits')[0]+'_averaged-channel_'+str(velocity_start)+'_to_'+str(velocity_end)+'km-s.fits', moment_0_map, header=header, overwrite=True)
    print(fitsfile.split('.fits')[0]+'_mom-0_'+str(velocity_start)+'_to_'+str(velocity_end)+'km-s.fits')
    return [fitsfile.split('.fits')[0]+'_mom-0_'+str(velocity_start)+'_to_'+str(velocity_end)+'km-s.fits',lower_channel,upper_channel]


def pixel_circle_calculation(fitsfile,longitude,latitude,radius):
    #longitude and latitude in degree
    #radius in arcsec
    #give central coordinates, size of circle and fitsfile and it returns array with the corresponding pixels 
    header = fits.getheader(fitsfile)
    w = WCS(fitsfile)
    delta = abs(header['CDELT1']) #in degree
    pixel_array = []
    if header['NAXIS']==3:
        central_px = w.all_world2pix(longitude,latitude,0,1)
    elif header['NAXIS']==2:
        central_px = w.all_world2pix(longitude,latitude,1)
    else:
        raise Exception('Something wrong with the header.')
    central_px = [int(np.round(central_px[0],decimals=0))-1,int(np.round(central_px[1],decimals=0))-1]
    if radius is not 'single':
        circle_size_px = 2*radius/3600. / delta
        circle_size_px = int(round(circle_size_px))
        px_start = [central_px[0]-circle_size_px/2,central_px[1]-circle_size_px/2]
        px_end = [central_px[0]+circle_size_px/2,central_px[1]+circle_size_px/2]
        px_start = [int(np.round(px_start[0],decimals=0)),int(np.round(px_start[1],decimals=0))]
        px_end = [int(np.round(px_end[0],decimals=0)),int(np.round(px_end[1],decimals=0))]
        for i_x in trange(px_start[0]-1,px_end[0]+1):
            for i_y in range(px_start[1]-1,px_end[1]+1):
                if sqrt((i_x-central_px[0])**2+(i_y-central_px[1])**2) < circle_size_px/2.:
                    pixel_array.append((i_x,i_y))
    else:
        pixel_array.append((central_px[0],central_px[1]))
    return pixel_array


def pixel_box_calculation(fitsfile,longitude,latitude,a,b):
    #longitude and latitude in degree
    #a,b: total size of longitude,latitude box in arcsec
    #give central coordinates, size of box and fitsfile and it returns array with the corresponding pixels 
    header = fits.getheader(fitsfile)
    w = WCS(fitsfile)
    delta = abs(header['CDELT1']) #in degree
    pixel_array = []
    if header['NAXIS']==3:
        central_px = w.all_world2pix(longitude,latitude,0,1)
    elif header['NAXIS']==2:
        central_px = w.all_world2pix(longitude,latitude,1)
    else:
        raise Exception('Something wrong with the header.')
    central_px = [int(np.round(central_px[0],decimals=0))-1,int(np.round(central_px[1],decimals=0))-1]
    if a is not 'single':
        box_size_px_a = a/3600. / delta
        box_size_px_a = int(round(box_size_px_a))
        box_size_px_b = b/3600. / delta
        box_size_px_b = int(round(box_size_px_b))
        px_start = [central_px[0]-box_size_px_a/2,central_px[1]-box_size_px_b/2]
        px_end = [central_px[0]+box_size_px_a/2,central_px[1]+box_size_px_b/2]
        px_start = [int(np.round(px_start[0],decimals=0)),int(np.round(px_start[1],decimals=0))]
        px_end = [int(np.round(px_end[0],decimals=0)),int(np.round(px_end[1],decimals=0))]
        for i_x in trange(px_start[0],px_end[0]):
            for i_y in range(px_start[1],px_end[1]):
                pixel_array.append((i_x,i_y))
    else:
        pixel_array.append((central_px[0],central_px[1]))
    return pixel_array

