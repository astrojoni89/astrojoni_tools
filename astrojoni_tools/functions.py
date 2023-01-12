import os
import numpy as np
from tqdm import trange
from astropy.io import fits
from astropy.wcs import WCS



def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx


def md_header_2d(fitsfile):
    header_2d = fits.getheader(fitsfile)
    del header_2d['NAXIS3']
    del header_2d['CRPIX3']
    del header_2d['CDELT3']
    del header_2d['CUNIT3']
    del header_2d['CTYPE3']
    del header_2d['CRVAL3']

    header_2d['NAXIS'] = 2
    header_2d['WCSAXES'] = 2
    return header_2d


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
    '''
    This function returns an average spectrum
    pixel_array: pixel indices to average
    '''
    header = fits.getheader(fitsfile)
    image = fits.getdata(fitsfile)
    number_of_channels = header['NAXIS3']
    spectrum_add = np.zeros(number_of_channels)
    n=0
    idxs = []
    for i in trange(0,len(pixel_array)):
        x_1,y_1 = pixel_array[i]
        spectrum_i = image[:,y_1,x_1]
        if any([np.isnan(spectrum_i[k]) for k in range(len(spectrum_i))]):
            print('Warning: region contains NaNs!')
            idxs.append(i)
            spectrum_add = spectrum_add + 0
            n+=1
        else:
            spectrum_add = spectrum_add + spectrum_i
    spectrum_average = spectrum_add / (len(pixel_array)-n)
    temp_array = np.delete(pixel_array, idxs, axis=0)
    pixel_array_without_nan_values = list(map(tuple, temp_array))
    return spectrum_average, pixel_array_without_nan_values


def calculate_average_value_pixelArray(fitsfile,pixel_array): #nan treatment?
    image = fits.getdata(fitsfile)
    value_add = 0
    n=0
    idxs = []
    for i in trange(0,len(pixel_array)):
        x_1,y_1 = pixel_array[i]
        value_i = image[y_1,x_1]
        if np.isnan(value_i):
            print('Warning: region contains NaNs!')
            idxs.append(i)
            value_add = value_add + 0
            n+=1
        else:
            value_add = value_add + value_i
    value_average = value_add / (len(pixel_array)-n)
    temp_array = np.delete(pixel_array, idxs, axis=0)
    pixel_array_without_nan_values = list(map(tuple, temp_array))
    return value_average, pixel_array_without_nan_values
 

def moment_0(fitsfile,velocity_start,velocity_end,noise=None,path_to_output='.',save_file=True,output_noise=True,suffix=''):
    import os
    image = fits.getdata(fitsfile)
    headerm0 = fits.getheader(fitsfile)
    velocity = velocity_axes(fitsfile)
    velocity = velocity.round(decimals=4)
    velocity_low = min(velocity_start,velocity_end)
    velocity_up = max(velocity_start,velocity_end)
    lower_channel = find_nearest(velocity,velocity_low)
    upper_channel = find_nearest(velocity,velocity_up)
    print('channel-range: '+str(lower_channel)+' - '+str(upper_channel))
    print('velocity-range: '+str(velocity[lower_channel])+' - '+str(velocity[upper_channel]))
    if noise is not None:
        num_ch = int(upper_channel - lower_channel)
        moment_0_noise = noise * np.sqrt(num_ch) * headerm0['CDELT3']/1000
        print('Moment 0 noise at 1 sigma: {}'.format(moment_0_noise))
    if headerm0['NAXIS']==4:
        moment_0_map = np.zeros((headerm0['NAXIS2'],headerm0['NAXIS1']))
        for i in range(lower_channel,upper_channel+1,1):
            moment_0_map = moment_0_map + image[0,i,:,:]
    elif headerm0['NAXIS']==3:
        moment_0_map = np.zeros((headerm0['NAXIS2'],headerm0['NAXIS1']))
        for i in range(lower_channel,upper_channel+1,1):
            moment_0_map = moment_0_map + image[i,:,:]
    else:
        print('Something wrong with the header.')
    moment_0_map = moment_0_map * headerm0['CDELT3']/1000
    headerm0['BUNIT'] = headerm0['BUNIT']+'.KM/S'

    filename_wext = os.path.basename(filename)
    filename_base, file_extension = os.path.splitext(filename_wext)
    newname = filename_base + '_mom-0_' + str(velocity_start) + '_to_' + str(velocity_end) + 'km-s' + suffix + '.fits'
    pathname = os.path.join(path_to_output, newname)

    if save_file is True:
        fits.writeto(pathname, moment_0_map, header=headerm0, overwrite=True)
        print("\n\033[92mSAVED FILE:\033[0m '{}' in '{}'".format(newname,path_to_output))
    else:
        print(newname.split('.fits')[0])
    if output_noise is True:
        name_noise_file = filename_base + '_mom-0_noise.txt'
        path_noise_file = os.path.join(path_to_output, name_noise_file)
        np.savetxt(path_noise_file, moment_0_noise)
    return moment_0_map

def moment_1(fitsfile,velocity_start,velocity_end,path_to_output='.',save_file=True, suffix=''):
    import os
    image = fits.getdata(fitsfile)
    headerm1 = fits.getheader(fitsfile)
    velocity = velocity_axes(fitsfile)
    velocity = velocity.round(decimals=4)
    lower_channel = find_nearest(velocity,velocity_start)
    upper_channel = find_nearest(velocity,velocity_end)
    print('channel-range: '+str(lower_channel)+' - '+str(upper_channel))
    print('velocity-range: '+str(velocity[lower_channel])+' - '+str(velocity[upper_channel]))
    if headerm1['NAXIS']==4:
        moment_1_map = np.zeros((headerm1['NAXIS2'],headerm1['NAXIS1']))
        for i in range(lower_channel,upper_channel+1,1):
            moment_1_map = moment_1_map + image[0,i,:,:] * velocity[i]
    elif headerm1['NAXIS']==3:
        moment_1_map = np.zeros((headerm1['NAXIS2'],headerm1['NAXIS1']))
        for i in range(lower_channel,upper_channel+1,1):
            moment_1_map = moment_1_map + image[i,:,:] * velocity[i]
    else:
        print('Something wrong with the header.')
    moment_0_map = moment_0(fitsfile,velocity_start,velocity_end,save_file=False)
    moment_1_map = (moment_1_map * headerm1['CDELT3']/1000) / moment_0_map
    headerm1['BUNIT'] = 'KM/S'

    filename_wext = os.path.basename(filename)
    filename_base, file_extension = os.path.splitext(filename_wext)
    newname = filename_base + '_mom-1_' + str(velocity_start) + '_to_' + str(velocity_end) + 'km-s' + suffix + '.fits'
    pathname = os.path.join(path_to_output, newname)

    if save_file is True:
        fits.writeto(pathname, moment_1_map, header=headerm1, overwrite=True)
        print("\n\033[92mSAVED FILE:\033[0m '{}' in '{}'".format(newname,path_to_output))
    else:
        print(newname.split('.fits')[0])
    return moment_1_map

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


def pixel_circle_calculation(fitsfile,xcoord,ycoord,r):
    """This function returns both a list of pixels [(y0,x0),(y1,x1),...] and a tuple ((y0,y1,...),(x0,x1,...)) corresponding to the circle region with central coordinates xcoord, ycoord, and radius r..
    
    Parameters
    ----------
    fitsfile : str
        path to FITS file.
    xcoord : numpy.ndarray or float
        x-coordinate of central pixel in units given in the header.
    ycoord : numpy.ndarray or float
        y-coordinate of central pixel in units given in the header.
    r : float
        radius of region in units of arcseconds.
    Returns
    -------
    pixel_coords : list
        List of pixel coordinates [(y0,x0),(y1,x1),...].
    indices_np : tuple
        Tuple of pixel indices ((y0,y1,...),(x0,x1,...)) to index a numpy.ndarray.
    """
    header = fits.getheader(fitsfile)
    w = WCS(fitsfile)
    delta = abs(header['CDELT1']) #in degree
    pixel_coords = []
    if header['NAXIS']==3:
        central_px = w.all_world2pix(xcoord,ycoord,0,1)
    elif header['NAXIS']==2:
        central_px = w.all_world2pix(xcoord,ycoord,1)
    else:
        raise Exception('Something wrong with the header!')
    central_px = [int(np.round(central_px[0],decimals=0))-1,int(np.round(central_px[1],decimals=0))-1]
    if r != 'single':
        circle_size_px = 2*r/3600. / delta
        circle_size_px = int(round(circle_size_px))
        px_start = [central_px[0]-circle_size_px/2,central_px[1]-circle_size_px/2]
        px_end = [central_px[0]+circle_size_px/2,central_px[1]+circle_size_px/2]
        px_start = [int(np.round(px_start[0],decimals=0)),int(np.round(px_start[1],decimals=0))]
        px_end = [int(np.round(px_end[0],decimals=0)),int(np.round(px_end[1],decimals=0))]
        for i_x in trange(px_start[0]-1,px_end[0]+1):
            for i_y in range(px_start[1]-1,px_end[1]+1):
                if np.sqrt((i_x-central_px[0])**2+(i_y-central_px[1])**2) < circle_size_px/2.:
                    pixel_coords.append((i_y,i_x))
    else:
        pixel_coords.append((central_px[1],central_px[0]))
    
    indices_np = tuple(zip(*pixel_coords))
    
    return pixel_coords, indices_np


def pixel_circle_calculation_px(fitsfile,x,y,r):
    '''
    This function returns array of pixels corresponding to circle region with central coordinates Glon, Glat, and radius r
    x: central pixel x
    y: central pixel y
    r: radius of region in arcseconds
    '''
    header = fits.getheader(fitsfile)
    w = WCS(fitsfile)
    delta = abs(header['CDELT1']) #in degree
    pixel_array = []
    central_px = [x,y]
    if r != 'single':
        circle_size_px = 2*r/3600. / delta
        circle_size_px = int(round(circle_size_px))
        px_start = [central_px[0]-circle_size_px/2,central_px[1]-circle_size_px/2]
        px_end = [central_px[0]+circle_size_px/2,central_px[1]+circle_size_px/2]
        px_start = [int(np.round(px_start[0],decimals=0)),int(np.round(px_start[1],decimals=0))]
        px_end = [int(np.round(px_end[0],decimals=0)),int(np.round(px_end[1],decimals=0))]
        for i_x in range(px_start[0]-1,px_end[0]+1):
            for i_y in range(px_start[1]-1,px_end[1]+1):
                if np.sqrt((i_x-central_px[0])**2+(i_y-central_px[1])**2) < circle_size_px/2.:
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
    if a != 'single':
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


def pixel_annulus_calculation(fitsfile,longitude,latitude,r_in,r_out):
    #longitude and latitude in degree
    #r_in and r_out: inner and outer radius in arcsec
    #give central coordinates, inner and outer radius of annulus and fitsfile and it returns array with the corresponding pixels 
    header = fits.getheader(fitsfile)
    w = WCS(fitsfile)
    delta = abs(header['CDELT1']) #in degree
    circle_in_px = 2*r_in/3600. / delta
    circle_in_px = int(round(circle_in_px))
    circle_out_px = 2*r_out/3600. / delta
    circle_out_px = int(round(circle_out_px))
    if header['NAXIS']==3:
        central_px = w.all_world2pix(longitude,latitude,0,1)
    elif header['NAXIS']==2:
        central_px = w.all_world2pix(longitude,latitude,1)
    else:
        raise Exception('Something wrong with the header.')
    central_px = [int(np.round(central_px[0],decimals=0))-1,int(np.round(central_px[1],decimals=0))-1]
    px_start = [central_px[0]-circle_out_px/2,central_px[1]-circle_out_px/2]
    px_end = [central_px[0]+circle_out_px/2,central_px[1]+circle_out_px/2]
    pixel_array = []
    for i_x in trange(int(np.round(px_start[0],decimals=0)-1),int(np.round(px_end[0],decimals=0)+1)):
        for i_y in range(int(np.round(px_start[1],decimals=0)-1),int(np.round(px_end[1],decimals=0)+1)):
            if (sqrt((i_x-central_px[0])**2+(i_y-central_px[1])**2) > circle_in_px/2.) and (sqrt((i_x-central_px[0])**2+(i_y-central_px[1])**2)) < circle_out_px/2.:
                pixel_array.append((i_x,i_y))
    return pixel_array


#Calculate ellipse pixel:
#TODO
def pixel_ellipse_calculation(central_pixel_x,central_pixel_y,a_pixel,b_pixel):
    header = fits.getheader(fits_file)
    delta = -1*header['CDELT1'] #in degree
    circle_size_px = 2*r/3600. / delta
    circle_size_px = int(round(circle_size_px))
    figure = aplpy.FITSFigure(fits_file, dimensions=[0,1], slices=[0],convention='wells')
    central_px = figure.world2pixel(ra,dec)
    central_px = [int(round(central_px[0]))-1,int(round(central_px[1]))-1]
    close()
    px_start = [central_pixel_x-2*a_pixel,central_pixel_y-2*b_pixel]    #2*a and 2*b just to be sure, that I cover the entire area
    px_end = [central_pixel_x+2*a_pixel,central_pixel_y+2*b_pixel]
    pixel_array = []
    print('pixel_start: '+str(px_start)+'   pixel_end: '+str(px_end))
    for i_x in range(px_start[0],px_end[0]):
        for i_y in range(px_start[1],px_end[1]):
            if (((i_x-central_pixel_x)**2 / float(a_pixel)**2) + ((i_y-central_pixel_y)**2 / float(b_pixel)**2) < 1 ):
                pixel_array.append((i_x,i_y))
            else:
                u = 1
    return pixel_array


#Calculate ellipse pixel annulus:
def pixel_ellipse_annulus_calculation(center_x,center_y,x_in,x_out,y_in,y_out):
    """This function returns both a list of pixels [(y0,x0),(y1,x1),...,(yn,xn)] and a tuple ((y0,y1,...,yn),(x0,x1,...,xn)) corresponding to the ellipse annulus region with central coordinates center_x, center_y, and inner and outer semimajor/semiminor axes x_in, x_out/ y_in, y_out (or the other way around).
    
    Parameters
    ----------
    center_x : int
        x-coordinate of central pixel in pixel units.
    center_y : int
        y-coordinate of central pixel in pixel units.
    x_in : int
        inner semimajor/minor axis of region along x-axis in pixel units.
    x_out : int
    	outer semimajor/minor axis of region along x-axis in pixel units.
    y_in : int
    	inner semimajor/minor axis of region along y-axis in pixel units.
    y_out : int
    	outer semimajor/minor axis of region along y-axis in pixel units.
    Returns
    -------
    pixel_coords : list
        List of pixel coordinates [(y0,x0),(y1,x1),...,(yn,xn)].
    indices_np : tuple
        Tuple of pixel indices ((y0,y1,...,yn),(x0,x1,...,xn)) to index a numpy.ndarray.
    """
    central_px = [int(np.round(center_x,decimals=0)),int(np.round(center_y,decimals=0))]
    px_start = [central_px[0]-x_out,central_px[1]-y_out]
    px_end = [central_px[0]+x_out,central_px[1]+y_out]
    pixel_coords = []
    if x_in==0 and y_in==0:
        for i_x in trange(int(np.round(px_start[0],decimals=0)-1),int(np.round(px_end[0],decimals=0)+1)):
            for i_y in range(int(np.round(px_start[1],decimals=0)-1),int(np.round(px_end[1],decimals=0)+1)):
                if (((i_x - center_x) / float(x_out))**2 + ((i_y - center_y) / float(y_out))**2) < 1:
                    pixel_coords.append((i_y,i_x))
    else:
        for i_x in trange(int(np.round(px_start[0],decimals=0)-1),int(np.round(px_end[0],decimals=0)+1)):
            for i_y in range(int(np.round(px_start[1],decimals=0)-1),int(np.round(px_end[1],decimals=0)+1)):
                if (((i_x - center_x) / float(x_out))**2 + ((i_y - center_y) / float(y_out))**2) < 1 and (((i_x - center_x) / float(x_in))**2 + ((i_y - center_y) / float(y_in))**2) > 1:
                    pixel_coords.append((i_y,i_x))
    indices_np = tuple(zip(*pixel_coords))
    return pixel_coords, indices_np


# get all off diagonal pixels of a map
def get_off_diagonal(name, offset=0):
    """This function returns both a list of pixel coordinate tuples [(y0,x0),(y1,x1),...,(yn,xn)] and a tuple ((y0,y1,...,yn),(x0,x1,...,xn)) corresponding to off diagonal elements of a 2D numpy.ndarray
    
    Parameters
    ----------
    name : str
        name of file.
    offset : int
        y-axis offset from diagonal in units of pixels.
    Returns
    -------
    pixel_coords : list
        List of pixel coordinates [(y0,x0),(y1,x1),...,(yn,xn)].
    indices_np : tuple
        Tuple of pixel indices ((y0,y1,...,yn),(x0,x1,...,xn)) to index a numpy.ndarray.
    """
    data = fits.getdata(name)
    xsize = data.shape[1]
    ysize = data.shape[0]
    m = ysize/float(xsize)
    pixel_coords = []
    for x in range(xsize):
        for y in range(ysize):
            if y > m * x + offset or y < m * x - offset:
                pixel_coords.append((y,x))
    indices_np = tuple(zip(*pixel_coords))
    return pixel_coords, indices_np


# make subcube of ppv cube
def make_subcube(filename, cubedata=None, longitudes=None, latitudes=None, velo_range=None, path_to_output='.', suffix=''):
    import os
    import astropy.units as u
    from astropy.io import fits
    from spectral_cube import SpectralCube, Projection

    if cubedata is None:
        data = fits.open(filename)  # Open the FITS file for reading
        try:
            cube = SpectralCube.read(data)  # Initiate a SpectralCube
        except:
            cube = Projection.from_hdu(data)
        data.close()  # Close the FITS file - we already read it in and don't need it anymore!
    else:
        cube = cubedata
    print(cube)

    # extract coordinates
    if cube.ndim == 3:
        _, b, _ = cube.world[0, :, 0]  #extract world coordinates from cube
        _, _, l = cube.world[0, 0, :]  #extract world coordinates from cube
        v, _, _ = cube.world[:, 0, 0]  #extract velocity world coordinates from cube
    elif cube.ndim == 2:
        b, _ = cube.world[:, 0]  #extract world coordinates from 2d image
        _, l = cube.world[0, :]  #extract world coordinates from 2d image

    # Define desired coordinate range
    physical_types = cube.wcs.world_axis_physical_types
    print('\nMake sure to give coordinates in the frame of the data header: {}\n'.format(physical_types))
    if latitudes is not None:
        lat_range = latitudes * u.deg
        lat_range_idx = sorted([find_nearest(b, lat_range[0]), find_nearest(b, lat_range[1])])
        #lat_range_idx[-1] = lat_range_idx[-1] + 1
    else:
        lat_range_idx = [None, None]
    if longitudes is not None:
        lon_range = longitudes * u.deg
        lon_range_idx = sorted([find_nearest(l, lon_range[0]), find_nearest(l, lon_range[1])])
        #lon_range_idx[-1] = lon_range_idx[-1] + 1
    else:
        lon_range_idx = [None, None]
	
    if cube.ndim == 3:
        if velo_range is not None:
            vel_range = velo_range * u.km/u.s
            vel_range_idx = sorted([find_nearest(v, vel_range[0]), find_nearest(v, vel_range[1])])
            #vel_range_idx[-1] = vel_range_idx[-1] + 1
        else:
            vel_range_idx = [None, None]
    else:
        vel_range_idx = [None, None]

    if all(x is None for x in lat_range_idx+lon_range_idx+vel_range_idx):
        raise ValueError('Have to specify coordinate ranges!')
    
    # Create a sub_cube cut to these coordinates
    if cube.ndim == 3:
        sub_cube = cube[vel_range_idx[0]:vel_range_idx[1], lat_range_idx[0]:lat_range_idx[1], lon_range_idx[0]:lon_range_idx[1]]
    elif cube.ndim == 2:
        sub_cube = cube[lat_range_idx[0]:lat_range_idx[1], lon_range_idx[0]:lon_range_idx[1]]

    print(sub_cube)
    
    filename_wext = os.path.basename(filename)
    filename_base, file_extension = os.path.splitext(filename_wext)
    newname = filename_base + '_subcube' + suffix + '.fits'
    pathname = os.path.join(path_to_output, newname)
    sub_cube.write(pathname, format='fits', overwrite=True)
    print("\n\033[92mSAVED FILE:\033[0m '{}' in '{}'".format(newname,path_to_output))


def jansky_to_kelvin(frequency,theta_1,theta_2): #in units of (GHz,arcsec,arcsec)
    from astropy import constants as const
    c = const.c
    k = const.k_B
    theta_1 = theta_1*2*np.pi/360./3600.
    theta_2 = theta_2*2*np.pi/360./3600.
    S = 2*k*(frequency*10**9)**2/c**2 / (10**(-26)) * 2*np.pi/(2*np.sqrt(2*np.log(2)))**2 * theta_1* theta_2
    temp = 1/S
    print('Jy/K = {:.5e}'.format(S))
    print('K/Jy = {:.5e}'.format(temp))


def convert_jybeam_to_kelvin(filename, path_to_output='.', suffix=''):
    import os
    from astropy.io import fits
    from spectral_cube import SpectralCube, Projection
    import astropy.units as u

    data = fits.open(filename) # Open the FITS file for reading
    try:
        cube = SpectralCube.read(data)  # Initiate a SpectralCube
    except:
        cube = Projection.from_hdu(data[0]) # as a fallback if fits is a 2d image
    data.close()  # Close the FITS file - we already read it in and don't need it anymore!

    cube.allow_huge_operations=True
    cube.unit  

    kcube = cube.to(u.K)  
    kcube.unit 
    
    newname = filename.split('/')[-1].split('.fits')[0] + '_unit_Tb' + suffix + '.fits'
    pathname = os.path.join(path_to_output, newname)
    kcube.write(pathname, format='fits', overwrite=True)
    print("\n\033[92mSAVED FILE:\033[0m '{}' in '{}'".format(newname,path_to_output))


def convert_jtok_huge_dataset(filename, suffix=''):
    import os
    import shutil
    from tqdm import tqdm
    from astropy.io import fits
    from spectral_cube import SpectralCube, Projection
    import astropy.units as u

    filename_wext = os.path.basename(filename)
    filename_base, file_extension = os.path.splitext(filename_wext)
    newname = filename_base + '_unit_Tb' + suffix + '.fits'

    try:
        cube = SpectralCube.read(filename)  # Initiate a SpectralCube
        jtok_factors = cube.beam.jtok(cube.with_spectral_unit(u.GHz).spectral_axis)
        shutil.copy(filename, newname)
        outfh = fits.open(newname, mode='update')
    
        with tqdm(total=len(jtok_factors)) as pbar:
            for index,(slice,factor) in enumerate(zip(cube,jtok_factors)):
                outfh[0].data[index] = slice * factor
                outfh.flush() # write the data to disk
                pbar.update(1)
        outfh[0].header['BUNIT'] = 'K'
        outfh.flush()
        print("\n\033[92mSAVED FILE:\033[0m '{}'".format(newname))
    except:
        data = fits.open(filename) # Open the FITS file for reading
        cube = Projection.from_hdu(data[0]) # as a fallback if fits is a 2d image
        kcube = cube.to(u.K)
        kcube.write(newname, format='fits', overwrite=True)
        print("\n\033[92mSAVED FILE:\033[0m '{}'".format(newname))


def find_common_beam(filenames):
    import radio_beam
    from spectral_cube import SpectralCube, Projection
    from astropy import units as u
    if not isinstance(filenames, list):
        raise TypeError("'filenames' needs to be a list of len=2")
    if not len(filenames)==2:
        raise ValueError("'filenames' needs to be a list of len=2")
    try:
        cube1 = SpectralCube.read(filenames[0])
    except:
        cube1 = Projection.from_hdu(fits.open(filenames[0])[0])
    try:
        cube2 = SpectralCube.read(filenames[1])
    except:
        cube2 = Projection.from_hdu(fits.open(filenames[1])[0])
    common_beam = radio_beam.commonbeam.common_2beams(radio_beam.Beams(beams=[cube1.beam, cube2.beam]))
    return common_beam


def spatial_smooth(filename, beam=None, major=None, minor=None, pa=0, path_to_output='.', suffix=None, allow_huge_operations=False, **kwargs): # smooth image with 2D Gaussian
    import radio_beam
    from spectral_cube import SpectralCube, Projection
    from astropy import units as u
    from astropy import convolution
    try:
        cube = SpectralCube.read(filename)
    except:
        cube = Projection.from_hdu(fits.open(filename)[0])
    if beam is None:
        if major is None or minor is None:
            raise ValueError('Need to specify beam size if no beam is given.')
        beam = radio_beam.Beam(major=major*u.arcsec, minor=minor*u.arcsec, pa=pa*u.deg)
    elif beam is not None:
        beam = beam
        major = int(np.around(beam.major.value * 3600, decimals=0))
    if suffix is not None:
        newname = filename.split('/')[-1].split('.fits')[0] + '_smooth' + str(major) + '_arcsec' + suffix + '.fits'
    else:
        newname = filename.split('/')[-1].split('.fits')[0] + '_smooth_' + str(major) + '_arcsec' + '.fits'
    if allow_huge_operations:
        cube.allow_huge_operations = True
    smoothcube = cube.convolve_to(beam, **kwargs)
    pathname = os.path.join(path_to_output, newname)
    smoothcube.write(pathname, format='fits', overwrite=True)
    print("\n\033[92mSAVED FILE:\033[0m '{}' in '{}'".format(newname,path_to_output))

	
def reproject_cube(filename, template, axes='spatial', path_to_output='.', suffix=None, allow_huge_operations=False):
    from spectral_cube import SpectralCube, Projection
    from astropy.io import fits

    try:
        cube1 = SpectralCube.read(filename)
    except:
        cube1 = Projection.from_hdu(fits.open(filename)[0])	
    try:
        cube2 = SpectralCube.read(template)
    except:
        cube2 = Projection.from_hdu(fits.open(template)[0])
    if axes=='spatial':
        header_template = cube2.header
        header_template['NAXIS3'] = cube1.header['NAXIS3']
        header_template['NAXIS'] = cube1.header['NAXIS']
        header_template['WCSAXES'] = cube1.header['WCSAXES']
        header_template['CRPIX3'] = cube1.header['CRPIX3']
        header_template['CDELT3'] = cube1.header['CDELT3']
        header_template['CUNIT3'] = cube1.header['CUNIT3']
        header_template['CTYPE3'] = cube1.header['CTYPE3']
        header_template['CRVAL3'] = cube1.header['CRVAL3']
    elif axes=='all':
        header_template = cube2.header
    if allow_huge_operations:
        cube1.allow_huge_operations = True
    cube1_reproj = cube1.reproject(header_template)

    if suffix is not None:
        newname = filename.split('/')[-1].split('.fits')[0] + '_reproject' + suffix + '.fits'
    else:
        newname = filename.split('/')[-1].split('.fits')[0] + '_reproject' + '.fits'
    pathname = os.path.join(path_to_output, newname)
    cube1_reproj.write(pathname, overwrite=True)
    print("\n\033[92mSAVED FILE:\033[0m '{}' in '{}'".format(newname,path_to_output))


#TODO
'''
subcube_from_region using spectral cube
'''


def smooth_1d(x,window_len=11,window='hanning'): # smooth spectrum
    """smooth the data using a window with requested size.
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.
    output:
        the smoothed signal
    """
    import numpy

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len<3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s=numpy.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=numpy.ones(window_len,'d')
    else:
        w=eval('numpy.'+window+'(window_len)')

    y=numpy.convolve(w/w.sum(),s,mode='valid')
    return y[(round(window_len/2-1)):-(round(window_len/2))]


def rebin(a, newshape):
    '''Rebin an array to a new shape.
    '''
    assert len(a.shape) == len(newshape)

    slices = [slice(0,old, float(old)/new) for old,new in zip(a.shape,newshape)]
    coordinates = mgrid[slices]
    indices = coordinates.astype('i') #choose the biggest smaller integer index
    return a[tuple(indices)]


#smooth spectrum by averaging adjacent channels
def smooth_ave(spectrum):
    '''spectrum has to be of shape (N, 2)'''
    if spectrum.ndim == 2:
        if spectrum.shape[1] == 2: 
            y = np.nanmean(np.pad(spectrum[:,1].astype(float), ( 0, ((2 - spectrum[:,1].size%2) % 2) ), mode='edge').reshape(-1, 2), axis=1) #pad with edge_values at the end if odd number of channels
            x = np.nanmean(np.pad(spectrum[:,0].astype(float), ( 0, ((2 - spectrum[:,0].size%2) % 2) ), mode='edge').reshape(-1, 2), axis=1)
            return np.column_stack((x,y))
        else:
            raise ValueError('Array needs to be of shape (N, 2) but instead has shape {}'.format(spectrum.shape))
    else:
        raise ValueError('Array needs to be 2-D.')

	
#this works only once since values are repeated to keep the same shape
def smooth_2(spectrum):
    new_spectrum = np.repeat(np.nanmean(np.pad(spectrum.astype(float), ( 0, ((2 - spectrum.size%2) % 2) ), mode='edge').reshape(-1, 2), axis=1), repeats=2) #pad with edge_values at the end if odd number of channels
    if spectrum.size%2 == 0:
        return y
    elif spectrum.size%2 == 1:
        return new_spectrum[:-1]
    else:
        raise ValueError('Something wrong with dimensions of array.')


def calculate_pixelArray_along_filament(filename,filamentdata,halfwidth_pc,distance_of_source):
    #filamentdata in px coordinates ; halfwidth_pc and distance in unit pc
    #find the tangent curve
    intensity = fits.getdata(filename)
    header = fits.getheader(filename)
    pixel_scale = math.tan(np.abs(header['CDELT1']) * math.pi/180) * distance_of_source #pc per px
    halfwidth = int(np.round(halfwidth_pc / pixel_scale,decimals=0))
    filament_coords = np.loadtxt(filamentdata)
    
    x = filament_coords[:,0]
    y = filament_coords[:,1]
    z = filament_coords[:,2]
    
    pixel_list = []

    for n in trange(1,len(filament_coords)-1):
        
        line_perpList = []
        line_perpList2 = []

        x0 = int(x[n])
        y0 = int(y[n])
        z0 = int(z[n])
        r0 = np.array([x0,y0], dtype=float) # point on the filament

        a = x[n+1] - x[n-1]
        b = y[n+1] - y[n-1]
        normal = np.array([a,b], dtype=float)
        #print a, b	
        #plt.plot(normal)
        #equation of normal plane: ax+by=const

        const = np.sum(normal*r0)
					
        # defining lists an array to be used below
        if header['NAXIS'] == 2:
            distance = np.zeros_like(intensity)
            distance2 = np.zeros_like(intensity)

            line_perp = np.zeros_like(intensity) #
            line_perp2 = np.zeros_like(intensity) #
        elif header['NAXIS'] == 3:
            distance = np.zeros_like(intensity[0])
            distance2 = np.zeros_like(intensity[0])

            line_perp = np.zeros_like(intensity[0]) #
            line_perp2 = np.zeros_like(intensity[0]) #
        else:
            raise Exception('FITS file must be 2D or 3D. Instead, file has shape {}'.format(intensity.shape))		
			
		
        # Loop 1: if the slope is negative
        if -float(b)/a > 0:
            npix = halfwidth
            #print(distance.shape)
            if y0+npix < distance.shape[0] and x0+npix < distance.shape[1]:
                npix = npix
                #print('npix unmodified: ', npix)
            elif y0+npix < distance.shape[0] and x0+npix >= distance.shape[1]:
                npix = distance.shape[1]-1- x0
                #print('npix modified: ', npix)
            elif y0+npix >= distance.shape[0] and x0+npix < distance.shape[1]:
                npix = distance.shape[0]-1- y0
                #print('npix modified: ', npix)
            elif y0+npix >= distance.shape[0] and x0+npix >= distance.shape[1]:
                npix_x = distance.shape[1]-1- x0
                npix_y = distance.shape[0]-1- y0
                if npix_x < npix_y:
                    npix = npix_x
                else: 
                    npix = npix_y
				
            for ii in range(y0-npix,y0+1):
                for jj in range(x0-npix,x0+1):
			
                    distance[ii,jj] = ((jj-x0)**2.+(ii-y0)**2.)**0.5 #distance between point (i,j) and filament
                    if (distance[ii,jj] <  npix-1):
                        dist_normal = (np.fabs(a*jj+b*ii-const))/(a**2+b**2)**0.5 #distance between point (i,j) and the normal 
                        #take the point if it is in the vicinity of the normal (distance < 2 pix)
                        if (dist_normal < 1):
                            line_perp[ii,jj] = distance[ii,jj] #storing the nearby points
                            line_perpList.extend((ii,jj,distance[ii,jj]))



            for ii in range(y0,y0+npix):
                for jj in range(x0,x0+npix):
	     
                    distance2[ii,jj] = ((jj-x0)**2.+(ii-y0)**2.)**0.5 
                    if (distance2[ii,jj] <  npix-1):
                        dist_normal2 = (np.fabs(a*jj+b*ii-const))/(np.sum(normal*normal))**0.5 
							
                        if (dist_normal2 < 1): 
                            line_perp2[ii,jj] = distance2[ii,jj] 
                            line_perpList2.extend((ii,jj,distance2[ii,jj]))			
							
	
        #Loop 2_ if the slope is positive
        elif -float(b)/a < 0:
            npix = halfwidth
				
				
            if y0+npix < distance.shape[0] and x0+npix < distance.shape[1]:
                npix = npix
                #print('npix unmodified: ', npix)
            elif y0+npix < distance.shape[0] and x0+npix >= distance.shape[1]:
                npix = distance.shape[1]-1- x0
                #print('npix modified: ', npix)
            elif y0+npix >= distance.shape[0] and x0+npix < distance.shape[1]:
                npix = distance.shape[0]-1- y0
                #print('npix modified: ', npix)
            elif y0+npix >= distance.shape[0] and x0+npix >= distance.shape[1]:
                npix_x = distance.shape[1]-1- x0
                npix_y = distance.shape[0]-1- y0
                if npix_x < npix_y:
                    npix = npix_x
                else: 
                    npix = npix_y
                #print('npix modified: ', npix)	
			
	
            for ii in range(y0,y0+npix):
                for jj in range(x0-npix,x0+1):
                    distance[ii,jj] = ((jj-x0)**2.+(ii-y0)**2.)**0.5
                    if (distance[ii,jj] <  npix-1):
                        dist_normal = (np.fabs(a*jj+b*ii-const))/(np.sum(normal*normal))**0.5
                        if (dist_normal < 1):
                            line_perp[ii,jj] = distance[ii,jj]
                            line_perpList.extend((ii,jj,distance[ii,jj]))


            for ii in range(y0-npix,y0+1):
                for jj in range(x0, x0+npix):
                    distance2[ii,jj] = ((jj-x0)**2.+(ii-y0)**2.)**0.5
                    if (distance2[ii,jj] <  npix-1):
                        dist_normal2 = (np.fabs(a*jj+b*ii-const))/(np.sum(normal*normal))**0.5
                        if (dist_normal2 < 1):
                            line_perp2[ii,jj] = distance2[ii,jj]
                            line_perpList2.extend((ii,jj,distance2[ii,jj]))

        perpendicularLine = np.array(line_perpList).reshape(-1,3)
        perpendicularLine2 = np.array(line_perpList2).reshape(-1,3)
        total_array = np.vstack((perpendicularLine2,perpendicularLine))
    
        pixel_list.append(total_array)
        pixel_array = np.array(pixel_list) 
    print('Filament positions selected over '+str(2*halfwidth_pc)+' pc slices perpendicular to filament')

    return pixel_array


#WORLD COORDS, adapted from filchap (Suri 2018)
def calculateLength_worldcoords(filamentDatafile,distance):
    #distance in pc
    filamentData = np.loadtxt(filamentDatafile)
    length = 0
    for ii in range(len(filamentData)-1):
        x_1 = filamentData[ii,0]
        y_1 = filamentData[ii,1]
        x_2 = filamentData[ii+1,0]
        y_2 = filamentData[ii+1,1]
        delta_x = (x_2 - x_1)**2
        delta_y = (y_2 - y_1)**2
        delta = math.sqrt(delta_x + delta_y)	
        theta = delta*math.pi/180 #convert degree to radians.
        R = math.tan(theta)*distance		
        length = length + R	
    return length

###ORIGINAL script, pixel coords
def calculateLength(filamentDatafile,distance,pix_size):
    #px size in arcsec per px, distance in pc
    filamentData = np.loadtxt(filamentDatafile)
    length = 0
    for ii in range(len(filamentData)-1):
        x_1 = filamentData[ii,0]
        y_1 = filamentData[ii,1]
        x_2 = filamentData[ii+1,0]
        y_2 = filamentData[ii+1,1]
        delta_x = (x_2 - x_1)**2
        delta_y = (y_2 - y_1)**2
        delta = math.sqrt(delta_x + delta_y)	
        theta = (delta*pix_size)/206265 #convert arcsec to radians.
        R = math.tan(theta)*distance		
        length = length + R	
    return length

#with file already read in
def calculateLength_from_array(filamentData,distance,pix_size):
    #px size in arcsec per px, distance in pc
    length = 0
    for ii in range(len(filamentData)-1):
        x_1 = filamentData[ii,0]
        y_1 = filamentData[ii,1]
        x_2 = filamentData[ii+1,0]
        y_2 = filamentData[ii+1,1]
        delta_x = (x_2 - x_1)**2
        delta_y = (y_2 - y_1)**2
        delta = math.sqrt(delta_x + delta_y)	
        theta = (delta*pix_size)/206265 #convert arcsec to radians.
        R = math.tan(theta)*distance		
        length = length + R	
    return length


#calculate Galactocentric radius from distance
def calculate_gal_radius_from_distance(distance,longitude,latitude,R_sun=8.15): # 8.15 kpc from Reid et al. (2019)
    R_hel_x = distance * np.cos(np.radians(latitude)) * np.cos(np.radians(longitude))
    R_hel_y = distance * np.cos(np.radians(latitude)) * np.sin(np.radians(longitude))
    R_hel_z = distance * np.sin(np.radians(latitude))

    R_sun_x = R_sun
    R_sun_y = 0
    R_sun_z = 0

    R_gal_x = R_hel_x - R_sun_x
    R_gal_y = R_hel_y - R_sun_y
    R_gal_z = R_hel_z - R_sun_z

    R_gal_distance = np.sqrt(R_gal_x**2 + R_gal_y**2 + R_gal_z**2)
    return R_gal_distance
