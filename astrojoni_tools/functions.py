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
    indices = []
    for i in range(0,len(pixel_array)):
        x_1,y_1 = pixel_array[i]
        spectrum_i = image[:,y_1,x_1]
        if any([np.isnan(spectrum_i[i]) for i in range(len(spectrum_i))]):
            print('Warning: region contains NaNs!')
            indices.append(i)
            spectrum_add = spectrum_add + 0
            n+=1
        else:
            spectrum_add = spectrum_add + spectrum_i
    spectrum_average = spectrum_add / (len(pixel_array)-n)
    pixel_array_without_nan_values = [i for i in pixel_array if pixel_array[i].index() != indices[i]]
    return spectrum_average, pixel_array_without_nan_values


def calculate_average_value_pixelArray(fitsfile,pixel_array): #nan treatment?
    image = fits.getdata(fitsfile)
    value_add = 0
    n=0
    indices = []
    for i in range(0,len(pixel_array)):
        x_1,y_1 = pixel_array[i]
        value_i = image[y_1,x_1]
        if np.isnan(value_i):
            print('Warning: region contains NaNs!')
            indices.append(i)
            value_add = value_add + 0
            n+=1
        else:
            value_add = value_add + value_i
    if n<(len(pixel_array)/3.):
        value_average = value_add / (len(pixel_array)-n)
    else:
        value_average = np.nan
    pixel_array_without_nan_values = [i for i in pixel_array if pixel_array[i].index() != indices[i]]
    return value_average, pixel_array_without_nan_values
 

def moment_0(fitsfile,velocity_start,velocity_end,path_to_output='.',save_file=True):
    import os
    image = fits.getdata(fitsfile)
    headerm0 = fits.getheader(fitsfile)
    velocity = velocity_axes(fitsfile)
    velocity = velocity.round(decimals=4)
    lower_channel = find_nearest(velocity,velocity_start)
    upper_channel = find_nearest(velocity,velocity_end)
    print('channel-range: '+str(lower_channel)+' - '+str(upper_channel))
    print('velocity-range: '+str(velocity[lower_channel])+' - '+str(velocity[upper_channel]))
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
    newname = fitsfile.split('/')[-1].split('.fits')[0] + '_mom-0_' + str(velocity_start) + '_to_' + str(velocity_end) + 'km-s.fits'
    pathname = os.path.join(path_to_output,newname)
    if save_file is True:
        fits.writeto(pathname, moment_0_map, header=headerm0, overwrite=True)
        print("\n\033[92mSAVED FILE:\033[0m '{}' in '{}'".format(newname,path_to_output))
    else:
        print(newname.split('.fits')[0])
    return moment_0_map

def moment_1(fitsfile,velocity_start,velocity_end,path_to_output='.',save_file=True):
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
    newname = fitsfile.split('/')[-1].split('.fits')[0] + '_mom-1_' + str(velocity_start) + '_to_' + str(velocity_end) + 'km-s.fits'
    pathname = os.path.join(path_to_output,newname)
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


def pixel_circle_calculation(fitsfile,glon,glat,r):
    '''
    This function returns array of pixels corresponding to circle region with central coordinates Glon, Glat, and radius r
    glon: Galactic longitude of central pixel
    glat: Galactic latitude of central pixel
    r: radius of region in arcseconds
    '''
    header = fits.getheader(fitsfile)
    w = WCS(fitsfile)
    delta = abs(header['CDELT1']) #in degree
    pixel_array = []
    if header['NAXIS']==3:
        central_px = w.all_world2pix(glon,glat,0,1)
    elif header['NAXIS']==2:
        central_px = w.all_world2pix(glon,glat,1)
    else:
        raise Exception('Something wrong with the header!')
    central_px = [int(np.round(central_px[0],decimals=0))-1,int(np.round(central_px[1],decimals=0))-1]
    if r is not 'single':
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
    if r is not 'single':
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
#TODO
def pixel_ellipse_annulus_calculation(central_pixel_x,central_pixel_y,a_pixel_out,b_pixel_out,a_pixel_in,b_pixel_in):
    px_start = [int(central_pixel_x-1.1*a_pixel_out),int(central_pixel_y-1.1*b_pixel_out)]    #1.1*a and 1.1*b just to be sure, that I cover the entire area
    px_end = [int(central_pixel_x+1.1*a_pixel_out),int(central_pixel_y+1.1*b_pixel_out)]
    pixel_array = []
    print('pixel_start: '+str(px_start)+'   pixel_end: '+str(px_end))
    if (a_pixel_in == 0):
        for i_x in range(px_start[0],px_end[0]):
            for i_y in range(px_start[1],px_end[1]):
                if ((((i_x-central_pixel_x)**2 / float(a_pixel_out)**2) + ((i_y-central_pixel_y)**2 / float(b_pixel_out)**2) < 1 )):
                    pixel_array.append((i_x,i_y))
                else:
                    u = 1        
    else:
        for i_x in range(px_start[0],px_end[0]):
            for i_y in range(px_start[1],px_end[1]):
                if ((((i_x-central_pixel_x)**2 / float(a_pixel_out)**2) + ((i_y-central_pixel_y)**2 / float(b_pixel_out)**2) < 1 ) and (((i_x-central_pixel_x)**2 / float(a_pixel_in)**2) + ((i_y-central_pixel_y)**2 / float(b_pixel_in)**2) > 1 )):
                    pixel_array.append((i_x,i_y))
                else:
                    u = 1
    return pixel_array


#make subcube of ppv cube
def make_subcube(filename, cubedata=None, longitudes=None, latitudes=None, velo_range=None, path_to_output='.', suffix=None):
    import os
    import astropy.units as u
    from astropy.io import fits
    from spectral_cube import SpectralCube

    if cubedata is None:
        data = fits.open(filename)  # Open the FITS file for reading
        cube = SpectralCube.read(data)  # Initiate a SpectralCube
        data.close()  # Close the FITS file - we already read it in and don't need it anymore!
    else:
        cube = cubedata
    print(cube)

    #extract coordinates
    _, b, _ = cube.world[0, :, 0]  #extract latitude world coordinates from cube
    _, _, l = cube.world[0, 0, :]  #extract longitude world coordinates from cube
    v, _, _ = cube.world[:, 0, 0]  #extract velocity world coordinates from cube

    # Define desired latitude and longitude range
    if latitudes is not None:
        lat_range = latitudes * u.deg
        lat_range_idx = sorted([find_nearest(b, lat_range[0]), find_nearest(b, lat_range[1])])
    else:
        lat_range_idx = [None, None]
    if longitudes is not None:
        lon_range = longitudes * u.deg
        lon_range_idx = sorted([find_nearest(l, lon_range[0]), find_nearest(l, lon_range[1])])
    else:
        lon_range_idx = [None, None]
    if velo_range is not None:
        vel_range = velo_range * u.km/u.s
        vel_range_idx = sorted([find_nearest(v, vel_range[0]), find_nearest(v, vel_range[1])])
    else:
        vel_range_idx = [None, None]
    
    # Create a sub_cube cut to these coordinates
    sub_cube = cube[vel_range_idx[0]:vel_range_idx[1], lat_range_idx[0]:lat_range_idx[1], lon_range_idx[0]:lon_range_idx[1]]

    print(sub_cube)
    
    if suffix is not None:
        newname = filename.split('/')[-1].split('.fits')[0] + '_lon{}to{}_lat{}to{}'.format(longitudes[0], longitudes[1], latitudes[0], latitudes[1]) + suffix + '.fits'
    else:
        newname = filename.split('/')[-1].split('.fits')[0] + '_lon{}to{}_lat{}to{}'.format(longitudes[0], longitudes[1], latitudes[0], latitudes[1]) + '.fits'
    pathname = os.path.join(path_to_output, newname)
    sub_cube.write(pathname, format='fits', overwrite=True)
    print("\n\033[92mSAVED FILE:\033[0m '{}' in '{}'".format(newname,path_to_output))


def jansky_to_kelvin(frequency,theta_1,theta_2): #in units of (GHz,arcsec,arcsec)
    from astropy import constants as const
    c = const.c
    k = const.k_B
    theta_1 = theta_1*2*pi/360./3600.
    theta_2 = theta_2*2*pi/360./3600.
    S = 2*k*(frequency*10**9)**2/c**2 / (10**(-26)) * 2*pi/(2*sqrt(2*log(2)))**2 * theta_1* theta_2
    temp = 1/S
    print('Jy/K = {:.5e}'.format(S))
    print('K/Jy = {:.5e}'.format(temp))


def convert_jybeam_to_kelvin(filename, path_to_output='.', suffix=None):
    import os
    from astropy.io import fits
    from spectral_cube import SpectralCube
    import astropy.units as u

    data = fits.open(filename) # Open the FITS file for reading
    cube = SpectralCube.read(data)  # Initiate a SpectralCube
    data.close()  # Close the FITS file - we already read it in and don't need it anymore!

    cube.allow_huge_operations=True
    cube.unit  

    kcube = cube.to(u.K)  
    kcube.unit 
    
    if suffix is not None:
        newname = filename.split('/')[-1].split('.fits')[0] + '_unit_Tb' + suffix + '.fits'
    else:
        newname = filename.split('/')[-1].split('.fits')[0] + '_unit_Tb' + '.fits'
    pathname = os.path.join(path_to_output, newname)
    kcube.write(pathname, format='fits', overwrite=True)
    print("\n\033[92mSAVED FILE:\033[0m '{}' in '{}'".format(newname,path_to_output))


def spatial_smooth(filename, major=None, minor=None, pa=0, path_to_output='.', suffix=None): # smooth image with 2D Gaussian
    import radio_beam
    from spectral_cube import SpectralCube
    from astropy import units as u

    cube = SpectralCube.read(filename)
    beam = radio_beam.Beam(major=major*u.arcsec, minor=minor*u.arcsec, pa=pa*u.deg)
    smoothcube = cube.convolve_to(beam)
	
    if suffix is not None:
        newname = filename.split('/')[-1].split('.fits')[0] + '_smooth' + str(major) + '_arcsec' + suffix + '.fits'
    else:
        newname = filename.split('/')[-1].split('.fits')[0] + '_smooth_' + str(major) + '_arcsec' + '.fits'
    pathname = os.path.join(path_to_output, newname)
    smoothcube.write(pathname, format='fits', overwrite=True)
    print("\n\033[92mSAVED FILE:\033[0m '{}' in '{}'".format(newname,path_to_output))


def smooth_1d(x,window_len=11,window='hanning'): # smooth spectrum
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
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
