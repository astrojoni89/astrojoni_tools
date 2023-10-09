import os
import shutil
from pathlib import Path
from typing import Union, Optional, List, Tuple

import numpy as np
import radio_beam
import astropy.units as u

from astropy.io import fits
from astropy import constants as const
from astropy.wcs import WCS, WCSSUB_SPECTRAL
from spectral_cube import SpectralCube, Projection
from tqdm import tqdm, trange

from .utils.wcs_utils import sanitize_wcs



def getting_ready(string: str):
    banner = len(string) * '='
    heading = '\n' + banner + '\n' + string + '\n' + banner
    print(heading)


def find_nearest(array: np.ndarray, value: float) -> int:
    """Find the index of an element in an array nearest to a given value.

    Parameters
    ----------
    array : numpy.ndarray
        Input array to index.
    value : float
        Value of the element to find the closest index for.

    Returns
    -------
    idx : int
        Index of the element with value closest to 'value'.
    """
    return np.abs(array-value).argmin()


def md_header_2d(hdr):
    """Get 2D header from FITS file.

    Parameters
    ----------
    fitsfile : path-like object or file-like object
        Path to FITS file to get header from.
    Returns
    -------
    header_2d : :class:`~astropy.io.fits.Header`
        Header object without third axis.
    """
    if isinstance(hdr, (Path, str)):
        header_2d = fits.getheader(hdr)
    elif isinstance(hdr, fits.Header):
        header_2d = hdr
    keys_3d = ['NAXIS3', 'CRPIX3', 'CDELT3', 'CUNIT3', 'CTYPE3', 'CRVAL3']
    for key in keys_3d:
        if key in header_2d.keys():
            del header_2d[key]
    header_2d['NAXIS'] = 2
    header_2d['WCSAXES'] = 2
    return header_2d


def save_fits(filename_basis: Path, data: np.ndarray,
              header, suffix: Optional[str] = '_new',
              path_to_output: Optional[str] = '.', **kwargs):
    """Save FITS file with given filename + suffix at a given location.

    Parameters
    ----------
    filename_basis : Path
        Path to FITS file to use as a basis for the new filename.
    data : numpy.ndarray
        Data to save under the new filename.
    header : :class:`~astropy.io.fits.Header`
        Header object that is associated with 'data'.
	    If None, a header of the appropriate type is created for the supplied data.
    suffix : str, optional
        Suffix to append to new filename. Default is '_new'.
    path_to_output : str
        Path to output where FITS will be saved.
    **kwargs
        Additional arguments are passed to
        :func:`~astropy.io.fits.writeto()`.
    """
    filename_wext = os.path.basename(filename_basis)
    filename_base, file_extension = os.path.splitext(filename_wext)
    outname = filename_base + suffix + '.fits'

    if not os.path.exists(path_to_output):
        os.makedirs(path_to_output)
    outfile = os.path.join(path_to_output, outname)

    fits.writeto(outfile, data, header=header, overwrite=True, **kwargs)
    print("\n\033[92mSAVED FILE:\033[0m '{}' in '{}'".format(outname, path_to_output))


def velocity_axes(name: Path) -> np.ndarray:
    """Get velocity axis from FITS file.

    Parameters
    ----------
    name : Path
        Path to FITS file to get velocity axis from.

    Returns
    -------
    velocity : numpy.ndarray
        Array of velocity axis.
    """
    header = fits.getheader(name)
    n = header['NAXIS3'] #number of channels on spectral axis
    velocity = (header['CRVAL3'] - header['CRPIX3'] * header['CDELT3']) + (np.arange(n)+1) * header['CDELT3']
    velocity = velocity / 1000
    return velocity


def latitude_axes(name: Path) -> np.ndarray:
    """Get latitude axis from FITS file.
    
    Parameters
    ----------
    name : Path
        Path to FITS file to get latitude axis from.

    Returns
    -------
    velocity : numpy.ndarray
        Array of latitude axis.
    """
    header = fits.getheader(name)
    n = header['NAXIS2'] #number of pixels along latitude axis
    return (header['CRVAL2'] - header['CRPIX2'] * header['CDELT2']) + (np.arange(n)+1) * header['CDELT2']


def longitude_axes(name: Path) -> np.ndarray:
    """Get longitude axis from FITS file.
    
    Parameters
    ----------
    name : Path
        Path to FITS file to get longitude axis from.

    Returns
    -------
    velocity : numpy.ndarray
        Array of longitude axis.
    """
    header = fits.getheader(name)
    n = header['NAXIS1'] #number of pixels along longitude axis
    return (header['CRVAL1'] - header['CRPIX1'] * header['CDELT1']) + (np.arange(n)+1) * header['CDELT1']


def world_to_pixel(fitsfile: Path, longitude: float,
                   latitude: float, velocity: Optional[float] = 0.) -> np.ndarray:
    """Convert world coordinates to pixel coordinates from a FITS file.
    
    Parameters
    ----------
    fitsfile : Path
        Path to FITS file to get coordinates from.
    longitude : float
        World coordinate value along the x-axis of the FITS file, e.g. longitude.
    latitude : float
        World coordinate value along the y-axis of the FITS file, e.g. latitude.
    velocity : float, optional
        Velocity value to convert (default is 0.).

    Returns
    -------
    result : numpy.ndarray
        Returns the pixel coordinates. If the input was a single array and origin,
	a single array is returned, otherwise a tuple of arrays is returned.
    """
    w = WCS(fitsfile)
    if w.wcs.naxis == 3:
        return w.all_world2pix(longitude, latitude, velocity, 1)
    elif w.wcs.naxis == 2:
        return w.all_world2pix(longitude, latitude, 1)
    else:
        raise ValueError('Something wrong with the header.')


def pixel_to_world(fitsfile: Path, x: float,
                   y: float, ch: Optional[float] = 0.):
    """Convert pixel coordinates to world coordinates from a FITS file.
    
    Parameters
    ----------
    fitsfile : str
        Path to FITS file to get coordinates from.
    x : float
        Pixel coordinate on the x-axis of the FITS file.
    y : float
        Pixel coordinate on the y-axis of the FITS file.
    ch : float, optional
        Velocity channel to convert (default is 0.).
    Returns
    -------
    result : numpy.ndarray
        Returns the world coordinates. If the input was a single array and origin,
	a single array is returned, otherwise a tuple of arrays is returned.
    """
    w = WCS(fitsfile)
    if w.wcs.naxis == 3:
        return w.all_pix2world(x, y, ch, 1)
    elif w.wcs.naxis == 2:
        return w.all_pix2world(x, y, 1)
    else:
        raise ValueError('Something wrong with the header.')


def calculate_spectrum(fitsfile: Path, pixel_array: List) -> Tuple[np.ndarray, List]:
    """Calculate an average spectrum given a p-p-v FITS cube and pixel coordinates.
    If NaN values are present at specific coordinates, these coordinates will be ignored. 
    
    Parameters
    ----------
    fitsfile : Path
        Path to FITS file to get average spectrum from.
    pixel_array : List
        List of tuples containing pixel coordinates [(y0,x0),(y1,x1),...]
	    over which to average.

    Returns
    -------
    spectrum_average : numpy.ndarray
        Averaged spectrum.
    pixel_list_without_nan_values : List
        List of tuples containing pixel coordinates [(y0,x0),(y1,x1),...]
	at which data contain finite values.
    """
    header = fits.getheader(fitsfile)
    image = fits.getdata(fitsfile)
    number_of_channels = header['NAXIS3']
    spectrum_add = np.zeros(number_of_channels)
    n=0
    idxs = []
    for i in trange(0,len(pixel_array)):
        y_1,x_1 = pixel_array[i]
        spectrum_i = image[:,y_1,x_1]
        if np.any([np.isnan(spectrum_i[k]) for k in range(len(spectrum_i))]):
            print('Warning: region contains NaNs!')
            idxs.append(i)
            n+=1
        else:
            spectrum_add = spectrum_add + spectrum_i
    spectrum_average = spectrum_add / (len(pixel_array)-n)
    temp_array = np.delete(pixel_array, idxs, axis=0)
    pixel_list_without_nan_values = list(map(tuple, temp_array))
    return spectrum_average, pixel_list_without_nan_values


def calculate_average_value_of_map(fitsfile: Path, pixel_array: List) -> Tuple[float, List]:
    """Calculate an average value given a 2-D FITS map and pixel coordinates.
    If NaN values are present at specific coordinates, these coordinates will be ignored. 
    
    Parameters
    ----------
    fitsfile : path-like object or file-like object
        Path to FITS file to get average value from.
    pixel_array : list
        List of tuples containing pixel coordinates [(y0,x0),(y1,x1),...]
	    over which to average.

    Returns
    -------
    value_average : float
        Averaged value.
    pixel_list_without_nan_values : list
        List of tuples containing pixel coordinates [(y0,x0),(y1,x1),...]
	at which data contain finite values.
    """
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
            n+=1
        else:
            value_add = value_add + value_i
    value_average = value_add / (len(pixel_array)-n)
    temp_array = np.delete(pixel_array, idxs, axis=0)
    pixel_list_without_nan_values = list(map(tuple, temp_array))
    return value_average, pixel_list_without_nan_values


def moment_0(filename: Path, velocity_start: float = None, velocity_end: float = None,
             noise: Optional[float] = None,
             path_to_output: Optional[str] = '.',
             save_file: Optional[bool] = True,
             output_noise: Optional[bool] = True,
             suffix: Optional[str] = ''):
    """Calculate the zeroth moment of a p-p-v FITS cube.
    
    Parameters
    ----------
    filename : Path
        Path to FITS file.
    velocity_start : float
        Start velocity from which to integrate.
    velocity_end : float
        End velocity up to which data are integrated.
    noise : float, optional
        Noise value of p-p-v data. If noise is given,
	    noise of the zeroth moment will be calculated.
    path_to_output : str, optional
        Path to output where moment 0 map will be saved.
	    By default, the subcube will be saved in the working directory.
    save_file : bool
        Whether moment 0 map should be saved as a file. Default is True.
    output_noise : bool
        Whether moment 0 noise should be stored in a .txt file. Default is True.
    suffix : str
        Suffix of moment 0 filename.
    Returns
    -------
    moment_0_map : numpy.ndarray
        Zeroth moment map.
    """
    getting_ready('Computing moment 0')
    image = fits.getdata(filename)
    headerm0 = fits.getheader(filename)
    header_save = md_header_2d(filename)
    velocity = velocity_axes(filename)
    velocity = velocity.round(decimals=4)
    if velocity_start is not None:
        if velocity_end is not None:
            velocity_low = min(velocity_start,velocity_end)
	else:
            velocity_low = velocity_start
        lower_channel = find_nearest(velocity,velocity_low)
    else:
        lower_channel = 0
    if velocity_end is not None:
        if velocity_start is not None:
            velocity_up = max(velocity_start,velocity_end)
	else:
            velocity_up = velocity_end
        upper_channel = find_nearest(velocity,velocity_up)
    else:
        upper_channel = -1
    print('channel-range: '+str(lower_channel)+' - '+str(upper_channel))
    print('velocity-range: '+str(velocity[lower_channel])+' - '+str(velocity[upper_channel]))
    moment_0_map = np.zeros((headerm0['NAXIS2'],headerm0['NAXIS1']))
    if headerm0['NAXIS']==4:
        for i in trange(lower_channel,upper_channel+1,1):
            moment_0_map = moment_0_map + image[0,i,:,:]
    elif headerm0['NAXIS']==3:
        for i in trange(lower_channel,upper_channel+1,1):
            moment_0_map = moment_0_map + image[i,:,:]
    else:
        print('Something wrong with the header.')
    moment_0_map = moment_0_map * headerm0['CDELT3']/1000
    headerm0['BUNIT'] = headerm0['BUNIT']+'.KM/S'

    filename_wext = os.path.basename(filename)
    filename_base, file_extension = os.path.splitext(filename_wext)
    newname = filename_base + f'_mom-0_{np.around(velocity[lower_channel],decimals=1):05.1f}_to_{np.around(velocity[upper_channel],decimals=1):05.1f}km-s' + suffix + '.fits'
    pathname = os.path.join(path_to_output, newname)

    if save_file is True:
        fits.writeto(pathname, moment_0_map, header=header_save, overwrite=True)
        print("\n\033[92mSAVED FILE:\033[0m '{}' in '{}'".format(newname,path_to_output))
    else:
        print(newname.split('.fits')[0])
    if noise is not None:
        num_ch = int(upper_channel - lower_channel)
        moment_0_noise = noise * np.sqrt(num_ch) * headerm0['CDELT3']/1000
        print('Moment 0 noise at 1 sigma: {}'.format(moment_0_noise))
        if output_noise is True:
            noise_array = np.array([1, moment_0_noise])
            name_noise_file = filename_base + '_mom-0_noise.txt'
            path_noise_file = os.path.join(path_to_output, name_noise_file)
            np.savetxt(path_noise_file, noise_array)
    return moment_0_map


def moment_1(filename: Path,
             velocity_start: float,
             velocity_end: float,
             path_to_output: Optional[str] = '.',
             save_file: Optional[bool] = True,
             suffix: Optional[str] = ''):
    """Calculate the intensity-weighted mean velocity (first moment) of a p-p-v FITS cube.
    
    Parameters
    ----------
    filename : Path
        Path to FITS file.
    velocity_start : float
        Start velocity from which to integrate.
    velocity_end : float
        End velocity up to which data are integrated.
    path_to_output : str, optional
        Path to output where moment 1 map will be saved.
	    By default, the subcube will be saved in the working directory.
    save_file : bool
        Whether moment 1 map should be saved as a file. Default is True.
    suffix : str, optional
        Suffix of moment 1 filename.
    Returns
    -------
    moment_1_map : numpy.ndarray
        First moment map.
    """
    getting_ready('Computing moment 1')
    image = fits.getdata(filename)
    headerm1 = fits.getheader(filename)
    header_save = md_header_2d(filename)
    velocity = velocity_axes(filename)
    velocity = velocity.round(decimals=4)
    lower_channel = find_nearest(velocity, velocity_start)
    upper_channel = find_nearest(velocity, velocity_end)
    print('channel-range: '+str(lower_channel)+' - '+str(upper_channel))
    print('velocity-range: '+str(velocity[lower_channel])+' - '+str(velocity[upper_channel]))
    moment_1_map = np.zeros((headerm1['NAXIS2'], headerm1['NAXIS1']))
    if headerm1['NAXIS'] == 4:
        for i in trange(lower_channel, upper_channel+1, 1):
            moment_1_map = moment_1_map + image[0, i, :, :] * velocity[i]
    elif headerm1['NAXIS'] == 3:
        for i in trange(lower_channel, upper_channel+1, 1):
            moment_1_map = moment_1_map + image[i, :, :] * velocity[i]
    else:
        print('Something wrong with the header.')
    moment_0_map = moment_0(filename, velocity_start, velocity_end, save_file=False)
    moment_1_map = (moment_1_map * headerm1['CDELT3']/1000) / moment_0_map
    headerm1['BUNIT'] = 'KM/S'

    filename_wext = os.path.basename(filename)
    filename_base, _ = os.path.splitext(filename_wext)
    newname = filename_base + '_mom-1_' + str(np.around(velocity[lower_channel],decimals=1)) + '_to_' + str(np.around(velocity[upper_channel],decimals=1)) + 'km-s' + suffix + '.fits'
    pathname = os.path.join(path_to_output, newname)

    if save_file is True:
        fits.writeto(pathname, moment_1_map, header=header_save, overwrite=True)
        print("\n\033[92mSAVED FILE:\033[0m '{}' in '{}'".format(newname,path_to_output))
    else:
        print(newname.split('.fits')[0])
    return moment_1_map


def add_up_channels(fitsfile: Path, velocity_start: float,
                    velocity_end: float,
                    path_to_output: Optional[str] = '.',
                    save_file: Optional[bool] = True,
                    suffix: Optional[str] = '') -> np.ndarray:
    """Add up slices of a p-p-v FITS cube along the velocity axis.
    
    Parameters
    ----------
    fitsfile : Path
        Path to FITS file.
    velocity_start : float
        Start velocity from which to sum up channels.
    velocity_end : float
        End velocity up to which data summed up.
    path_to_output : str, optional
        Path to output where moment 1 map will be saved.
	    By default, the subcube will be saved in the working directory.
    save_file : bool, optional
        Whether moment 1 map should be saved as a file. Default is True.
    suffix : str, optional
        Suffix of moment 1 filename.
    Returns
    -------
    map_sum : numpy.ndarray
        Map of summed up channels.
    """
    getting_ready('Adding up channels')
    image = fits.getdata(fitsfile)
    header = fits.getheader(fitsfile)
    velocity = velocity_axes(fitsfile)
    velocity = velocity.round(decimals=4)
    lower_channel = find_nearest(velocity, velocity_start)
    upper_channel = find_nearest(velocity, velocity_end)
    print('channel-range: '+str(lower_channel)+' - '+str(upper_channel))
    print('velocity-range: '+str(velocity[lower_channel])+' - '+str(velocity[upper_channel]))
    map_sum = np.zeros((header['NAXIS2'], header['NAXIS1']))
    if header['NAXIS'] == 4:
        for i in range(lower_channel, upper_channel+1, 1):
            map_sum += image[0, i, :, :]
    elif header['NAXIS'] == 3:
        for i in range(lower_channel, upper_channel+1, 1):
            map_sum += image[i, :, :]
    else:
        print('Something wrong with the header.')

    filename_wext = os.path.basename(fitsfile)
    filename_base, _ = os.path.splitext(filename_wext)
    newname = filename_base + '_sum_' + str(np.around(velocity[lower_channel],decimals=1)) + '_to_' + str(np.around(velocity[upper_channel],decimals=1)) + 'km-s' + suffix + '.fits'
    pathname = os.path.join(path_to_output, newname)

    if save_file is True:
        fits.writeto(pathname, map_sum, header=header, overwrite=True)
        print("\n\033[92mSAVED FILE:\033[0m '{}' in '{}'".format(newname,path_to_output))
    else:
        print(newname.split('.fits')[0])
    return map_sum


def channel_averaged(fitsfile: Path, velocity_start: float,
                     velocity_end: float,
                     path_to_output: Optional[str] = '.',
                     save_file: Optional[bool] = True,
                     suffix: Optional[str] = '') -> np.ndarray:
    """Average slices of a p-p-v FITS cube along the velocity axis.
    
    Parameters
    ----------
    fitsfile : path-like object or file-like object
        Path to FITS file.
    velocity_start : float
        Start velocity from which to sum up channels.
    velocity_end : float
        End velocity up to which data summed up.
    path_to_output : str, optional
        Path to output where moment 1 map will be saved.
	    By default, the subcube will be saved in the working directory.
    save_file : bool, optional
        Whether moment 1 map should be saved as a file. Default is True.
    suffix : str, optional
        Suffix of moment 1 filename.

    Returns
    -------
    average_map : numpy.ndarray
        Map of averaged channels.
    """
    getting_ready('Averaging channels')
    image = fits.getdata(fitsfile)
    header = fits.getheader(fitsfile)
    velocity = velocity_axes(fitsfile)
    velocity = velocity.round(decimals=4)
    lower_channel = find_nearest(velocity, velocity_start)
    upper_channel = find_nearest(velocity, velocity_end)
    print('channel-range: '+str(lower_channel)+' - '+str(upper_channel))
    print('velocity-range: '+str(velocity[lower_channel])+' - '+str(velocity[upper_channel]))
    average_map = np.zeros((header['NAXIS2'],header['NAXIS1']))

    n = 0
    if header['NAXIS'] == 4:
        for i in range(lower_channel,upper_channel+1,1):
            if np.all(np.isnan(image[0, i, :, :])):
                continue
            else:
                average_map += image[0, i, :, :]
                n += 1
    elif header['NAXIS'] == 3:
        for i in range(lower_channel, upper_channel+1, 1):
            if np.all(np.isnan(image[i, :, :])):
                continue
            else:
                average_map += image[i, :, :]
                n += 1
    else:
        print('Something wrong with the header...')
    average_map = average_map / n

    filename_wext = os.path.basename(fitsfile)
    filename_base, _ = os.path.splitext(filename_wext)
    newname = filename_base + '_averaged-channel_' + str(np.around(velocity[lower_channel],decimals=1)) + '_to_' + str(np.around(velocity[upper_channel],decimals=1)) + 'km-s' + suffix + '.fits'
    pathname = os.path.join(path_to_output, newname)

    if save_file is True:
        fits.writeto(pathname, average_map, header=header, overwrite=True)
        print("\n\033[92mSAVED FILE:\033[0m '{}' in '{}'".format(newname, path_to_output))
    else:
        print(newname.split('.fits')[0])
    return average_map


def pixel_circle_calculation(fitsfile: Path,
                             xcoord: Union[np.ndarray, float],
                             ycoord: Union[np.ndarray, float],
                             r: float) -> Tuple[List, Tuple]:
    """Extract both a list of pixels [(y0,x0),(y1,x1),...] and a tuple ((y0,y1,...),(x0,x1,...))
    corresponding to the circle region with central coordinates xcoord, ycoord, and radius r.
    
    Parameters
    ----------
    fitsfile : Path
        Path to FITS file.
    xcoord : numpy.ndarray | float
        x-coordinate of central pixel in units given in the header.
    ycoord : numpy.ndarray | float
        y-coordinate of central pixel in units given in the header.
    r : float
        Radius of region in units of arcseconds.
    Returns
    -------
    pixel_coords : List
        List of pixel coordinates [(y0,x0),(y1,x1),...].
    indices_np : Tuple
        Tuple of pixel indices ((y0,y1,...),(x0,x1,...)) to index a numpy.ndarray.
    """
    header = fits.getheader(fitsfile)
    w = WCS(fitsfile)
    delta = abs(header['CDELT1']) #in degree
    pixel_coords = []
    if header['NAXIS'] == 3:
        central_px = w.all_world2pix(xcoord, ycoord, 0, 1)
    elif header['NAXIS'] == 2:
        central_px = w.all_world2pix(xcoord, ycoord, 1)
    else:
        raise Exception('Something wrong with the header!')
    central_px = [int(np.round(central_px[0],decimals=0))-1, int(np.round(central_px[1],decimals=0))-1]
    if r != 'single':
        circle_size_px = 2*r/3600. / delta
        circle_size_px = int(round(circle_size_px))
        px_start = [central_px[0]-circle_size_px/2, central_px[1]-circle_size_px/2]
        px_end = [central_px[0]+circle_size_px/2, central_px[1]+circle_size_px/2]
        px_start = [int(np.round(px_start[0], decimals=0)), int(np.round(px_start[1], decimals=0))]
        px_end = [int(np.round(px_end[0], decimals=0)), int(np.round(px_end[1], decimals=0))]
        for i_x in trange(px_start[0]-1, px_end[0]+1):
            for i_y in range(px_start[1]-1, px_end[1]+1):
                if np.sqrt((i_x-central_px[0])**2+(i_y-central_px[1])**2) < circle_size_px/2.:
                    pixel_coords.append((i_y, i_x))
    else:
        pixel_coords.append((central_px[1], central_px[0]))
    
    indices_np = tuple(zip(*pixel_coords))
    return pixel_coords, indices_np


def pixel_circle_calculation_px(fitsfile: Path,
                                x: Union[np.ndarray, float],
                                y: Union[np.ndarray, float],
                                r: float) -> Tuple[List, Tuple]:
    """Extract both a list of pixels [(y0,x0),(y1,x1),...] and a tuple ((y0,y1,...),(x0,x1,...))
    corresponding to the circle region with central pixels x, y, and radius r.
    
    Parameters
    ----------
    fitsfile : Path
        Path to FITS file.
    x : numpy.ndarray or float
        Central x pixel.
    y : numpy.ndarray or float
        Central y pixel.
    r : float
        Radius of region in units of arcseconds.
    Returns
    -------
    pixel_array : List
        List of pixel coordinates [(y0,x0),(y1,x1),...].
    indices_np : Tuple
        Tuple of pixel indices ((y0,y1,...),(x0,x1,...)) to index a numpy.ndarray.
    """
    header = fits.getheader(fitsfile)
    delta = abs(header['CDELT1']) #in degree
    pixel_array = []
    central_px = [x, y]
    if r != 'single':
        circle_size_px = 2*r/3600. / delta
        circle_size_px = int(round(circle_size_px))
        px_start = [central_px[0]-circle_size_px/2, central_px[1]-circle_size_px/2]
        px_end = [central_px[0]+circle_size_px/2, central_px[1]+circle_size_px/2]
        px_start = [int(np.round(px_start[0], decimals=0)), int(np.round(px_start[1], decimals=0))]
        px_end = [int(np.round(px_end[0], decimals=0)), int(np.round(px_end[1], decimals=0))]
        for i_x in range(px_start[0]-1, px_end[0]+1):
            for i_y in range(px_start[1]-1, px_end[1]+1):
                if np.sqrt((i_x-central_px[0])**2+(i_y-central_px[1])**2) < circle_size_px/2.:
                    pixel_array.append((i_y, i_x))
    else:
        pixel_array.append((central_px[0], central_px[1]))
    
    indices_np = tuple(zip(*pixel_array))
    return pixel_array, indices_np


def pixel_box_calculation(fitsfile: Path, longitude: float,
                          latitude: float, a: float, b: float):
    """
    Parameters
    ----------
    longitude : float
        [deg]
    latitude : float
        [deg]
    a,b: total size of longitude,latitude box in arcsec

    Returns
    -------
    give central coordinates, size of box and fitsfile and it returns array with the corresponding pixels 
    """
    header = fits.getheader(fitsfile)
    w = WCS(fitsfile)
    delta = abs(header['CDELT1']) #in degree
    pixel_array = []
    if header['NAXIS'] == 3:
        central_px = w.all_world2pix(longitude, latitude, 0, 1)
    elif header['NAXIS'] == 2:
        central_px = w.all_world2pix(longitude, latitude, 1)
    else:
        raise Exception('Something wrong with the header.')
    central_px = [int(np.round(central_px[0], decimals=0))-1, int(np.round(central_px[1], decimals=0))-1]
    if a != 'single':
        box_size_px_a = a/3600. / delta
        box_size_px_a = int(round(box_size_px_a))
        box_size_px_b = b/3600. / delta
        box_size_px_b = int(round(box_size_px_b))
        px_start = [central_px[0]-box_size_px_a/2, central_px[1]-box_size_px_b/2]
        px_end = [central_px[0]+box_size_px_a/2, central_px[1]+box_size_px_b/2]
        px_start = [int(np.round(px_start[0], decimals=0)), int(np.round(px_start[1], decimals=0))]
        px_end = [int(np.round(px_end[0], decimals=0)), int(np.round(px_end[1], decimals=0))]
        for i_x in trange(px_start[0], px_end[0]):
            for i_y in range(px_start[1], px_end[1]):
                pixel_array.append((i_x, i_y))
    else:
        pixel_array.append((central_px[0], central_px[1]))
    return pixel_array


def pixel_annulus_calculation(fitsfile, longitude, latitude, r_in, r_out):
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
    """Extract both a list of pixels [(y0,x0),(y1,x1),...,(yn,xn)] and a tuple ((y0,y1,...,yn),(x0,x1,...,xn))
    corresponding to the ellipse annulus region with central coordinates center_x, center_y,
    and inner and outer semimajor/semiminor axes x_in, x_out/ y_in, y_out (or the other way around).
    
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


def get_off_diagonal(name, offset=0):
    """Extract both a list of pixel coordinate tuples [(y0,x0),(y1,x1),...,(yn,xn)] and a tuple ((y0,y1,...,yn),(x0,x1,...,xn))
    corresponding to off diagonal elements of a 2D numpy.ndarray
    
    Parameters
    ----------
    name : str
        Name of file.
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


def make_subcube(filename, cubedata=None, longitudes=None, latitudes=None, velo_range=None, path_to_output='.', suffix='', verbose=True):
    """Create a subcube from an existing SpectralCube (or 2D Projection) given some coordinate ranges.
    
    Parameters
    ----------
    filename : str
        Path to file to create a subcube from.
    cubedata : None or :class:`~spectralcube.SpectralCube` or :class:`~spectralcube.Projection`, optional
        SpectralCube of data if data is already read in. This option avoids reading data into memory again.
    longitudes : list
        List of coordinate range of first world coordinate axis.
    latitudes : list
        List of coordinate range of second world coordinate axis.
    velo_range : list
        List of velocity range of spectral coordinate axis.
    path_to_output : str, optional
        Path to output where subcube will be saved. By default, the subcube will be saved in the working directory.
    suffix : str, optional
        Suffix that is appended to output filename.
    verbose : bool, optional
        Option to print subcube info and save messages. Default is True.
    """
    getting_ready('Making subcube')
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

    if verbose:
        print(sub_cube)
    
    filename_wext = os.path.basename(filename)
    filename_base, file_extension = os.path.splitext(filename_wext)
    newname = filename_base + '_subcube' + suffix + '.fits'
    pathname = os.path.join(path_to_output, newname)
    sub_cube.write(pathname, format='fits', overwrite=True)
    if verbose:
        print("\n\033[92mSAVED FILE:\033[0m '{}' in '{}'".format(newname,path_to_output))


def make_lv(filename, mode='avg', weights=None, noise=None, path_to_output='.', suffix=''):
    """Create a longitude-velocity map from a datacube.
    
    Parameters
    ----------
    filename : str
        Path to file to create a longitude-velocity map from.
    mode : str
        Mode to compute the value of collapsed latitude axis.
        'avg' is the arithmetic mean along the latitude axis.
        'max' gives the maximum value along the latitude axis.
        'sum' gives thes sum along the latitude axis.
        'weighted' gives a weighted arithmetic mean along the latitude axis.
    weights : str, optional
        Path to file containing weights. Needs to be of same shape as data of filename.
    noise : float
        Noise to use as a threshold. Every pixel value below the noise will be masked.
    path_to_output : str, optional
        Path to output where subcube will be saved. By default, the subcube will be saved in the working directory.
    suffix : str, optional
        Suffix that is appended to output filename.
    """
    getting_ready('Making l-v map')
    modes = {'avg': 0, 'max': 1, 'sum': 2, 'weighted': 3}
    if mode not in modes.keys():
        raise KeyError("\nUnknown mode. Mode needs to be one of the following: {}.".format(modes.keys()))
    data = fits.getdata(filename)
    header = fits.getheader(filename)
    if mode=='weighted':
        if weights is None:
            print("\nNo weights have been given! Will compute arithmetic mean with equal weights instead.")
            weight = np.ones_like(data)
        else:
            weight = fits.getdata(weights)
            if weight.shape != data.shape:
                raise ValueError("\nShape of weights need to be equal to shape of data!")
    
    pv_array = np.empty((header['NAXIS3'],header['NAXIS1']))
    wcs = WCS(header)
    if wcs is not None:
        wcs = sanitize_wcs(wcs)
    wcs_slice = wcs.sub([0, WCSSUB_SPECTRAL])
    try:
        wcs_slice.wcs.pc[1,0] = wcs_slice.wcs.pc[0,1] = 0
    except AttributeError:
        pass

    # Set spatial parameters
    wcs_slice.wcs.crpix[0] = header['CRPIX1']
    wcs_slice.wcs.cdelt[0] = header['CDELT1']
    wcs_slice.wcs.crval[0] = header['CRVAL1']
    wcs_slice.wcs.ctype[0] = 'GLON'
    wcs_slice.wcs.cunit[0] = 'deg'

    new_header = wcs_slice.to_header()
    new_header['BUNIT'] = header['BUNIT']
    new_header.comments['CUNIT1'] = header.comments['CUNIT1']
    new_header.comments['CTYPE1'] = header.comments['CTYPE1']
    new_header.comments['CRPIX1'] = header.comments['CRPIX1']
    new_header.comments['CDELT1'] = header.comments['CDELT1']
    new_header.comments['CRVAL1'] = header.comments['CRVAL1']
    if mode=='avg':
        new_header.comments['BUNIT'] = 'Avg. brightness (pixel) unit'
    elif mode=='max':
        new_header.comments['BUNIT'] = 'Maximum brightness (pixel) unit'
    elif mode=='sum':
        new_header.comments['BUNIT'] = 'Sum of brightness pixels'
    elif mode=='weighted':
        new_header.comments['BUNIT'] = 'Weighted mean of brightness (pixel) unit'
    else:
        pass

    if noise is not None:
        noisemask = np.where(data < noise)
        weight[noisemask] = np.nan
        data[noisemask] = np.nan

    for vel in trange(data.shape[0]):
        for lon in range(data.shape[2]):
            if mode=='avg':
                avg = np.nanmean(data[vel,:,lon])
            elif mode=='max':
                avg = np.nanmax(data[vel,:,lon])
            elif mode=='sum':
                avg = np.nansum(data[vel,:,lon])
            elif mode=='weighted':
                norm_weight = np.absolute(weight[vel,:,lon]) / np.nansum(np.absolute(weight[vel,:,lon]))
                avg = np.nansum(norm_weight * data[vel,:,lon])
            pv_array[vel,lon] = avg

    filename_wext = os.path.basename(filename)
    filename_base, file_extension = os.path.splitext(filename_wext)
    mode_suffix = '_longitude_velocity_{}'.format(mode)
    newname = filename_base + mode_suffix + suffix + '.fits'
    pathname = os.path.join(path_to_output, newname)

    fits.writeto(pathname, pv_array, header=new_header, overwrite=True)
    print("\n\033[92mSAVED FILE:\033[0m '{}' in '{}'".format(newname,path_to_output))


def jansky_to_kelvin(frequency,theta_1,theta_2): #in units of (GHz,arcsec,arcsec)
    c = const.c
    k = const.k_B
    theta_1 = theta_1*2*np.pi/360./3600.
    theta_2 = theta_2*2*np.pi/360./3600.
    S = 2*k*(frequency*10**9)**2/c**2 / (10**(-26)) * 2*np.pi/(2*np.sqrt(2*np.log(2)))**2 * theta_1* theta_2
    temp = 1/S
    print('Jy/K = {:.5e}'.format(S))
    print('K/Jy = {:.5e}'.format(temp))


def convert_jybeam_to_kelvin(filename, path_to_output='.', suffix=''):
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
    
    filename_wext = os.path.basename(filename)
    filename_base, file_extension = os.path.splitext(filename_wext)
    newname = filename_base + '_unit_Tb' + suffix + '.fits'
    pathname = os.path.join(path_to_output, newname)
    kcube.write(pathname, format='fits', overwrite=True)
    print("\n\033[92mSAVED FILE:\033[0m '{}' in '{}'".format(newname,path_to_output))


def convert_jtok_huge_dataset(filename, suffix=''):
    filename_wext = os.path.basename(filename)
    filename_base, file_extension = os.path.splitext(filename_wext)
    newname = filename_base + '_unit_Tb' + suffix + '.fits'

    data = fits.open(filename) # Open the FITS file for reading
    try:
        cube = SpectralCube.read(data)  # Initiate a SpectralCube
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
        cube = Projection.from_hdu(data) # as a fallback if fits is a 2d image
        kcube = cube.to(u.K)
        kcube.write(newname, format='fits', overwrite=True)
        print("\n\033[92mSAVED FILE:\033[0m '{}'".format(newname))
    data.close()


def find_common_beam(filenames):
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


def spatial_smooth(filename, beam=None, major=None, minor=None, pa=0, path_to_output='.', suffix='', allow_huge_operations=False, datatype='regular', **kwargs): # smooth image with 2D Gaussian
    try:
        cube = SpectralCube.read(filename)
    except:
        cube = Projection.from_hdu(fits.open(filename)[0])
    if beam is None:
        if major is None or minor is None:
            raise ValueError('Need to specify beam size if no beam is given.')
        if minor is None:
            print('No minor beam size was given, will assume major=minor')
        beam = radio_beam.Beam(major=major*u.arcsec, minor=minor*u.arcsec, pa=pa*u.deg)
    elif beam is not None:
        beam = beam
        major = np.around(beam.major.value * 3600, decimals=2)
        minor = np.around(beam.minor.value * 3600, decimals=2)
    meanbeam = np.sqrt(major*minor) # geometric mean

    filename_wext = os.path.basename(filename)
    filename_base, file_extension = os.path.splitext(filename_wext)
    newname = filename_base + '_smooth' + str(meanbeam) + '_arcsec' + suffix + '.fits'
    pathname = os.path.join(path_to_output, newname)

    if allow_huge_operations:
        cube.allow_huge_operations = True
    if datatype=='large':
        shutil.copy(filename, pathname)
        outfh = fits.open(newname, mode='update')
        with tqdm(total=cube.shape[0]) as pbar:
            for index in range(cube.shape[0]):
                smooth_slice = cube[index].convolve_to(beam, **kwargs)
                outfh[0].data[index] = smooth_slice.array
                outfh.flush() # write the data to disk
                pbar.update(1)
        outfh[0].header.update(beam.to_header_keywords())
        outfh[0].header.update({'BEAM' : f'Beam: BMAJ={beam.major.value:.1f} arcsec BMIN={beam.minor.value:.1f} arcsec BPA={beam.pa.value:.1f} deg'})
        outfh.flush()
        print("\n\033[92mSAVED FILE:\033[0m '{}'".format(newname))
    else:
        smoothcube = cube.convolve_to(beam, **kwargs)
        smoothcube.write(pathname, format='fits', overwrite=True)
        print("\n\033[92mSAVED FILE:\033[0m '{}' in '{}'".format(newname,path_to_output))

	
def reproject_cube(filename, template, axes='spatial', path_to_output='.', suffix='', allow_huge_operations=False, datatype='regular'):
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

    filename_wext = os.path.basename(filename)
    filename_base, file_extension = os.path.splitext(filename_wext)
    newname = filename_base + '_reproject' + suffix + '.fits'
    pathname = os.path.join(path_to_output, newname)

    #TODO
    if allow_huge_operations:
        cube1.allow_huge_operations = True
    if datatype=='large':
        if axes=='spatial':
            shutil.copy(template, pathname)
            outfh = fits.open(newname, mode='update')
            header_template = md_header_2d(header_template)
            with tqdm(total=cube2.shape[0]) as pbar:
                for index in range(cube2.shape[0]):
                    cube_slice_reproj = cube1[index].reproject(header_template)
                    outfh[0].data[index] = cube_slice_reproj.array
                    outfh.flush() # write the data to disk
                    pbar.update(1)
            outfh.flush()
            print("\n\033[92mSAVED FILE:\033[0m '{}'".format(newname))
    else:
        cube1_reproj = cube1.reproject(header_template)

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


def estimate_cont_noise(data, inner_pct, outer_pct):
    #global inner_pct, outer_pct
    y_size = data.shape[0]
    x_size = data.shape[1]
    central_px_x = x_size/2.
    central_px_y = y_size/2.
    px_array, indices = pixel_ellipse_annulus_calculation(central_px_x, central_px_y, inner_pct*x_size/2., outer_pct*x_size/2., inner_pct*y_size/2., outer_pct*y_size/2.)

    # only select data within that annulus region
    data_in_annulus = data[indices]

    # check if annulus contains nan values. If so, decrease inner and outer radii by some value and estimate_cont_noise again
    if np.any(np.isnan(data_in_annulus)):
        print('\n\033[93mRegion contains NaNs!\033[0m\nMaking region smaller now...')
        inner_pct -= 0.05
        outer_pct -= 0.05
        return estimate_cont_noise(data, inner_pct, outer_pct)
    else:
        # estimate rms and std in that region
        std = np.std(data_in_annulus)
        rms = np.sqrt(np.mean(data_in_annulus**2))
        print('\nstd, rms = {}, {}'.format(std,rms))
        print('\nwith inner and outer pct: {}, {}'.format(inner_pct,outer_pct))
        print('\n\033[92mJONAS IS AWESOME\033[0m')
        return std, rms
