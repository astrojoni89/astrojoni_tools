import os
import random
import numpy as np

import matplotlib.pyplot as plt

from astropy.io import fits
from astropy import units as u
from tqdm import trange

from .functions import find_nearest, velocity_axes, pixel_to_world, pixel_circle_calculation, pixel_circle_calculation_px, calculate_spectrum


### SCALEBAR PLOTTING IMSHOW
def plot_scalebar(length, wcs, distance_of_source, ax=None, loc='bottom right', labelcolor='white', labelsize='small', offset=0.05, unit='pc', **kwargs):
    """This function plots a scalebar onto an existing figure axis.
    
    Parameters
    ----------
    length : float
        Length (in units of a projected physical scale; like '1.' [AU/pc/ly]) of scalebar that is plotted on the axis object. Should have the same unit as 'distance_of_source'.
    wcs : str or :class:`~astropy.wcs.WCS`
        Path to FITS file or WCS instance.
    distance_of_source : float
        Distance of the plotted source. Should have the same unit as 'length'.
    ax : None or :class:`~astropy.visualization.wcsaxes.WCSAxes`
        WCSAxes instance in which the scalebar is displayed. The WCS must be celestial.
    loc : str
        Location of scalebar. The default is 'bottom right'.
    labelcolor : str
        Color of scalebar label.
    labelsize : str or float
        Fontsize of the label.
    offset : float, optional
        Offset between scalebar and corresponding label. Given in units of axis fraction.
    unit : str
        Unit of scalebar label that is plotted.
    **kwargs
        Additional arguments are passed to :class:`~astropy.visualization.wcsaxes.WCSAxes.plot` of the actual scalebar.
    """
    loc_dict = {
        'bottom left': [0.03,0.15],
        'top left': [0.03,0.95],
        'bottom right': [0.97,0.15],
        'top right': [0.97,0.95]
    }
    from astropy.wcs import WCS
    if isinstance(wcs, WCS):
        header = wcs.to_header()
    elif isinstance(wcs, str):
        header = fits.getheader(wcs)
    # pixel scale
    degppx = abs(header['CDELT1']) # deg per pixel; assuming square pixels
    distance = distance_of_source # arbitrary unit u
    # parsec per pixel
    pxscale = np.sin(np.radians(degppx)) * distance
    # length of scalebar (as well in unit u)
    pxscalebar = int(np.around(length / pxscale,decimals=0))
    
    if ax is None:
        ax = plt.gca()
    axis_to_data = ax.transAxes + ax.transData.inverted()
    
    if loc not in loc_dict:
        raise KeyError('Possible locations are {}'.format(loc_dict.keys()))
    points_axis = loc_dict[loc]
    offset_label = [points_axis[0], points_axis[1] - offset]
    points_data = axis_to_data.transform(points_axis)
    offset_label_data = axis_to_data.transform(offset_label)
    
    if loc in ['bottom right', 'top right']:
        x = np.arange(points_data[0]-pxscalebar, points_data[0])
        x_label = points_data[0]-pxscalebar/2.
    elif loc in ['bottom left', 'top left']:
        x = np.arange(points_data[0], points_data[0]+pxscalebar)
        x_label = points_data[0]+pxscalebar/2.
    y = np.ones_like(x) * points_data[1]
    
    ax.plot(x,y,transform=ax.get_transform('world'),**kwargs)
    
    # label depends on loc
    if unit == 'pc' or unit == 'parsec':
        ax.text(x_label, offset_label_data[1], '{} pc'.format(length), color=labelcolor, ha='center', va='top', family='serif', size=labelsize)
    elif unit == 'au' or unit == 'AU':
        ax.text(x_label, offset_label_data[1], '{} AU'.format(length), color=labelcolor, ha='center', va='top', family='serif', size=labelsize)
    elif unit == 'ly' or unit == 'lightyear':
        ax.text(x_label, offset_label_data[1], '{} ly'.format(length), color=labelcolor, ha='center', va='top', family='serif', size=labelsize)
    elif unit == 'Lichtjahr':
        ax.text(x_label, offset_label_data[1], '{} Lichtjahre'.format(length), color=labelcolor, ha='center', va='top', family='serif', size=labelsize)

    
### SCALEBAR PLOTTING IMSHOW
def add_scalebar(length, wcs, distance_of_source, ax=None, loc='bottom right', frame=False, borderpad=0.4, pad=0.5, unit='pc', **kwargs):
    """This function plots a scalebar onto an existing figure axis.
    
    Parameters
    ----------
    length : float
        Length (in units of a projected physical scale; like '1.' [AU/pc/ly]) of scalebar that is plotted on the axis object. Should have the same unit as 'distance_of_source'.
    wcs : str or :class:`~astropy.wcs.WCS`
        Path to FITS file or WCS instance.
    distance_of_source : float
        Distance of the plotted source. Should have the same unit as 'length'.
    ax : None or :class:`~astropy.visualization.wcsaxes.WCSAxes`
        WCSAxes instance in which the scalebar is displayed. The WCS must be celestial.
    loc : str
        Location of scalebar. The default is 'bottom right'.
    labelcolor : str
        Color of scalebar label.
    labelsize : str or float
        Fontsize of the label.
    offset : float, optional
        Offset between scalebar and corresponding label. Given in units of axis fraction.
    unit : str
        Unit of scalebar label that is plotted.
    **kwargs
        Additional arguments are passed to :class:`~astropy.visualization.wcsaxes.WCSAxes.plot` of the actual scalebar.
    """
    from astropy.wcs import WCS
    if isinstance(wcs, WCS):
        header = wcs.to_header()
    elif isinstance(wcs, str):
        header = fits.getheader(wcs)
    # pixel scale
    degppx = abs(header['CDELT1']) # deg per pixel; assuming square pixels
    distance = distance_of_source # arbitrary unit u
    # parsec per pixel
    pxscale = np.sin(np.radians(degppx)) * distance
    # length of scalebar (as well in unit u)
    pxscalebar = int(np.around(length / pxscale,decimals=0))
    
    if ax is None:
        ax = plt.gca()
    axis_to_data = ax.transAxes + ax.transData.inverted()
   
    # label depends on loc
    if unit == 'pc' or unit == 'parsec':
        label = '{} pc'.format(length)
    elif unit == 'au' or unit == 'AU':
        label = '{} AU'.format(length)
    elif unit == 'ly' or unit == 'lightyear':
        label = '{} ly'.format(length)
    elif unit == 'Lichtjahr':
        label = '{} Lichtjahre'.format(length)

    scalebar = AnchoredSizeBar(
        ax.transData,
        pxscalebar,
        label,
        loc,
        pad=pad,
        borderpad=borderpad,
        sep=5,
        frameon=frame,
        **kwargs,
    )

    ax.add_artist(scalebar)

### PLOTTING TOOL FOR AVERAGED SPECTRA
def styles():
    color_list = ['k', 'b', 'b', 'r', 'g']
    draw_list = ['steps-mid', 'default', 'steps-mid', 'steps-mid', 'steps-mid']
    line_list = ['-', '--', '-', '-', '-']
    return color_list, draw_list, line_list

def get_figure_params(n_spectra, rowsize, rowbreak):
    colsize = 1.3 * rowsize
    cols = int(np.sqrt(n_spectra))
    rows = int(n_spectra / (cols))
    if n_spectra % cols != 0:
        rows += 1
    if rows < rowbreak:
        rowbreak = rows
    if (rowbreak*rowsize*100 > 2**16) or (cols*colsize*100 > 2**16):
        errorMessage = \
            "Image size is too large. It must be less than 2^16 pixels in each direction. Restrict the number of columns or rows."
        raise Exception(errorMessage)

    return cols, rows, rowbreak, colsize


def xlabel_from_header(header, vel_unit):
    xlabel = 'Channels'

    if header is None:
        return xlabel

    if 'CTYPE3' in header.keys():
        xlabel = '{} [{}]'.format(header['CTYPE3'], vel_unit)

    return xlabel


def ylabel_from_header(header):
    if header is None:
        return 'Intensity'

    btype = 'Intensity'
    if 'BTYPE' in header.keys():
        btype = header['BTYPE']

    bunit = ''
    if 'BUNIT' in header.keys():
        bunit = ' [{}]'.format(header['BUNIT'])

    return btype + bunit


def add_figure_properties(ax, header=None, fontsize=10, velocity_range=None, vel_unit=u.km/u.s):
    ax.set_xlim(np.amin(velocity_range), np.amax(velocity_range))
    #ax.set_ylim()
    ax.set_xlabel(xlabel_from_header(header, vel_unit), fontsize=fontsize)
    ax.set_ylabel(ylabel_from_header(header), fontsize=fontsize)

    ax.tick_params(labelsize=fontsize - 2)

    
def scale_fontsize(rowsize):
    rowsize_scale = 4
    if rowsize >= rowsize_scale:
        fontsize = 10 + int(rowsize - rowsize_scale)
    else:
        fontsize = 10 - int(rowsize - rowsize_scale)
    return fontsize


def plot_spectra(fitsfiles, outfile='spectra.pdf', coordinates=None, radius=None, path_to_plots='.', n_spectra=9, rowsize=4., rowbreak=10, dpi=72, velocity_range=None, vel_unit=u.km/u.s):
    '''
    fitsfiles: list of fitsfiles to plot spectra from
    coordinates: array of central coordinates [[Glon, Glat]] to plot spectra from
    radius: radius of area to be averaged for each spectrum [arcseconds]
    '''
    
    print("\nPlotting...")
    
    fontsize = scale_fontsize(rowsize)
    color_list, draw_list, line_list = styles()
    
    if coordinates is not None:
        n_spectra = len(coordinates)
        cols, rows, rowbreak, colsize = get_figure_params(n_spectra, rowsize, rowbreak)
        figsize = (cols*colsize, rowbreak*rowsize)
        fig = plt.figure(figsize=figsize)
        
        if radius is not None:
            for i in trange(len(coordinates)):
                ax = fig.add_subplot(rows,cols,i+1)
                for idx, fitsfile in enumerate(fitsfiles):
                    pixel_array, _ = pixel_circle_calculation(fitsfile,coordinates[i,0],coordinates[i,1],radius)
                    spectrum, _ = calculate_spectrum(fitsfile,pixel_array)
                    header = fits.getheader(fitsfile)
                    velocity = velocity_axes(fitsfile)
                    if velocity_range is not None:
                        velo_min, velo_max = find_nearest(velocity,np.amin(velocity_range)), find_nearest(velocity,np.amax(velocity_range))
                    else:
                        velo_min, velo_max = 0, velocity.shape[0]
                    ax.plot(velocity[velo_min:velo_max], spectrum[velo_min:velo_max], drawstyle=draw_list[idx], color=color_list[idx], linestyle=line_list[idx])
                add_figure_properties(ax, header=header, fontsize=fontsize, velocity_range=velocity_range, vel_unit=vel_unit)
                plt.annotate('Glon: {} deg\nGlat: {} deg'.format(round(coordinates[i][0],2),round(coordinates[i][1],2)), xy=(0.05, 0.85), xycoords='axes fraction', fontsize=fontsize)
                

        else:
            for i in trange(len(coordinates)):
                ax = fig.add_subplot(rows,cols,i+1)
                for idx, fitsfile in enumerate(fitsfiles):
                    header = fits.getheader(fitsfile)
                    beam = header['BMAJ']
                    radius = 1/2. * (beam*3600)
                    pixel_array, _ = pixel_circle_calculation(fitsfile,coordinates[i,0],coordinates[i,1],radius)
                    spectrum, _ = calculate_spectrum(fitsfile,pixel_array)
                    velocity = velocity_axes(fitsfile)
                    if velocity_range is not None:
                        velo_min, velo_max = find_nearest(velocity,np.amin(velocity_range)), find_nearest(velocity,np.amax(velocity_range))
                    else:
                        velo_min, velo_max = 0, velocity.shape[0]
                    ax.plot(velocity[velo_min:velo_max], spectrum[velo_min:velo_max], drawstyle=draw_list[idx], color=color_list[idx], linestyle=line_list[idx])
                add_figure_properties(ax, header=header, fontsize=fontsize, velocity_range=velocity_range, vel_unit=vel_unit)
                plt.annotate('Glon: {} deg\nGlat: {} deg'.format(round(coordinates[i][0],2),round(coordinates[i][1],2)), xy=(0.05, 0.85), xycoords='axes fraction', fontsize=fontsize)

    else:
        random.seed(111)
        xsize = fits.getdata(fitsfiles[0]).shape[2]
        ysize = fits.getdata(fitsfiles[0]).shape[1]
        cols, rows, rowbreak, colsize = get_figure_params(n_spectra, rowsize, rowbreak)
        figsize = (cols*colsize, rowbreak*rowsize)
        fig = plt.figure(figsize=figsize)

        if radius is not None:
            for i in trange(n_spectra):
                temp_header = fits.getheader(fitsfiles[0])
                px_scale = abs(temp_header['CDELT1'])
                edge = int(np.ceil((radius/3600) / px_scale))
                xValue = random.randint(edge+1,xsize-edge-1)
                yValue = random.randint(edge+1,ysize-edge-1)
                ax = fig.add_subplot(rows,cols,i+1)
                for idx, fitsfile in enumerate(fitsfiles):
                    pixel_array = pixel_circle_calculation_px(fitsfile,x=xValue,y=yValue,r=radius)
                    spectrum, _ = calculate_spectrum(fitsfile,pixel_array)
                    header = fits.getheader(fitsfile)
                    velocity = velocity_axes(fitsfile)
                    if velocity_range is not None:
                        velo_min, velo_max = find_nearest(velocity,np.amin(velocity_range)), find_nearest(velocity,np.amax(velocity_range))
                    else:
                        velo_min, velo_max = 0, velocity.shape[0]
                    ax.plot(velocity[velo_min:velo_max], spectrum[velo_min:velo_max], drawstyle=draw_list[idx], color=color_list[idx], linestyle=line_list[idx])
                add_figure_properties(ax, header=header, fontsize=fontsize, velocity_range=velocity_range, vel_unit=vel_unit)
                coordinate = pixel_to_world(fitsfiles[0],xValue,yValue)
                plt.annotate('Glon: {} deg\nGlat: {} deg'.format(round(coordinate[0].item(0),2),round(coordinate[1].item(0),2)), xy=(0.05, 0.85), xycoords='axes fraction', fontsize=fontsize)

        else:
            for i in trange(n_spectra):
                temp_header = fits.getheader(fitsfiles[0])
                px_scale = abs(temp_header['CDELT1'])
                temp_beam = temp_header['BMAJ']
                temp_radius = 1/2. * temp_beam
                edge = int(np.ceil(temp_radius / px_scale))
                xValue = random.randint(edge+1,xsize-edge-1)
                yValue = random.randint(edge+1,ysize-edge-1)
                ax = fig.add_subplot(rows,cols,i+1)
                for idx, fitsfile in enumerate(fitsfiles):
                    header = fits.getheader(fitsfile)
                    beam = header['BMAJ']
                    radius = 1/2. * (beam*3600)
                    pixel_array = pixel_circle_calculation_px(fitsfile,x=xValue,y=yValue,r=radius)
                    spectrum, _ = calculate_spectrum(fitsfile,pixel_array)
                    velocity = velocity_axes(fitsfile)
                    if velocity_range is not None:
                        velo_min, velo_max = find_nearest(velocity,np.amin(velocity_range)), find_nearest(velocity,np.amax(velocity_range))
                    else:
                        velo_min, velo_max = 0, velocity.shape[0]
                    ax.plot(velocity[velo_min:velo_max], spectrum[velo_min:velo_max], drawstyle=draw_list[idx], color=color_list[idx], linestyle=line_list[idx])
                add_figure_properties(ax, header=header, fontsize=fontsize, velocity_range=velocity_range, vel_unit=vel_unit)
                coordinate = pixel_to_world(fitsfiles[0],xValue,yValue)
                plt.annotate('Glon: {} deg\nGlat: {} deg'.format(round(coordinate[0].item(0),2),round(coordinate[1].item(0),2)), xy=(0.05, 0.85), xycoords='axes fraction', fontsize=fontsize)

    #for axs in fig.axes:
        #axs.label_outer()
    fig.tight_layout()

    if not os.path.exists(path_to_plots):
        os.makedirs(path_to_plots)
    filename = outfile
    pathname = os.path.join(path_to_plots, filename)
    fig.savefig(pathname, dpi=dpi, bbox_inches='tight')
    #plt.close()
    print("\n\033[92mSAVED FILE:\033[0m '{}' in '{}'".format(filename, path_to_plots))
