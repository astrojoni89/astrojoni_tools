import numpy as np
from astropy import constants as const


def tau_hisa(t_hisa, p, t_on, t_off, t_cont):
    """Compute the optical depth of HI self-absorption
    given a HISA spin temperature, background fraction, on and off brightness,
    and the continuum brightness.
    
    Parameters
    ----------
    t_hisa : float or numpy.ndarray
        Spin temperature of cold HI gas that produces self-absorption.
    p : float or numpy.ndarray
        Background fraction of HI gas that induces self-absorption.
    t_on : float or numpy.ndarray
        Brightness temperature of HI gas if self-absorption is present.
    t_off : float or numpy.ndarray
        Brightness temperature of HI gas if self-absorption were absent.
    t_cont : float or numpy.ndarray
        Brightness of continuum emission.
    Returns
    -------
    tau_hisa : float or numpy.ndarray
        Optical depth of HI self-absorbing gas.
    """
    tau_hisa = -np.log(1-((t_on-t_off)/(t_hisa-t_cont-p*t_off)))
    return tau_hisa

def hi_coldens(t_s, tau, dv):
    """Compute the column density of atomic hydrogen (HI)
    given a spin temperature, optical depth and velocity resolution element.
    
    Parameters
    ----------
    t_s : float or numpy.ndarray
        Spin temperature of HI gas [K].
    tau : float or numpy.ndarray
        Optical depth of HI gas.
    dv : float or numpy.ndarray
        Velocity resolution element [km/s].
    Returns
    -------
    n_HI : float or numpy.ndarray
        Column density of HI [cm-2].
    """
    n_HI = 1.8224 * 10**18 * t_s * tau * dv
    return n_HI

def t_hisa_max(p, t_on, t_off, t_cont):
    """Compute the maximum spin temperature of HI self-absorbing gas
    that would still give a solution to the optical depth computation.
    This limit is reached if the optical depth is set to infty.
    
    Parameters
    ----------
    p : float or numpy.ndarray
        Background fraction of HI gas that induces self-absorption.
    t_on : float or numpy.ndarray
        Brightness temperature of HI gas if self-absorption is present.
    t_off : float or numpy.ndarray
        Brightness temperature of HI gas if self-absorption were absent.
    t_cont : float or numpy.ndarray
        Brightness of continuum emission.
    Returns
    -------
    t_spin_max : float or numpy.ndarray
        Maximum spin temperature of HI self-absorbing gas.
    """
    t_spin_max = t_on + t_cont - (1-p) * t_off
    return t_spin_max

def calculate_gal_radius_from_distance(distance,longitude,latitude,R_sun=8.15):
    """Compute the Galactocentric distance of a source
    given its distance from the sun, Galactic longitude, Galactic latitude,
    and the Galactocentric distance of the sun.
    
    Parameters
    ----------
    distance : float or numpy.ndarray
        Heliocentric distance of the source [kpc].
    longitude : float or numpy.ndarray
        Galactic longitude of the source [deg].
    latitude : float or numpy.ndarray
        Galactic latitude of the source [deg].
    R_sun : float, optional
        The Galactocentric distance of the sun [kpc].
        The default is R_sun=8.15 (Reid et al. 2019).
    Returns
    -------
    R_gal_distance : float or numpy.ndarray
        The Galactocentric distance of the source [kpc].
    """
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

def thermal_linewidth(t_kin, mu=1.27):
    """Compute the thermal linewidth of a chemical species.
    
    Parameters
    ----------
    t_kin : float or numpy.ndarray
        Gas kinetic temperature [Kelvin].
    mu : float or numpy.ndarray
        Mean molecular weight. Default is '1.27' [atomic hydrogen].
        Another prominent example is: '2.34' [carbon monoxide]. 
    Returns
    -------
    thermal_lw : float or numpy.ndarray
        The thermal linewidth of the gas given in units of sigma [km/s].
        To convert to FWHM: thermal_lw * np.sqrt(8*np.log(2)).
    """
    kB = const.k_B
    mp = const.u
    thermal_lw = np.sqrt((kB* t_kin)/(mu*mp)) /1000
    return thermal_lw.value

def radial_velocity(l,r,rotvel=220.,v0=220.,r0=8.5):
    """Compute the radial velocity of circular motion around the Galactic center.
    
    Parameters
    ----------
    l : float or numpy.ndarray
        Galactic longitude in degrees.
    r : float
        Galactocentric distance of source in units of kiloparsec. 
    rotcurve : float
        Value of rotation velocity at distance r.
    v0 : float
        Rotational velocity of the sun in units of km/s. Default is v0=220.
    r0 : float
        Galactocentric distance of sun in units of kiloparsec. Default is r0=8.5.
    Returns
    -------
    vrad : float or numpy.ndarray
        The radial velocity given the rotation velocity at distance r and longitude l.
    """
    r = r*3.09e16
    r0 = r0*3.09e16
    vrad = v0*r0* ((rotvel/(v0*r)) - 1/r0) * np.sin(np.radians(l))
    return vrad

def rotation_curve(r,r0=8.5,a1=1.00767,a2=0.0394,a3=0.00712):
    """Compute the rotational velocity at distance r from the Galactic center.
    This rotational velocity is computed according to the Brand&Blitz (1993) rotation curve.
    
    Parameters
    ----------
    r : float
        Galactocentric distance of source in units of kiloparsec. 
    r0 : float
        Galactocentric distance of sun in units of kiloparsec. Default is r0=8.5.
    a1 : float
        Fit parameter of the Brand&Blitz rotation curve.
    a2 : float
        Fit parameter of the Brand&Blitz rotation curve.
    a3 : float
        Fit parameter of the Brand&Blitz rotation curve.
    Returns
    -------
    v_over_v0 : float
        The rotational velocity at distance r from the Galactic center in units of the sun's rotational velocity, which is 220 km/s in this model.
    """
    r = r*3.09e16
    r0 = r0*3.09e16
    v_over_v0 = a1 * (r/r0)**a2 + a3
    return v_over_v0
