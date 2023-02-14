import numpy as np

def tau_hisa(t_hisa, p, t_on, t_off, t_cont):
    """This function returns the optical depth of HI self-absorption
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
    """This function returns the column density of atomic hydrogen (HI)
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
    """This function returns the maximum spin temperature of HI self-absorbing gas
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

def calculate_gal_radius_from_distance(distance,longitude,latitude,R_sun=8.15): # 8.15 kpc from Reid et al. (2019)
    """This function returns the Galactocentric distance of a source
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

