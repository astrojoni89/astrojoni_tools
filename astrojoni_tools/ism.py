import numpy as np

def tau_hisa(t_hisa, p, t_on, t_off, t_cont):
    return -np.log(1-((t_on-t_off)/(t_hisa-t_cont-p*t_off)))

def hi_coldens(t_s, tau, dv):
    return 1.8224 * 10**18 * t_s * tau * dv

def t_hisa_max(p, t_on, t_off, t_cont):
    return t_on + t_cont - (1-p) * t_off

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

