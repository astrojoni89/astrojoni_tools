import numpy as np

def tau_hisa(t_hisa, p, t_on, t_off, t_cont):
    return -np.log(1-((t_on-t_off)/(t_hisa-t_cont-p*t_off)))

def hi_coldens(t_s, tau, dv):
    return 1.8224 * 10**18 * t_s * tau * dv

def t_hisa_max(p, t_on, t_off, t_cont):
    return t_on + t_cont - (1-p) * t_off
