import numpy as np

def tau_hisa(T_hisa, p, T_on, T_off, T_cont):
    return -np.log(1-((T_on-T_off)/(T_hisa-T_cont-p*T_off)))

def hi_coldens(T_s, tau, dv):
    return 1.8224 * 10**18 * T_s * tau * dv

def T_hisa_max(p, T_on, T_off, T_cont):
    return T_on + T_cont - (1-p) * T_off
