import numpy as np
from astropy import units as u
from astropy.wcs import WCSSUB_SPECTRAL


def sanitize_wcs(mywcs):
    pc = np.matrix(mywcs.wcs.get_pc())
    if (pc[:,2].sum() != pc[2,2] or pc[2,:].sum() != pc[2,2]):
        raise ValueError("Non-independent 3rd axis.")
    axtypes = mywcs.get_axis_types()
    if ((axtypes[0]['coordinate_type'] != 'celestial' or
         axtypes[1]['coordinate_type'] != 'celestial' or
         axtypes[2]['coordinate_type'] != 'spectral')):
        cunit3 = mywcs.wcs.cunit[2]
        ctype3 = mywcs.wcs.ctype[2]
        if cunit3 != '':
            cunit3 = u.Unit(cunit3)
            if cunit3.is_equivalent(u.m/u.s):
                mywcs.wcs.ctype[2] = 'VELO'
            elif cunit3.is_equivalent(u.Hz):
                mywcs.wcs.ctype[2] = 'FREQ'
            elif cunit3.is_equivalent(u.m):
                mywcs.wcs.ctype[2] = 'WAVE'
            else:
                raise ValueError("Could not determine type of 3rd axis.")
        elif ctype3 != '':
            if 'VELO' in ctype3:
                mywcs.wcs.ctype[2] = 'VELO'
            elif 'FELO' in ctype3:
                mywcs.wcs.ctype[2] = 'VELO-F2V'
            elif 'FREQ' in ctype3:
                mywcs.wcs.ctype[2] = 'FREQ'
            elif 'WAVE' in ctype3:
                mywcs.wcs.ctype[2] = 'WAVE'
            else:
                raise ValueError("Could not determine type of 3rd axis.")
        else:
            raise ValueError("Cube axes not in expected orientation: PPV")
    return mywcs
