import pkg_resources
import glob
from astropy.table import Table, Column


def get_file_names(model='Reid2019'):
    """Load the spiral arm file names.
    """
    stream = pkg_resources.resource_stream(__name__, 'data/{}/*.dat'.format(model))
    return glob.glob(stream)
    
    
def load_data(filename, model='Reid2019'):
    """Return an astropy.Table containing spiral arm data.
    """
    # This is a stream-like object. If you want the actual info, call
    # stream.read()
    stream = pkg_resources.resource_stream(__name__, 'data/{}/{}'.format(model,filename))
    return Table.read(stream, header_start=0, data_start=1, format='ascii')
