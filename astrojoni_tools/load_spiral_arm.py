import pkg_resources
import glob
import pandas as pd
from astropy.table import Table, Column


def get_file_names(model='Reid2019'):
    """Load the spiral arm file names.
    """
    stream = pkg_resources.resource_stream(__name__, 'data/{}/filenames.txt'.format(model))
    df = pd.read_table(stream)
    return df['names'].to_list()
    
    
def load_data(filename, model='Reid2019'):
    """Return an astropy.Table containing spiral arm data.
    """
    # This is a stream-like object. If you want the actual info, call
    # stream.read()
    stream = pkg_resources.resource_stream(__name__, 'data/{}/{}'.format(model,filename))
    return Table.read(stream, header_start=0, data_start=1, format='ascii')
