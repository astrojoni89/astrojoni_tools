import pkg_resources
from astropy.table import Table, Column


def load_data(filename, model='Reid2019'):
    """Return an astropy.Table containing spiral arm data.
    """
    # This is a stream-like object. If you want the actual info, call
    # stream.read()
    stream = pkg_resources.resource_stream(__name__, 'data/{}/{}'.format(model,filename))
    return Table.read(stream, header_start=0, data_start=1, format='ascii')
