from .base import SeisSolReaderBase
from .seissolxdmf import seissolxdmf
from .vtkhdf import VTKHDFReader, VTKHDFSeries

try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:
    from importlib_metadata import version, PackageNotFoundError

try:
    __version__ = version("seissolxdmf")
except PackageNotFoundError:
    # If the package is not installed, don't add __version__
    pass
