from .seissolxdmfwriter import *

try:
    from importlib.metadata import PackageNotFoundError, version
except ImportError:
    from importlib_metadata import PackageNotFoundError, version

try:
    __version__ = version("seissolxdmfwriter")
except PackageNotFoundError:
    # If the package is not installed, don't add __version__
    pass
