"""Public d3map API.

The validated numerical implementation remains in the ``dmap`` compatibility
namespace during the 1.0 release series.  This module is the canonical public
entry point and deliberately re-exports that API while the internal package is
migrated without duplicating scientific kernels.
"""

from dmap import *
from dmap import __all__ as _legacy_all
from dmap import __version__ as __version__

__all__: list[str] = list(_legacy_all)
__all__.append("__version__")
