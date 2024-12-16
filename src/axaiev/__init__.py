from .release import __version__
import sys

# the package is imported during installation (to obtain the version)
# however installation happens in an isolated build environment
# where no dependencies are installed.
# the following try-except block ensures that the installation works
# (where dependencies are not available)

try:

    from . import core

    # This mechanism serves to simplify the co-development of the package
    # and a jupyter notebook: one reload-statement is then sufficient
    # to also reload submodules

    attr_name = f"_package_{__name__}_is_imported"

    if getattr(sys, attr_name, False):
        # print(f"reloading package {__name__}")
        import importlib as il
        il.reload(core)
    else:
        # print(f"importing package {__name__}")
        setattr(sys, attr_name, True)

    from .core import *
except ImportError:
    import os
    if "PIP_BUILD_TRACKER" in os.environ:
        pass
    else:
        # raise the original exception
        raise
