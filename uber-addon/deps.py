import pip

try:
    import funcy
    import cachetools
    import tetgen
    import pymesh
    import meshpy
except ImportError:
    pip.main(['install', 'funcy', 'cachetools', 'tetgen', 'pymesh', 'meshpy', 'pybind11'])
