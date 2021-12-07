import pip

try:
    import funcy
    import cachetools
except ImportError:
    pip.main(['install', 'funcy', 'cachetools'])
