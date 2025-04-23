import pytest

# List of dependencies
dependencies = [
    "affine",
    "astropy",
    "bs4",
    "colored_logging",
    "descartes",
    "ephem",
    "geopandas",
    "GEOS5FP",
    "gedi_canopy_height",
    "harmonized_landsat_sentinel",
    "h5py",
    "imageio",
    "jdcal",
    "jupyter",
    "keras",
    "koppengeiger",
    "matplotlib",
    "nose",
    "numpy",
    "pycksum",
    "pygrib",
    "pykdtree",
    "pyresample",
    "PIL",
    "pysolar",
    "pytest",
    "rasters",
    "rasterio",
    "requests",
    "skimage",
    "scipy",
    "sentinelsat",
    "six",
    "shapely",
    "solar_apparent_time",
    "spacetrack",
    "sun_angles",
    "termcolor",
    "tensorflow",
    "untangle",
    "urllib3",
    "wget",
    "xmltodict"
]

# Generate individual test functions for each dependency
@pytest.mark.parametrize("dependency", dependencies)
def test_dependency_import(dependency):
    __import__(dependency)
