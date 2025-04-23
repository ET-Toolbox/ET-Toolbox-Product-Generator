import pytest

# List of dependencies
dependencies = [
    "affine",
    "bs4",
    "colored_logging",
    "ephem",
    "geopandas",
    "GEOS5FP",
    "gedi_canopy_height",
    "harmonized_landsat_sentinel",
    "h5py",
    "koppengeiger",
    "matplotlib",
    "nose",
    "numpy",
    "pycksum",
    "pygrib",
    "rasters",
    "rasterio",
    "requests",
    "skimage",
    "scipy",
    "shapely",
    "solar_apparent_time",
    "spacetrack",
    "sun_angles",
    "urllib3",
    "xmltodict"
]

# Generate individual test functions for each dependency
@pytest.mark.parametrize("dependency", dependencies)
def test_dependency_import(dependency):
    __import__(dependency)
