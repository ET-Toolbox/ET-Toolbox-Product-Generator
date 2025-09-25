import pytest

# List of dependencies
dependencies = [
    "check_distribution",
    "colored_logging",
    "gedi_canopy_height",
    "GEOS5FP",
    "global_forecasting_system",
    "harmonized_landsat_sentinel",
    "LandsatL2C2",
    "MODISCI",
    "NASADEM",
    "PTJPL",
    "rasters",
    "sentinel_tiles",
    "soil_capacity_wilting",
    "solar_apparent_time",
    "VNP09GA_002",
    "VNP21A1D_002"
]

# Generate individual test functions for each dependency
@pytest.mark.parametrize("dependency", dependencies)
def test_dependency_import(dependency):
    __import__(dependency)
