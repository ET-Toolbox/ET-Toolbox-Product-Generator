"""
This module contains the unit tests for GeoTIFF compatibility.
"""

import os
import unittest

__author__ = "Gregory Halverson"


class TestGeoTIFF(unittest.TestCase):
    FLOATING_POINT_ERROR_TOLERANCE = 0.0000001
    GEOTIFF_TEST_FILE = 'geotiff_test_file.tif'
    # TODO should also test for EPSG code translation
    GEOTIFF_CRS = '+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs'

    def test_io_geotiff(self):
        import numpy as np
        import rasterio
        from affine import Affine
        from pyproj import Proj

        # generate coordinate vectors
        lat_vector = np.linspace(90, -90, num=180 + 1)[1:-1]
        lon_vector = np.linspace(-180, 180, num=360 + 1)[1:-1]

        # calculate matrix shape
        rows = len(lat_vector)
        cols = len(lon_vector)

        # generate matrix of normal floats
        data = np.random.randn(rows, cols)

        pixel_width = 1
        upper_left_x = -180
        pixel_height = -1
        upper_left_y = 90

        affine = Affine(
            pixel_width,
            0,
            upper_left_x,
            0,
            pixel_height,
            upper_left_y
        )

        # remove test file if it already exists
        if os.path.exists(self.GEOTIFF_TEST_FILE):
            os.remove(self.GEOTIFF_TEST_FILE)

        profile = {
            'driver': 'GTiff',
            'width': cols,
            'height': rows,
            'count': 1,
            'crs': self.GEOTIFF_CRS,
            'transform': affine,
            'dtype': np.float32
        }

        with rasterio.open(self.GEOTIFF_TEST_FILE, 'w', **profile) as geotiff_file:
            geotiff_file.write(data.astype(np.float32), 1)

        with rasterio.open(self.GEOTIFF_TEST_FILE, 'r') as geotiff_file:
            retrieved_crs = Proj(geotiff_file.crs).srs
            retrieved_data = geotiff_file.read(1)

        # check that the same data was both written and retrieved
        assert ((retrieved_data - data).mean() < self.FLOATING_POINT_ERROR_TOLERANCE)

        # delete test file
        if os.path.exists(self.GEOTIFF_TEST_FILE):
            os.remove(self.GEOTIFF_TEST_FILE)
