"""
This module contains the unit tests for the rasterio package.
"""

import os
import unittest

__author__ = "Gregory Halverson"

directory = os.path.abspath(os.path.dirname(__file__))


class TestRasterio(unittest.TestCase):
    def test_import_rasterio(self):
        print('testing rasterio import')
        import rasterio as rio
        print("rasterio version: {}".format(rio.__version__))


